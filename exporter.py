from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from processing.data_model import DataStream


def stream_to_dataframe(stream: DataStream) -> pd.DataFrame:
    """Convert all frames in a :class:`DataStream` to a flat DataFrame.

    Parameters
    ----------
    stream:
        A fully processed ``DataStream`` (frames must already be populated by
        running a :class:`~processing.pipeline.Pipeline`).

    Returns
    -------
    pd.DataFrame
        One row per resampled waveform sample across all frames.  Columns:

        - ``experiment_id`` (str): identifier of the source recording.
        - ``frame_number`` (int): zero-based index of the frame within this stream.
        - ``sample_index_in_frame`` (int): position of the sample within the
          resampled beat template (0 … resample_len−1).
        - ``time`` (float, seconds): absolute time of the sample, anchored to
          the start of the centre beat of the frame's crop window and spaced at
          the effective resampled rate (``beat_duration / resample_len`` s apart).
        - ``<ppg_col_names>`` (float): one column per PPG channel in
          ``stream.ppg_col_names`` (WL1 channels first, then WL2), containing
          the averaged template amplitude for that sample.
        - ``fSaO2`` (float): fetal SpO2 value interpolated from ``stream.fsao2``
          at each sample's absolute time.
        - ``dc_<ppg_col_names>`` (float): one DC column per PPG channel.  Each
          value is the per-channel minimum of the averaged raw beat template for
          that frame (constant within a frame, broadcast to every sample row).

        Returns an empty DataFrame with the same columns if ``stream.frames``
        is empty.
    """
    col_names = stream.ppg_col_names if stream.ppg_col_names else [f"ch{i}" for i in range(stream.ppg.shape[1])]
    dc_col_names = [f"dc_{c}" for c in col_names]
    base_cols = ["experiment_id", "frame_number", "sample_index_in_frame", "time", *col_names, "fSaO2", *dc_col_names]

    if not stream.frames:
        return pd.DataFrame(columns=base_cols)

    all_experiment_id = []
    all_frame_number = []
    all_sample_idx = []
    all_times = []
    all_templates = []
    all_dc = []

    # Pre-compute a time axis for the raw fsao2 signal so we can interpolate it.
    fsao2_times = np.arange(len(stream.fsao2), dtype=float) / stream.fs

    for frame_number, frame in enumerate(stream.frames):
        template = frame.template          # (resample_len, 10)
        resample_len = template.shape[0]

        # Use the centre beat of the window as the time reference.
        mid = len(frame.crop_start_idx) // 2
        beat_start_sec = float(frame.crop_start_idx[mid]) / stream.fs
        beat_duration_sec = (float(frame.crop_end_idx[mid]) - float(frame.crop_start_idx[mid])) / stream.fs

        # Effective sample rate after resampling: resample_len samples span one beat period.
        # Time of sample s = beat_start + s * beat_duration / resample_len
        # (endpoint=False so the last sample doesn't overshoot into the next beat)
        sample_times = np.linspace(beat_start_sec, beat_start_sec + beat_duration_sec, resample_len, endpoint=False)

        all_experiment_id.append(np.full(resample_len, stream.experiment_id))
        all_frame_number.append(np.full(resample_len, frame_number, dtype=int))
        all_sample_idx.append(np.arange(resample_len, dtype=int))
        all_times.append(sample_times)
        all_templates.append(template)
        # DC is constant within a frame — broadcast to every sample row
        all_dc.append(np.tile(frame.dc, (resample_len, 1)))

    all_sample_times = np.concatenate(all_times)
    fsao2_interp = np.interp(all_sample_times, fsao2_times, stream.fsao2.astype(float))

    df = pd.DataFrame(np.vstack(all_templates), columns=col_names)
    df.insert(0, "time", all_sample_times)
    df.insert(0, "sample_index_in_frame", np.concatenate(all_sample_idx))
    df.insert(0, "frame_number", np.concatenate(all_frame_number))
    df.insert(0, "experiment_id", np.concatenate(all_experiment_id))
    df["fSaO2"] = fsao2_interp
    dc_data = np.vstack(all_dc)  # (total_rows, 10)
    for i, dc_col in enumerate(dc_col_names):
        df[dc_col] = dc_data[:, i]
    return df


def export_to_csv(
    streams: list[DataStream],
    out_csv: Path,
) -> pd.DataFrame:
    """Concatenate frames from multiple streams and save to CSV.

    Parameters
    ----------
    streams:
        List of processed :class:`DataStream` objects (one per experiment).
    out_csv:
        Destination CSV path.  Parent directories are created if needed.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame that was written to disk.
    """
    rows = [stream_to_dataframe(s) for s in streams]
    final_df = pd.concat(rows, ignore_index=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved {len(final_df)} rows → {out_csv}")
    return final_df
