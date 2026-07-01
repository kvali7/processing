from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, argrelextrema

from processing import FrequencyWarping as fw

from processing.data_model import DataStream, Frame

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PipelineStep(Protocol):
    """Any callable that mutates a :class:`DataStream` in-place."""

    def is_valid(self, stream: DataStream) -> bool:
        """Return True if all prerequisites for this step are present in *stream*."""
        ...

    def __call__(self, stream: DataStream) -> None: ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Run a sequence of :class:`PipelineStep` objects against a stream.

    Parameters
    ----------
    steps:
        Ordered list of steps to apply.  Each step mutates the stream in-place.
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self.steps = steps

    def run(self, stream: DataStream) -> DataStream:
        for step in self.steps:
            if not step.is_valid(stream):
                raise RuntimeError(
                    f"{type(step).__name__}.is_valid() failed: missing prerequisites in DataStream."
                )
            step(stream)
        return stream


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass
class HighPassFilter:
    """Apply a zero-phase Butterworth high-pass filter to ``stream.ppg`` in-place.

    Parameters
    ----------
    cutoff_hz:
        High-pass cutoff frequency in Hz.
    order:
        Filter order.
    """

    cutoff_hz: float = 1.0
    order: int = 4

    def is_valid(self, stream: DataStream) -> bool:
        return stream.ppg is not None and stream.ppg.ndim == 2

    def __call__(self, stream: DataStream) -> None:
        stream.extras["ppg_raw"] = stream.ppg.copy()
        b, a = butter(self.order, self.cutoff_hz / (0.5 * stream.fs), btype="high", analog=False)
        stream.ppg = filtfilt(b, a, stream.ppg, axis=0)


@dataclass
class HRSmoother:
    """Smooth the HR signal and write the result to ``stream.extras["hr_smooth"]``.

    Reads ``stream.hr`` (full-rate, N samples), downsamples to 1 Hz, applies a
    centred rolling mean, then upsamples back to N samples.

    Parameters
    ----------
    window_seconds:
        Width of the rolling-mean window in seconds.
    """

    window_seconds: int = 60

    def is_valid(self, stream: DataStream) -> bool:
        return stream.hr is not None and len(stream.hr) > 0

    def __call__(self, stream: DataStream) -> None:
        N = len(stream.hr)
        stride = int(stream.fs)

        hr_ds = stream.hr[::stride]  # 1 Hz
        hr_smooth_1hz = (
            pd.Series(hr_ds)
            .rolling(window=self.window_seconds, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
        stream.extras["hr_smooth"] = np.repeat(hr_smooth_1hz, stride)[:N]


@dataclass
class PeakDetector:
    """Detect beat peaks in the HP-filtered PPG and store them in ``stream.extras["peaks"]``.

    Uses frequency warping on the chosen PPG channel followed by
    ``scipy.signal.argrelextrema`` to locate beat troughs in the warped signal
    (which correspond to beat boundaries).

    Requires ``stream.extras["hr_smooth"]`` to be populated (run :class:`HRSmoother`
    first).

    Parameters
    ----------
    channel_idx:
        Column index of ``stream.ppg`` to use for peak detection (0-based).
        Default ``3`` corresponds to ch4 (the 4th WL1 channel).
    """

    channel_idx: int = 3

    def is_valid(self, stream: DataStream) -> bool:
        return "hr_smooth" in stream.extras

    def __call__(self, stream: DataStream) -> None:
        ppg_ch = stream.ppg[:, self.channel_idx]
        hr_rep = stream.extras["hr_smooth"][: len(ppg_ch)]
        warped = fw.frequency_warpping_signal_extraction(ppg_ch, stream.fs, hr_rep, 1)
        stream.extras["peaks"] = argrelextrema(warped, np.less)[0]


@dataclass
class FrameBuilder:
    """Build averaged PSAFE frames from accepted beat segments.

    For each sliding window of ``frames_to_average`` consecutive *valid* beats:

    * Resample each beat to ``resample_len`` samples (linear interpolation, all 10 channels).
    * Average across beats → ``Frame.template`` of shape ``(resample_len, 10)``.
    * Record boundary indices → ``Frame.crop_start_idx`` / ``Frame.crop_end_idx``,
        each of shape ``(frames_to_average,)``.

    A beat is *valid* (accepted) when its sample count falls within
    ``segment_tolerance_ratio`` of the expected beat length derived from the
    smoothed HR.

    Requires ``stream.extras["peaks"]`` and ``stream.extras["hr_smooth"]``
    (run :class:`PeakDetector` and :class:`HRSmoother` first).

    Parameters
    ----------
    frames_to_average:
        Number of consecutive beats to average into one frame.
    stride:
        Step size (in accepted beats) between successive frames.
    resample_len:
        Number of samples each beat is resampled to before averaging.
    segment_tolerance_ratio:
        Fractional tolerance around the expected beat length.  A beat whose
        raw sample count deviates by more than this fraction is rejected.
    """

    frames_to_average: int = 120
    stride: int = 2
    resample_len: int = 100
    segment_tolerance_ratio: float = 0.2

    def is_valid(self, stream: DataStream) -> bool:
        return "peaks" in stream.extras and "hr_smooth" in stream.extras and "ppg_raw" in stream.extras

    def resample_segment(self, seg: np.ndarray) -> np.ndarray:
        """Resample a single raw beat segment to ``(resample_len, 10)``.

        Parameters
        ----------
        seg:
            Raw PPG samples for one beat, shape ``(idxlen, 10)``.

        Returns
        -------
        np.ndarray
            Resampled segment of shape ``(resample_len, 10)``.
        """
        x_old = np.linspace(0, 1, seg.shape[0])
        x_new = np.linspace(0, 1, self.resample_len)
        return np.stack(
            [np.interp(x_new, x_old, seg[:, c]) for c in range(seg.shape[1])],
            axis=1,
        )

    def compute_template(self, segments: list[np.ndarray]) -> np.ndarray:
        """Compute the frame template from a window of resampled segments.

        Parameters
        ----------
        segments:
            List of ``frames_to_average`` arrays, each of shape
            ``(resample_len, 10)``.

        Returns
        -------
        np.ndarray
            Averaged template of shape ``(resample_len, 10)``.
        """
        return np.mean(segments, axis=0)

    def is_beat_accepted(self, start_idx: int, end_idx: int, stream: DataStream) -> bool:
        """Return True if the beat defined by [start_idx, end_idx) passes the tolerance check.

        Parameters
        ----------
        start_idx:
            First sample index of the beat (inclusive).
        end_idx:
            Last sample index of the beat (exclusive).
        stream:
            The current :class:`DataStream`; ``extras["hr_smooth"]`` is used to
            derive the expected beat length at 1 Hz.
        """
        hr_1hz = stream.extras["hr_smooth"][:: int(stream.fs)]
        idxlen = end_idx - start_idx
        if idxlen <= 1:
            return False
        time_idx = int(
            np.clip(
                int(np.floor((start_idx + end_idx) / (2.0 * stream.fs))),
                0,
                len(hr_1hz) - 1,
            )
        )
        expected_len = stream.fs / max(float(hr_1hz[time_idx]), 1e-6)
        tolerance = self.segment_tolerance_ratio * expected_len
        return bool(expected_len - tolerance < idxlen < expected_len + tolerance)

    def __call__(self, stream: DataStream) -> None:
        ppg = stream.ppg
        ppg_raw = stream.extras["ppg_raw"]
        peaks = stream.extras["peaks"]

        accepted_starts: list[int] = []
        accepted_ends: list[int] = []
        resampled_segs: list[np.ndarray] = []
        resampled_raw_segs: list[np.ndarray] = []

        for i in range(len(peaks) - 1):
            start_idx = int(peaks[i])
            end_idx = int(peaks[i + 1])

            if not self.is_beat_accepted(start_idx, end_idx, stream):
                continue

            accepted_starts.append(start_idx)
            accepted_ends.append(end_idx)
            resampled_segs.append(self.resample_segment(ppg[start_idx:end_idx, :]))
            resampled_raw_segs.append(self.resample_segment(ppg_raw[start_idx:end_idx, :]))

        n_valid = len(resampled_segs)
        if n_valid < self.frames_to_average:
            return

        fta = self.frames_to_average
        for i in range(0, n_valid - fta + 1, self.stride):
            template = self.compute_template(resampled_segs[i : i + fta])
            # DC: minimum per channel of the averaged raw beat template (matches PSA.compute_DC col_min)
            raw_template = self.compute_template(resampled_raw_segs[i : i + fta])
            dc = np.min(raw_template, axis=0)  # (10,)
            crop_start = np.array(accepted_starts[i : i + fta], dtype=int)
            crop_end = np.array(accepted_ends[i : i + fta], dtype=int)
            stream.frames.append(Frame(template=template, crop_start_idx=crop_start, crop_end_idx=crop_end, dc=dc))
