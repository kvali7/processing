"""
Utility functions for applying PASFE (Pulse Artifact Subtraction using Frame Extraction).

This module provides helper functions to prepare data for use with compute_frame functions.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from TFO_dataset import SheepData

from psafe_custom.compute_frame import compute_frame, compute_start_indices


def heartrate_to_frame_length(
    heartrate: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Convert heart rate (in Hz) to frame length in samples.

    The frame length represents the number of samples in one cardiac cycle.
    This is calculated as sampling_rate / heartrate and rounded to the nearest integer.

    Args:
        heartrate: 1D numpy array of heart rate values in Hz (same length as time series)
        sampling_rate: Sampling rate of the signal in Hz (samples per second)

    Returns:
        1D numpy array of frame lengths in samples (integers), same length as heartrate

    Example:
        >>> import numpy as np
        >>> # Constant heart rate of 1.0 Hz at 100 Hz sampling
        >>> heartrate = np.full(1000, 1.0)
        >>> frame_lengths = heartrate_to_frame_length(heartrate, 100.0)
        >>> frame_lengths[0]
        100

        >>> # Variable heart rate
        >>> heartrate = np.array([1.0, 1.2, 0.8, 1.5])
        >>> frame_lengths = heartrate_to_frame_length(heartrate, 100.0)
        >>> frame_lengths
        array([100,  83, 125,  67])
    """
    # Calculate frame lengths: samples per cardiac cycle
    # frame_length = sampling_rate / heartrate
    frame_lengths = sampling_rate / heartrate

    # Round to nearest integer
    frame_lengths = np.round(frame_lengths).astype(int)

    return frame_lengths


def compute_frames_from_df(
    dataframe: pd.DataFrame,
    time_series_columns: list[str],
    heartrate_column: str,
    sampling_rate: float,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear",
) -> dict[str, list[np.ndarray]]:
    """
    Apply compute_frame to multiple time series columns from a dataframe.

    This function extracts heart rate data from the dataframe, converts it to frame lengths,
    and applies the compute_frame function to each specified time series column.

    Args:
        dataframe: DataFrame containing the time series data and heart rate
        time_series_columns: List of column names to process
        heartrate_column: Name of the column containing heart rate values (in Hz)
        sampling_rate: Sampling rate of the signals in Hz
        average_frame_length: Target length for interpolated frames
        frames_to_average: Number of frames to collect and average together
        sample_skip: Number of samples to skip between computing averaged frames
        interpolation_kind: Type of interpolation ('linear', 'cubic', 'quadratic', 'nearest', etc.)

    Returns:
        Dictionary where keys are column names and values are lists of averaged frames

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create sample dataframe
        >>> df = pd.DataFrame({
        ...     'ch1': np.sin(np.linspace(0, 10, 1000)),
        ...     'ch2': np.cos(np.linspace(0, 10, 1000)),
        ...     'heartrate': np.full(1000, 1.0)
        ... })
        >>>
        >>> # Process multiple channels
        >>> results = compute_frames_from_df(
        ...     dataframe=df,
        ...     time_series_columns=['ch1', 'ch2'],
        ...     heartrate_column='heartrate',
        ...     sampling_rate=100.0,
        ...     average_frame_length=100,
        ...     frames_to_average=5,
        ...     sample_skip=25
        ... )
        >>>
        >>> # Results is a dictionary with keys 'ch1' and 'ch2'
        >>> len(results['ch1'])  # Number of averaged frames for ch1
    """
    # Extract heart rate from dataframe and convert to numpy array
    heartrate = dataframe[heartrate_column].to_numpy()

    # Convert heart rate to frame lengths
    frame_lengths = heartrate_to_frame_length(heartrate, sampling_rate)

    # Dictionary to store results for each column
    results = {}

    # Process each time series column
    for column in time_series_columns:
        # Extract time series as numpy array
        time_series = dataframe[column].to_numpy()

        # Apply compute_frame
        averaged_frames = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=average_frame_length,
            frames_to_average=frames_to_average,
            sample_skip=sample_skip,
            interpolation_kind=interpolation_kind,
        )

        # Store results
        results[column] = averaged_frames

    return results


def  get_sheep_dataframe(idx: int) -> pd.DataFrame:
    """
    Returns a dataframe containing sheep PPG data along with the fSaO2. The indices match the ones from Weitai's paper

    Columns Names:
        - ch#voltsWL#
        - fSpO2
        - FHR  # (This is in Hz)
    (Note : Different files might have different order in channel names. Use the above format to select channels!
    Also - there is a small naming issue with sp2021_S2R1. Theere is one single FHR file but the iq_demod has parts
    Just using a band-aid solution for this)
    """
    tags_yaml_path = Path(__file__).parent.resolve() / "tags.yaml"
    with open(tags_yaml_path) as f:
        tags = yaml.safe_load(f)["tags"]
    selected_tag = tags[idx]
    selected_tag = {
        "experiment_number": int(selected_tag["experiment_number"]),
        "experiment_round": int(selected_tag["experiment_round"]),
        "experiment_year_prefix": selected_tag["experiment_year_prefix"],
        "additional_info": selected_tag["additional_info"],
    }
    ppg_data = SheepData("iq_demod_optical").get_data_from_tag(selected_tag)
    fsao2 = SheepData("fSaO2").get_data_from_tag(selected_tag)
    # There is a slight issue with FHR - BAND-AID solution
    selected_tag["additional_info"] = ""
    # fhr = SheepData("FHR").get_data_from_tag(selected_tag)
    # Check if the FHR file exists for the selected tag, if not try with the original additional_info
    # if not SheepData("FHR")
    if selected_tag in [{k:d[k] for k in d.keys() if k in selected_tag.keys()} for d in SheepData("FHR_corrected").get_available_tags()]:
        fhr = SheepData("FHR_corrected").get_data_from_tag(selected_tag) # (Randall Note): Changed FHR to FHR_corrected.
        mhr = SheepData("MHR_corrected").get_data_from_tag(selected_tag)
    else:
        fhr = SheepData("FHR").get_data_from_tag(selected_tag)
        mhr = SheepData("MHR").get_data_from_tag(selected_tag)
        # print(f"Warning: Corrected_FHR not found for {tags[idx]['experiment_id']}. Using regular FHR and MHR instead.")
    fhr.columns = ["FHR"]  # By default doesn't have nameeeee
    fhr["FHR"] /= 60.0
    
    # (Randall Note): Added MHR as well.
    mhr.columns = ["MHR"]
    mhr["MHR"] /= 60.0
    
    mrr = None
    if selected_tag in [{k:d[k] for k in d.keys() if k in selected_tag.keys()} for d in SheepData("MRR").get_available_tags()]:
        mrr = SheepData("MRR").get_data_from_tag(selected_tag)
        mrr.columns = ["MRR"]
        mrr["MRR"] /= 60.0
        mrr_upsampled = mrr.loc[mrr.index.repeat(80)].reset_index(drop=True)

    fsao2_upsampled = fsao2.loc[fsao2.index.repeat(80)].reset_index(drop=True)
    fhr_upsampled = fhr.loc[fhr.index.repeat(80)].reset_index(drop=True)
    mhr_upsampled = mhr.loc[mhr.index.repeat(80)].reset_index(drop=True)
    min_len = min(len(ppg_data), len(fsao2_upsampled), len(fhr_upsampled), len(mhr_upsampled))
    if mrr is not None:
        min_len = min(min_len, len(mrr_upsampled))
        mrr_upsampled = mrr_upsampled.iloc[:min_len].reset_index(drop=True)
    # min_len = min(len(ppg_data), len(fsao2_upsampled), len(fhr_upsampled))
    ppg_data = ppg_data.iloc[:min_len].reset_index(drop=True)
    fsao2_upsampled = fsao2_upsampled.iloc[:min_len].reset_index(drop=True)
    fhr_upsampled = fhr_upsampled.iloc[:min_len].reset_index(drop=True)
    mhr_upsampled = mhr_upsampled.iloc[:min_len].reset_index(drop=True)
    # ppg_data = pd.concat([ppg_data, fsao2_upsampled, fhr_upsampled], axis=1)
    if mrr is not None:
        ppg_data = pd.concat([ppg_data, fsao2_upsampled, fhr_upsampled, mhr_upsampled, mrr_upsampled], axis=1)
    else:
        ppg_data = pd.concat([ppg_data, fsao2_upsampled, fhr_upsampled, mhr_upsampled], axis=1)
    return ppg_data


def get_sheep_experiment_id(idx: int) -> dict[str, Any]:
    """
    Get the corresponding experiment id for get_sheep_dataframe
    """
    with open(Path(__file__).parent.resolve() / "tags.yaml") as f:
        tags = yaml.safe_load(f)["tags"]
    return tags[idx]


def get_sorted_ac_columns() -> dict[str, list[str]]:
    """
    Returns sorted list of AC column names per wavelength (wl1 & wl2).
    """
    wl1_columns = [f"ch{i + 1}voltsWL1" for i in range(5)]
    wl2_columns = [f"ch{i + 1}voltsWL2" for i in range(5)]
    return {"wl1": wl1_columns, "wl2": wl2_columns}


def compute_ac_ratio(
    dataframe: pd.DataFrame,
    time_series_columns: list[str],
    heartrate_column: str,
    sampling_rate: float,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear",
) -> dict[str, np.ndarray]:
    """
    Extension of compute_frames_from_df that computes the AC Ratio between "wl1" and "wl2" channels.

    This function:
    1. Computes frames for all specified channels
    2. Converts each frame to a single AC value (AC = max - min)
    3. Creates AC time series for each channel
    4. Computes ratios between corresponding WL1 and WL2 channels

    Args:
        dataframe: DataFrame containing the time series data and heart rate
        time_series_columns: List of column names to process (should include both WL1 and WL2 channels)
        heartrate_column: Name of the column containing heart rate values (in Hz)
        sampling_rate: Sampling rate of the signals in Hz
        average_frame_length: Target length for interpolated frames
        frames_to_average: Number of frames to collect and average together
        sample_skip: Number of samples to skip between computing averaged frames
        interpolation_kind: Type of interpolation ('linear', 'cubic', 'quadratic', 'nearest', etc.)

    Returns:
        Dictionary where keys are "Ratio chX" and values are numpy arrays of AC ratio time series
        (e.g., {"Ratio ch1": array([...]), "Ratio ch2": array([...])})

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from psafe_custom import get_sheep_dataframe, get_sorted_ac_columns, compute_ac_ratio
        >>>
        >>> # Load sheep data
        >>> data = get_sheep_dataframe(7)
        >>> ac_columns = get_sorted_ac_columns()
        >>>
        >>> # Include both wavelengths in the columns list
        >>> all_channels = ac_columns["wl1"][:2] + ac_columns["wl2"][:2]
        >>>
        >>> # Compute AC ratios
        >>> ratio_dict = compute_ac_ratio(
        ...     dataframe=data,
        ...     time_series_columns=all_channels,
        ...     heartrate_column='FHR',
        ...     sampling_rate=80.0,
        ...     average_frame_length=100,
        ...     frames_to_average=120,
        ...     sample_skip=80
        ... )
        >>>
        >>> # Results dictionary has keys like "Ratio ch1", "Ratio ch2"
        >>> list(ratio_dict.keys())
        ['Ratio ch1', 'Ratio ch2']
        >>> # Each value is a numpy array (AC ratio time series)
        >>> ratio_dict['Ratio ch1'].shape
        (245,)
    """
    # First, compute frames for all channels
    frames_dict = compute_frames_from_df(
        dataframe=dataframe,
        time_series_columns=time_series_columns,
        heartrate_column=heartrate_column,
        sampling_rate=sampling_rate,
        average_frame_length=average_frame_length,
        frames_to_average=frames_to_average,
        sample_skip=sample_skip,
        interpolation_kind=interpolation_kind,
    )

    # Step 1: Convert each frame to AC value (max - min) for all channels
    ac_time_series_dict = {}
    for channel, frames_list in frames_dict.items():
        # Convert each frame to a single AC value
        ac_values = np.zeros(len(frames_list))
        for idx, frame in enumerate(frames_list):
            max_val = np.max(frame)
            min_val = np.min(frame)
            ac_value = max_val - min_val
            ac_values[idx] = ac_value
        # Store as numpy array (AC time series)
        ac_time_series_dict[channel] = ac_values

    # Step 2: Separate channels into WL1 and WL2
    wl1_channels = {
        col: ac_time_series_dict[col]
        for col in time_series_columns
        if "WL1" in col and col in ac_time_series_dict
    }
    wl2_channels = {
        col: ac_time_series_dict[col]
        for col in time_series_columns
        if "WL2" in col and col in ac_time_series_dict
    }

    # Step 3: Compute ratios for channels 1 to 5
    ratio_dict = {}

    for ch_num in range(1, 6):  # ch1 to ch5
        # Try to find both WL1 and WL2 for this channel number
        wl1_key = None
        wl2_key = None

        # Find WL1 channel (try both formats)
        for key in wl1_channels.keys():
            if f"ch{ch_num}" in key:
                wl1_key = key
                break

        # Find WL2 channel (try both formats)
        for key in wl2_channels.keys():
            if f"ch{ch_num}" in key:
                wl2_key = key
                break

        # If both exist, compute the ratio
        if wl1_key is not None and wl2_key is not None:
            wl1_ac_series = wl1_channels[wl1_key]
            wl2_ac_series = wl2_channels[wl2_key]

            # Compute ratio time series (add small epsilon to avoid division by zero)
            ratio_series = wl1_ac_series / (wl2_ac_series + 1e-10)

            # Store with key format "Ratio chX"
            ratio_dict[f"Ratio ch{ch_num}"] = ratio_series

    return ratio_dict


def compute_averaged_fsao2(
    dataframe: pd.DataFrame,
    heartrate_column: str,
    sampling_rate: float,
    frames_to_average: int,
    sample_skip: int,
) -> np.ndarray:
    """
    Compute average fSaO2 values corresponding to each frame window.

    This function uses compute_start_indices to determine the temporal boundaries of each
    frame window, then extracts and averages the fSaO2 values within those boundaries.

    Args:
        dataframe: DataFrame containing fSaO2 and heart rate data
        heartrate_column: Name of the column containing heart rate values (in Hz)
        sampling_rate: Sampling rate of the signals in Hz
        frames_to_average: Number of frames to collect and average together
        sample_skip: Number of samples to skip between computing averaged frames

    Returns:
        np.ndarray: 1D numpy array of average fSaO2 values, one per frame window

    Example:
        >>> from psafe_custom import get_sheep_dataframe, compute_averaged_fsao2
        >>>
        >>> # Load sheep data
        >>> data = get_sheep_dataframe(7)
        >>>
        >>> # Compute averaged fSaO2
        >>> avg_fsao2 = compute_averaged_fsao2(
        ...     dataframe=data,
        ...     heartrate_column='FHR',
        ...     sampling_rate=80.0,
        ...     frames_to_average=120,
        ...     sample_skip=80
        ... )
        >>>
        >>> print(f"Number of fSaO2 values: {len(avg_fsao2)}")
        >>> print(f"Mean fSaO2: {np.mean(avg_fsao2):.2f}")
    """
    # Extract heart rate and convert to frame lengths
    heartrate = dataframe[heartrate_column].to_numpy()
    frame_lengths = heartrate_to_frame_length(heartrate, sampling_rate)

    # Compute start indices for all windows
    time_series_length = len(heartrate)
    start_indices = compute_start_indices(
        time_series_length=time_series_length,
        current_frame_length=frame_lengths,
        frames_to_average=frames_to_average,
        sample_skip=sample_skip,
    )

    # Extract fSaO2 column
    fsao2 = dataframe["fSpO2"].to_numpy()

    # Initialize array to store averaged fSaO2 values
    num_windows = len(start_indices)
    averaged_fsao2 = np.zeros(num_windows)

    # For each window, compute average fSaO2
    for idx, window_indices in enumerate(start_indices):
        # window_indices[0] is the window start
        # window_indices[-1] is the last frame start
        first_frame_start = window_indices[0]
        last_frame_start = window_indices[-1]

        # Crop fSaO2 from first frame start to last frame start
        fsao2_segment = fsao2[first_frame_start:last_frame_start]

        # Compute average and store
        averaged_fsao2[idx] = np.mean(fsao2_segment)

    return averaged_fsao2


if __name__ == "__main__":
    idx = 7
    ac_columns = get_sorted_ac_columns()["wl1"][:2] + get_sorted_ac_columns()["wl2"][:2]
    data = get_sheep_dataframe(idx)

    # Compute AC ratios
    ac_ratio = compute_ac_ratio(data, ac_columns, "FHR", 80.0, 100, 120, 80, "linear")
    print("AC Ratios:")
    for key, values in ac_ratio.items():
        print(f"  {key}: shape={values.shape}, mean={np.mean(values):.4f}")

    # Compute averaged fSaO2
    avg_fsao2 = compute_averaged_fsao2(data, "FHR", 80.0, 120, 80)
    print("Averaged fSaO2:")
    print(f"  shape={avg_fsao2.shape}, mean={np.mean(avg_fsao2):.2f}")
    print(f"  Matches AC ratio length: {len(avg_fsao2) == len(ac_ratio['Ratio ch1'])}")
