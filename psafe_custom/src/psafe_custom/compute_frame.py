"""
Frame computation and averaging module.

This module provides functionality to compute averaged frames from time series data
with variable frame lengths.
"""

from typing import List

import numpy as np
from scipy import interpolate


def compute_start_indices(
    time_series_length: int,
    current_frame_length: np.ndarray,
    frames_to_average: int,
    sample_skip: int,
) -> List[List[int]]:
    """
    Compute starting indices for frame extraction windows.

    This function determines where each window of frames should start in the time series
    and tracks the start index of each individual frame within the window.

    Args:
        time_series_length: Length of the time series
        current_frame_length: 1D numpy array of frame lengths (same length as time_series)
        frames_to_average: Number of frames to collect and average together
        sample_skip: Number of samples to skip between computing averaged frames

    Returns:
        List of lists where each inner list contains:
        - First element: start index of the averaging window
        - Subsequent elements: start indices of each individual frame within that window

        Example: [[0, 0, 50, 100, 150, 200], [25, 25, 75, 125, 175, 225], ...]
        means the first window starts at index 0, with individual frames starting at 0, 50, 100, 150, 200
    """
    start_indices = []
    start_index = 0

    while True:
        current_index = start_index
        valid_frames = 0
        window_indices = [start_index]  # First element is the window start

        # Check if we can extract frames_to_average frames starting from start_index
        for frame_num in range(frames_to_average):
            if current_index >= time_series_length:
                break

            frame_len = int(current_frame_length[current_index])

            if current_index + frame_len > time_series_length:
                break

            window_indices.append(current_index)  # Add individual frame start index
            valid_frames += 1
            current_index += frame_len

        # Only add this window if we can extract all required frames
        if valid_frames < frames_to_average:
            break

        start_indices.append(window_indices)
        start_index += sample_skip

        if start_index >= time_series_length:
            break

    return start_indices


def compute_frame_with_metadata(
    time_series: np.ndarray,
    current_frame_length: np.ndarray,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear",
) -> tuple[List[np.ndarray], List[List[int]], np.ndarray]:
    """
    Compute averaged frames with metadata about starting indices.

    Similar to compute_frame but also returns the starting index for each averaged frame
    and all interpolated frames as a 3D matrix.

    Args:
        time_series: 1D numpy array containing the time series data
        current_frame_length: 1D numpy array of frame lengths (same length as time_series)
        average_frame_length: Target length for interpolated frames
        frames_to_average: Number of frames to collect and average together
        sample_skip: Number of samples to skip between computing averaged frames
        interpolation_kind: Type of interpolation for scipy.interpolate.interp1d.
            Options: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next'. Defaults to 'linear'.

    Returns:
        Tuple of (averaged_frames, start_indices, all_interpolated_frames)
        - averaged_frames: List of averaged frames
        - start_indices: List of lists, where each inner list contains [window_start, frame1_start, frame2_start, ...]
        - all_interpolated_frames: 3D numpy array of shape (num_windows, frames_to_average, average_frame_length)
          containing all interpolated frames before averaging
    """
    # Input validation
    if not isinstance(time_series, np.ndarray):
        raise ValueError("time_series must be a numpy array")
    if not isinstance(current_frame_length, np.ndarray):
        raise ValueError("current_frame_length must be a numpy array")

    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array")
    if current_frame_length.ndim != 1:
        raise ValueError("current_frame_length must be a 1D array")

    if len(time_series) != len(current_frame_length):
        raise ValueError("time_series and current_frame_length must have equal length")

    if average_frame_length <= 0:
        raise ValueError("average_frame_length must be positive")
    if frames_to_average <= 0:
        raise ValueError("frames_to_average must be positive")
    if sample_skip <= 0:
        raise ValueError("sample_skip must be positive")

    # Compute start indices for all windows
    total_length = len(time_series)
    start_indices = compute_start_indices(
        time_series_length=total_length,
        current_frame_length=current_frame_length,
        frames_to_average=frames_to_average,
        sample_skip=sample_skip,
    )

    averaged_frames = []
    all_interpolated_frames = []

    # Process each window starting at the computed indices
    for window_indices in start_indices:
        frames = []
        # window_indices[0] is the window start, window_indices[1:] are individual frame starts

        # Extract frames for this window using the individual frame start indices
        for frame_idx in range(1, len(window_indices)):
            frame_start = window_indices[frame_idx]
            frame_len = int(current_frame_length[frame_start])
            frame = time_series[frame_start : frame_start + frame_len]
            frames.append(frame)

        interpolated_frames = np.zeros((frames_to_average, average_frame_length))
        for idx, frame in enumerate(frames):
            if len(frame) == average_frame_length:
                # No interpolation needed
                interpolated_frames[idx, :] = frame
            else:
                # Create interpolation function
                x_old = np.linspace(0, 1, len(frame))
                x_new = np.linspace(0, 1, average_frame_length)
                f = interpolate.interp1d(x_old, frame, kind=interpolation_kind)
                interpolated_frame = f(x_new)
                interpolated_frames[idx, :] = interpolated_frame

        # Compute averaged frame and store results
        averaged_frame = np.mean(interpolated_frames, axis=0)
        averaged_frames.append(averaged_frame)
        all_interpolated_frames.append(interpolated_frames)

    # Stack all interpolated frames into a 3D array
    # Shape: (num_windows, frames_to_average, average_frame_length)
    if len(all_interpolated_frames) > 0:
        all_interpolated_frames_3d = np.stack(all_interpolated_frames, axis=0)
    else:
        all_interpolated_frames_3d = np.array([]).reshape(
            0, frames_to_average, average_frame_length
        )

    return averaged_frames, start_indices, all_interpolated_frames_3d


def compute_frame(
    time_series: np.ndarray,
    current_frame_length: np.ndarray,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear",
) -> List[np.ndarray]:
    """
    Compute averaged frames from a time series with variable frame lengths.

    This function extracts frames from a time series, interpolates them to a uniform
    length, and averages them together. The process is repeated at intervals defined
    by sample_skip.

    Algorithm:
    1. Start from the 0th index of the time series
    2. Check the corresponding current_frame_length
    3. Crop out a frame of that length from the time series
    4. Jump the same length as the current frame
    5. Repeat to grab frames_to_average number of frames
    6. Linear interpolate all frames to average_frame_length
    7. Compute the average frame and store it
    8. Jump ahead by sample_skip and repeat
    9. Continue until insufficient samples remain

    Args:
        time_series: 1D numpy array containing the time series data
        current_frame_length: 1D numpy array of frame lengths (same length as time_series)
        average_frame_length: Target length for interpolated frames
        frames_to_average: Number of frames to collect and average together
        sample_skip: Number of samples to skip between computing averaged frames
        interpolation_kind: Type of interpolation for scipy.interpolate.interp1d.
            Options: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next'. Defaults to 'linear'.

    Returns:
        List of averaged frames, each of length average_frame_length

    Raises:
        ValueError: If time_series and current_frame_length have different lengths
        ValueError: If any input parameter is invalid

    Example:
        >>> time_series = np.sin(np.linspace(0, 10, 1000))
        >>> frame_lengths = np.full(1000, 50)
        >>> avg_frames = compute_frame(time_series, frame_lengths, 50, 5, 10)
        >>> len(avg_frames[0])
        50
    """
    averaged_frames, _, _ = compute_frame_with_metadata(
        time_series,
        current_frame_length,
        average_frame_length,
        frames_to_average,
        sample_skip,
        interpolation_kind,
    )
    return averaged_frames
