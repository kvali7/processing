"""
Signal Quality Index (SQI) computation module.

This module provides functions to compute various signal quality metrics
for assessing the quality of physiological signal frames.
"""

import numpy as np
from scipy.stats import skew


def get_sqi(
    frame: np.ndarray,
    sqi_type: str = "skewness",
) -> float:
    """
    Compute Signal Quality Index (SQI) for a given frame of data.

    This function computes various signal quality metrics to assess the quality
    of a signal frame. Different SQI types can be used depending on the application.

    Args:
        frame: 1D numpy array containing the signal frame data
        sqi_type: Type of SQI to compute. Options:
            - 'skewness': Magnitude of skewness (default)

    Returns:
        float: The computed SQI value

    Raises:
        ValueError: If frame is not a 1D numpy array
        ValueError: If sqi_type is not recognized
        ValueError: If frame is empty or has insufficient data

    Example:
        >>> import numpy as np
        >>> from psafe_custom.sqi import get_sqi
        >>> frame = np.random.randn(100)
        >>> sqi_value = get_sqi(frame, sqi_type='skewness')
        >>> isinstance(sqi_value, float)
        True

        >>> # Using with computed frames
        >>> from psafe_custom import compute_frame
        >>> time_series = np.sin(np.linspace(0, 10, 1000))
        >>> frame_lengths = np.full(1000, 50)
        >>> avg_frames = compute_frame(time_series, frame_lengths, 50, 5, 10)
        >>> sqi_values = [get_sqi(frame) for frame in avg_frames]
    """
    # Input validation
    if not isinstance(frame, np.ndarray):
        raise ValueError("frame must be a numpy array")

    if frame.ndim != 1:
        raise ValueError("frame must be a 1D array")

    if len(frame) == 0:
        raise ValueError("frame cannot be empty")

    if len(frame) < 3:
        raise ValueError("frame must have at least 3 samples for skewness computation")

    # Compute SQI based on type
    if sqi_type == "skewness":
        # Compute magnitude of skewness
        skewness_value = skew(frame)
        sqi_value = np.abs(skewness_value)
    elif sqi_type == "perfusion":
        mean = np.mean(frame)
        sqi_value = np.abs((np.max(frame) - np.mean(frame))/ mean * 100.0)
    else:
        raise ValueError(f"Unknown sqi_type: '{sqi_type}'. Supported types: 'skewness'")

    return float(sqi_value)

