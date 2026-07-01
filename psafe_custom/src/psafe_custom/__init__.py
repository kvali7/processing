"""
psafe_custom - A custom Python library
"""

__version__ = "0.1.0"
__author__ = "Rishad"
__email__ = "sa.dip987@gmail.com"

# Import main components
from .apply_pasfe import (
    compute_ac_ratio,
    compute_averaged_fsao2,
    compute_frames_from_df,
    get_sheep_dataframe,
    get_sorted_ac_columns,
    heartrate_to_frame_length,
)
from .compute_frame import (
    compute_frame,
    compute_frame_with_metadata,
    compute_start_indices,
)
from .sqi import get_sqi

__all__ = [
    "__version__",
    "compute_frame",
    "compute_frame_with_metadata",
    "compute_start_indices",
    "heartrate_to_frame_length",
    "compute_frames_from_df",
    "compute_ac_ratio",
    "compute_averaged_fsao2",
    "get_sheep_dataframe",
    "get_sorted_ac_columns",
    "get_sqi",
]
