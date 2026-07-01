# psafe_custom Library Documentation

This documentation file provides comprehensive information about all functions, classes, and their usage examples for agentic AI integration.

## Table of Contents

1. [Installation](#installation)
2. [Core Modules](#core-modules)
3. [Function Reference](#function-reference)
4. [Utility Functions](#utility-functions)
5. [Usage Examples](#usage-examples)
6. [Type Signatures](#type-signatures)

---

## Installation

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

**Dependencies:**
- numpy >= 1.20.0
- scipy >= 1.7.0

---

## Core Modules

### compute_frame

The `compute_frame` module provides functionality for extracting, interpolating, and averaging variable-length frames from time series data.

**Module Path:** `psafe_custom.compute_frame`

### apply_pasfe

The `apply_pasfe` module provides utility functions for preparing data for use with compute_frame functions, particularly for physiological signal processing.

**Module Path:** `psafe_custom.apply_pasfe`

---

## Function Reference

### 1. compute_frame

**Signature:**
```python
def compute_frame(
    time_series: np.ndarray,
    current_frame_length: np.ndarray,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear",
) -> List[np.ndarray]
```

**Description:**
Extracts frames from a time series with variable frame lengths, interpolates them to a uniform length, and computes averaged frames at regular intervals.

**Algorithm Steps:**
1. Start from index 0 of the time series
2. Check the corresponding current_frame_length at that index
3. Extract a frame of that length from the time series
4. Jump ahead by the frame length
5. Repeat steps 2-4 until `frames_to_average` frames are collected
6. Interpolate all frames to `average_frame_length` using specified method
7. Compute the mean of all interpolated frames
8. Store the averaged frame
9. Return to the initial index and jump ahead by `sample_skip`
10. Repeat until insufficient samples remain

**Parameters:**
- `time_series` (np.ndarray): 1D numpy array containing the time series data
- `current_frame_length` (np.ndarray): 1D numpy array of frame lengths (same length as time_series)
- `average_frame_length` (int): Target length for all interpolated frames
- `frames_to_average` (int): Number of consecutive frames to collect and average together
- `sample_skip` (int): Number of samples to skip between computing each averaged frame
- `interpolation_kind` (str, optional): Type of interpolation for scipy.interpolate.interp1d. Options: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'. Defaults to 'linear'

**Returns:**
- `List[np.ndarray]`: List of averaged frames, each with length `average_frame_length`

**Raises:**
- `ValueError`: If time_series and current_frame_length have different lengths
- `ValueError`: If arrays are not 1D
- `ValueError`: If any numeric parameter is not positive
- `ValueError`: If inputs are not numpy arrays

**Constraints:**
- Both input arrays must be 1D numpy arrays of equal length
- All numeric parameters must be positive integers
- Function stops when insufficient samples remain to collect `frames_to_average` frames

---

### 2. compute_frame_with_metadata

**Signature:**
```python
def compute_frame_with_metadata(
    time_series: np.ndarray,
    current_frame_length: np.ndarray,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear",
) -> tuple[List[np.ndarray], List[int], np.ndarray]
```

**Description:**
Similar to `compute_frame` but also returns the starting index for each averaged frame and all interpolated frames as a 3D matrix. This function uses `compute_start_indices` internally to determine valid window positions, then processes each window to extract, interpolate, and average frames.

**Parameters:**
Same as `compute_frame` (including `interpolation_kind`)

**Returns:**
- `tuple` containing:
  - `List[np.ndarray]`: List of averaged frames
  - start_indices: List of lists, where each inner list contains [window_start, frame1_start, frame2_start, ...]
  - `np.ndarray`: 3D numpy array of shape `(num_windows, frames_to_average, average_frame_length)` containing all interpolated frames before averaging

**Use Case:**
When you need to:
- Maintain correspondence between averaged frames and their original positions in the time series
- Access all interpolated frames before averaging for detailed analysis
- Analyze individual frames within each averaging window

**Example:**
```python
import numpy as np
from psafe_custom import compute_frame_with_metadata

time_series = np.sin(np.linspace(0, 10 * np.pi, 1000))
frame_lengths = np.full(1000, 50)

averaged_frames, start_indices, all_interpolated = compute_frame_with_metadata(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=50,
    frames_to_average=5,
    sample_skip=25
)

print(f"Number of averaged frames: {len(averaged_frames)}")
print(f"3D array shape: {all_interpolated.shape}")  # (num_windows, 5, 50)
print(f"First window indices: {start_indices[0]}")
# Output: [0, 0, 50, 100, 150, 200] - window starts at 0, frames at 0, 50, 100, 150, 200
print(f"Second window indices: {start_indices[1]}")
# Output: [25, 25, 75, 125, 175, 225] - window starts at 25, frames at 25, 75, 125, 175, 225
```

---

### 3. compute_start_indices

**Signature:**
```python
def compute_start_indices(
    time_series_length: int,
    current_frame_length: np.ndarray,
    frames_to_average: int,
    sample_skip: int,
) -> List[List[int]]
```

**Description:**
Computes starting indices for frame extraction windows. This function determines where each window of frames should start in the time series and tracks the start index of each individual frame within the window. It validates that each starting position has enough frames available to extract `frames_to_average` frames.

**Parameters:**
- `time_series_length` (int): Length of the time series
- `current_frame_length` (np.ndarray): 1D numpy array of frame lengths (same length as time_series)
- `frames_to_average` (int): Number of frames to collect and average together
- `sample_skip` (int): Number of samples to skip between computing averaged frames

**Returns:**
- `List[List[int]]`: List of lists where each inner list contains:
  - First element: start index of the averaging window
  - Subsequent elements: start indices of each individual frame within that window
  
  Example: `[[0, 0, 50, 100, 150, 200], [25, 25, 75, 125, 175, 225], ...]` means the first window starts at index 0, with individual frames starting at indices 0, 50, 100, 150, 200

**Use Case:**
- Pre-compute valid window positions before processing
- Analyze how many windows can be extracted from a time series
- Determine temporal coverage of frame extraction
- Use with custom frame processing logic

**Example:**
```python
import numpy as np
from psafe_custom import compute_start_indices

# Create sample data
time_series = np.sin(np.linspace(0, 10 * np.pi, 1000))
frame_lengths = np.full(1000, 50)

# Compute start indices
start_indices = compute_start_indices(
    time_series_length=len(time_series),
    current_frame_length=frame_lengths,
    frames_to_average=5,
    sample_skip=25
)

print(f"Number of valid windows: {len(start_indices)}")
print(f"First window: {start_indices[0]}")
# Output: [0, 0, 50, 100, 150, 200] 
# Window starts at 0, frames start at 0, 50, 100, 150, 200

print(f"Second window: {start_indices[1]}")
# Output: [25, 25, 75, 125, 175, 225]
# Window starts at 25, frames start at 25, 75, 125, 175, 225

print(f"Window spacing: {start_indices[1][0] - start_indices[0][0]} samples")
```

**Notes:**
- Only returns indices where a full set of `frames_to_average` frames can be extracted
- Used internally by `compute_frame_with_metadata`
- Useful for understanding temporal distribution of frame extraction

---

## Utility Functions

### heartrate_to_frame_length

**Signature:**
```python
def heartrate_to_frame_length(
    heartrate: np.ndarray,
    sampling_rate: float,
) -> np.ndarray
```

**Description:**
Converts heart rate values (in Hz) to frame lengths in samples. This is useful when you have heart rate data and need to create the `current_frame_length` array for use with `compute_frame`.

The calculation is: `frame_length = sampling_rate / heartrate`, rounded to the nearest integer.

**Parameters:**
- `heartrate` (np.ndarray): 1D numpy array of heart rate values in Hz (same length as time series)
- `sampling_rate` (float): Sampling rate of the signal in Hz (samples per second)

**Returns:**
- `np.ndarray`: 1D numpy array of frame lengths in samples (integers), same length as heartrate

**Raises:**
- `ValueError`: If heartrate is not a 1D numpy array
- `ValueError`: If heartrate contains zero or negative values
- `ValueError`: If sampling_rate is zero or negative

**Example:**
```python
import numpy as np
from psafe_custom import heartrate_to_frame_length

# Constant heart rate of 1.0 Hz at 100 Hz sampling
heartrate = np.full(1000, 1.0)
frame_lengths = heartrate_to_frame_length(heartrate, 100.0)
# Result: array([100, 100, 100, ..., 100])

# Variable heart rate
heartrate = np.array([1.0, 1.2, 0.8, 1.5])
frame_lengths = heartrate_to_frame_length(heartrate, 100.0)
# Result: array([100, 83, 125, 67])
```

---

### compute_frames_from_df

**Signature:**
```python
def compute_frames_from_df(
    dataframe: pd.DataFrame,
    time_series_columns: list[str],
    heartrate_column: str,
    sampling_rate: float,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear",
) -> dict[str, list[np.ndarray]]
```

**Description:**
Applies `compute_frame` to multiple time series columns from a pandas DataFrame. This is a convenience function for batch processing multiple signals with the same heart rate data.

**Parameters:**
- `dataframe` (pd.DataFrame): DataFrame containing the time series data and heart rate
- `time_series_columns` (list[str]): List of column names to process
- `heartrate_column` (str): Name of the column containing heart rate values (in Hz)
- `sampling_rate` (float): Sampling rate of the signals in Hz
- `average_frame_length` (int): Target length for interpolated frames
- `frames_to_average` (int): Number of frames to collect and average together
- `sample_skip` (int): Number of samples to skip between computing averaged frames
- `interpolation_kind` (str, optional): Type of interpolation. Defaults to 'linear'

**Returns:**
- `dict[str, list[np.ndarray]]`: Dictionary where keys are column names and values are lists of averaged frames

**Example:**
```python
import pandas as pd
import numpy as np
from psafe_custom import compute_frames_from_df

# Create sample dataframe with multiple channels
df = pd.DataFrame({
    'ch1': np.sin(np.linspace(0, 20 * np.pi, 1000)),
    'ch2': np.cos(np.linspace(0, 20 * np.pi, 1000)),
    'ch3': np.sin(np.linspace(0, 15 * np.pi, 1000)),
    'heartrate': np.linspace(0.9, 1.1, 1000)
})

# Process all channels at once
results = compute_frames_from_df(
    dataframe=df,
    time_series_columns=['ch1', 'ch2', 'ch3'],
    heartrate_column='heartrate',
    sampling_rate=100.0,
    average_frame_length=100,
    frames_to_average=5,
    sample_skip=25
)

# Access results for each channel
for channel, frames in results.items():
    print(f'{channel}: {len(frames)} averaged frames')
```

---

### get_sheep_dataframe

**Signature:**
```python
def get_sheep_dataframe(idx: int) -> pd.DataFrame
```

**Description:**
Returns a DataFrame containing sheep PPG (photoplethysmography) data along with fSpO2 (fractional arterial oxygen saturation) and FHR (fetal heart rate). This function loads pre-configured sheep experiment data based on an index that matches the dataset ordering from Weitai's paper.

The function:
1. Loads experiment tags from the internal `tags.yaml` configuration file
2. Retrieves PPG data (IQ demodulated optical signals) for the specified experiment
3. Retrieves corresponding fSpO2 measurements
4. Retrieves FHR (fetal heart rate) data and converts from BPM to Hz (divides by 60)
5. Upsamples both fSpO2 and FHR data by repeating each value 80 times to match the PPG sampling rate
6. Concatenates all datasets into a single DataFrame

**Parameters:**
- `idx` (int): Index of the sheep experiment dataset (0-12). The indices correspond to the experiments in Weitai's paper and match the order of files in the data directory.

**Returns:**
- `pd.DataFrame`: DataFrame containing:
  - PPG channel columns with naming format: `ch#voltsWL#` (e.g., `ch1voltsWL1`, `ch2voltsWL2`)
  - `fSpO2` column with upsampled fractional oxygen saturation values
  - `FHR` column with upsampled fetal heart rate values in Hz

**Important Notes:**
- Different files may have different ordering of channel names. Always use the `ch#voltsWL#` format pattern to select channels programmatically.
- The function automatically handles length mismatches by truncating all datasets to the minimum length.
- Both fSpO2 and FHR data are upsampled 80x by repeating each value to match the higher PPG sampling rate.
- FHR values are converted from beats per minute (BPM) to Hz for direct use with frame computation functions.
- There is a known naming issue with sp2021_S2R1 dataset where there's one FHR file but multiple iq_demod parts. The function handles this with a band-aid solution by clearing the `additional_info` field when loading FHR data.

**Example:**
See the complete pipeline example in the Usage Examples section below.

**Data Sources:**
- PPG data: Retrieved from `SheepData("iq_demod_optical")`
- fSpO2 data: Retrieved from `SheepData("fSaO2")` and upsampled 80x
- FHR data: Retrieved from `SheepData("FHR")`, converted from BPM to Hz, and upsampled 80x

**Configuration:**
The function uses experiment tags stored in `tags.yaml` with the following structure:
- `experiment_number`: Sheep identifier number
- `experiment_round`: Round/session number
- `experiment_year_prefix`: Year/season prefix (e.g., "sp2022", "su2020", "sp2021")
- `additional_info`: Optional additional information string

---

### get_sorted_ac_columns

**Signature:**
```python
def get_sorted_ac_columns() -> dict[str, list[str]]
```

**Description:**
Returns sorted lists of AC (alternating current) column names organized by wavelength. This is a convenience function that provides the standard channel naming format for the two wavelengths used in sheep PPG measurements.

**Parameters:**
None

**Returns:**
- `dict[str, list[str]]`: Dictionary with two keys:
  - `"wv1"`: List of channel names for wavelength 1 (5 channels: `["ch1voltsWL1", "ch2voltsWL1", ..., "ch5voltsWL1"]`)
  - `"wv2"`: List of channel names for wavelength 2 (5 channels: `["ch1voltsWL2", "ch2voltsWL2", ..., "ch5voltsWL2"]`)

**Example:**
```python
from psafe_custom import get_sorted_ac_columns

# Get all channel names organized by wavelength
ac_columns = get_sorted_ac_columns()

# Access wavelength 1 channels
wv1_channels = ac_columns["wv1"]
print(wv1_channels)
# Output: ['ch1voltsWL1', 'ch2voltsWL1', 'ch3voltsWL1', 'ch4voltsWL1', 'ch5voltsWL1']

# Access wavelength 2 channels
wv2_channels = ac_columns["wv2"]
print(wv2_channels)
# Output: ['ch1voltsWL2', 'ch2voltsWL2', 'ch3voltsWL2', 'ch4voltsWL2', 'ch5voltsWL2']

# Select first two channels from wavelength 1
selected_channels = ac_columns["wv1"][:2]
# Result: ['ch1voltsWL1', 'ch2voltsWL1']
```

---

### get_sqi

**Signature:**
```python
def get_sqi(
    frame: np.ndarray,
    sqi_type: str = "skewness",
) -> float
```

**Description:**
Computes Signal Quality Index (SQI) for a given frame of data. This function provides various signal quality metrics to assess the quality of physiological signal frames. Different SQI types can be used depending on the application and signal characteristics.

**Parameters:**
- `frame` (np.ndarray): 1D numpy array containing the signal frame data
- `sqi_type` (str, optional): Type of SQI to compute. Defaults to `"skewness"`.
  - `"skewness"`: Magnitude (absolute value) of statistical skewness

**Returns:**
- `float`: The computed SQI value

**Raises:**
- `ValueError`: If frame is not a 1D numpy array
- `ValueError`: If sqi_type is not recognized
- `ValueError`: If frame is empty or has insufficient data (< 3 samples for skewness)

**Example:**
```python
import numpy as np
from psafe_custom import get_sqi

# Compute SQI for a single frame
frame = np.random.randn(100)
sqi_value = get_sqi(frame, sqi_type='skewness')
print(f"SQI (skewness): {sqi_value}")

# Use with computed frames from pipeline
from psafe_custom import get_sheep_dataframe, compute_frames_from_df, get_sorted_ac_columns

# Load data and compute frames
idx = 7
ac_columns = get_sorted_ac_columns()
selected_channels = ac_columns["wv1"][:1]
data = get_sheep_dataframe(idx)

frames_dict = compute_frames_from_df(
    dataframe=data,
    time_series_columns=selected_channels,
    heartrate_column='FHR',
    sampling_rate=80.0,
    average_frame_length=100,
    frames_to_average=120,
    sample_skip=80,
    interpolation_kind="linear"
)

# Compute SQI for all frames
channel = selected_channels[0]
sqi_values = [get_sqi(frame) for frame in frames_dict[channel]]
print(f"Average SQI: {np.mean(sqi_values):.4f}")
print(f"SQI range: {np.min(sqi_values):.4f} to {np.max(sqi_values):.4f}")

# Filter frames based on SQI threshold
threshold = 0.5
high_quality_frames = [
    frame for frame, sqi in zip(frames_dict[channel], sqi_values)
    if sqi < threshold
]
print(f"High quality frames: {len(high_quality_frames)} out of {len(frames_dict[channel])}")
```

**Notes:**
- Skewness-based SQI is useful for detecting signal artifacts and asymmetry
- Lower skewness magnitude often indicates better signal quality
- SQI thresholds should be determined based on your specific application and signal characteristics
- Additional SQI types can be added in future versions (e.g., kurtosis, SNR, entropy)

---

## Usage Examples

### Complete Pipeline: Loading Sheep Data and Computing Frames

This example demonstrates the complete workflow using all utility functions in `apply_pasfe.py`:

```python
from psafe_custom import get_sheep_dataframe, get_sorted_ac_columns, compute_frames_from_df

# Step 1: Select a dataset index (0-12 corresponding to the 13 available sheep experiments)
idx = 7

# Step 2: Get the sorted AC column names for wavelength 1
# This returns a dictionary with 'wv1' and 'wv2' keys
ac_columns = get_sorted_ac_columns()

# Step 3: Select the first 2 channels from wavelength 1
# You can select any subset of channels you want to process
selected_channels = ac_columns["wv1"][:2]  # ['ch1voltsWL1', 'ch2voltsWL1']

# Step 4: Load the sheep dataframe
# This returns a DataFrame with PPG channels, fSpO2, and FHR (heart rate in Hz)
data = get_sheep_dataframe(idx)

# Step 5: Process the selected channels using compute_frames_from_df
# Parameters:
#   - data: The DataFrame from get_sheep_dataframe
#   - selected_channels: List of column names to process
#   - 'FHR': Column name containing heart rate in Hz
#   - 80.0: Sampling rate in Hz
#   - 100: Target length for interpolated frames
#   - 120: Number of frames to collect and average together
#   - 80: Number of samples to skip between computing averaged frames
#   - "linear": Interpolation method
frames_dict = compute_frames_from_df(
    dataframe=data,
    time_series_columns=selected_channels,
    heartrate_column='FHR',
    sampling_rate=80.0,
    average_frame_length=100,
    frames_to_average=120,
    sample_skip=80,
    interpolation_kind="linear"
)

# Step 6: Access the results
# frames_dict is a dictionary where keys are channel names
# and values are lists of averaged frames (numpy arrays)
for channel_name, frames_list in frames_dict.items():
    print(f"{channel_name}: {len(frames_list)} averaged frames")
    print(f"  Each frame shape: {frames_list[0].shape}")
    
# Example output:
# ch1voltsWL1: 245 averaged frames
#   Each frame shape: (100,)
# ch2voltsWL1: 245 averaged frames
#   Each frame shape: (100,)

# Step 7: Further analysis with the frames
# For example, you can analyze the first averaged frame from channel 1
import numpy as np
first_frame = frames_dict['ch1voltsWL1'][0]
print(f"First frame mean: {np.mean(first_frame)}")
print(f"First frame std: {np.std(first_frame)}")
```

**Key Points:**
- This pipeline works with all 13 sheep datasets (idx 0-12)
- The FHR column is automatically converted from BPM to Hz by `get_sheep_dataframe`
- You can select any combination of channels from either wavelength
- The sampling rate for sheep data is 80 Hz
- Adjust `average_frame_length`, `frames_to_average`, and `sample_skip` based on your analysis needs

---

## Legacy Usage Examples (Individual Functions)

### Example 1 (Legacy): Basic Usage with Uniform Frame Lengths

```python
import numpy as np
from psafe_custom import compute_frame

# Create a simple sinusoidal time series
time_series = np.sin(np.linspace(0, 10 * np.pi, 1000))

# Uniform frame length of 50 samples
frame_lengths = np.full(1000, 50)

# Compute averaged frames with default linear interpolation
averaged_frames = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=50,
    frames_to_average=5,
    sample_skip=25
)

print(f"Number of averaged frames: {len(averaged_frames)}")
print(f"Shape of each frame: {averaged_frames[0].shape}")
# Output: Number of averaged frames: ~20
# Output: Shape of each frame: (50,)
```

### Example 2 (Legacy): Variable Frame Lengths with Interpolation

```python
import numpy as np
from psafe_custom import compute_frame

# Create time series
time_series = np.random.randn(500)

# Variable frame lengths (alternating between 10 and 20)
frame_lengths = np.tile([10, 20], 250)

# Interpolate all frames to length 15 and average (using default linear interpolation)
averaged_frames = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=15,
    frames_to_average=4,
    sample_skip=10
)

# All frames will have length 15 after interpolation
for i, frame in enumerate(averaged_frames):
    print(f"Frame {i}: length={len(frame)}, mean={np.mean(frame):.3f}")
```

### Example 3 (Legacy): Tracking Frame Positions with Metadata

```python
import numpy as np
from psafe_custom import compute_frame_with_metadata

# Create time series with trend
time_series = np.linspace(0, 100, 800) + np.random.randn(800) * 5

# Uniform frame lengths
frame_lengths = np.full(800, 40)

# Get frames and their starting positions
frames, start_indices = compute_frame_with_metadata(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=40,
    frames_to_average=3,
    sample_skip=20
)

# Display frame information
for i, (frame, start_idx) in enumerate(zip(frames, start_indices)):
    print(f"Frame {i}: starts at index {start_idx}, mean value={np.mean(frame):.2f}")
```

### Example 4 (Legacy): Processing Real-World Signals

```python
import numpy as np
from psafe_custom import compute_frame

# Simulate ECG-like signal
t = np.linspace(0, 10, 2000)
ecg_signal = (
    np.sin(2 * np.pi * 1.2 * t) +  # Heart rate component
    0.3 * np.sin(2 * np.pi * 4 * t) +  # Higher frequency component
    0.1 * np.random.randn(2000)  # Noise
)

# Frame lengths might vary based on detected heartbeats
base_frame_length = 80
frame_lengths = base_frame_length + np.random.randint(-5, 6, size=2000)

# Extract and average cardiac cycles
cardiac_cycles = compute_frame(
    time_series=ecg_signal,
    current_frame_length=frame_lengths,
    average_frame_length=80,
    frames_to_average=10,  # Average 10 cycles
    sample_skip=40  # Half-overlapping windows
)

print(f"Extracted {len(cardiac_cycles)} averaged cardiac cycles")
```

### Example 5 (Legacy): Using Different Interpolation Methods

```python
import numpy as np
from psafe_custom import compute_frame

# Create signal with variable frame lengths
time_series = np.sin(np.linspace(0, 20 * np.pi, 500))
frame_lengths = np.array([10, 15, 20] * 166 + [10, 15])

# Linear interpolation (default)
frames_linear = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=15,
    frames_to_average=3,
    sample_skip=10,
    interpolation_kind="linear"
)

# Cubic interpolation for smoother results
frames_cubic = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=15,
    frames_to_average=3,
    sample_skip=10,
    interpolation_kind="cubic"
)

# Nearest neighbor for step-like data
frames_nearest = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=15,
    frames_to_average=3,
    sample_skip=10,
    interpolation_kind="nearest"
)

# Quadratic interpolation
frames_quadratic = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=15,
    frames_to_average=3,
    sample_skip=10,
    interpolation_kind="quadratic"
)

print(f"Linear: {len(frames_linear)} frames")
print(f"Cubic: {len(frames_cubic)} frames")
print(f"Nearest: {len(frames_nearest)} frames")
print(f"Quadratic: {len(frames_quadratic)} frames")
```

### Example 6 (Legacy): Windowed Analysis

```python
import numpy as np
from psafe_custom import compute_frame

# Create signal with changing characteristics
signal = np.concatenate([
    np.sin(np.linspace(0, 10, 500)),  # Low frequency
    np.sin(np.linspace(0, 50, 500))   # High frequency
])

frame_lengths = np.full(1000, 50)

# Small sample_skip for detailed analysis
detailed_frames = compute_frame(
    time_series=signal,
    current_frame_length=frame_lengths,
    average_frame_length=50,
    frames_to_average=2,
    sample_skip=5  # High overlap
)

print(f"Detailed analysis: {len(detailed_frames)} frames")

# Large sample_skip for overview
overview_frames = compute_frame(
    time_series=signal,
    current_frame_length=frame_lengths,
    average_frame_length=50,
    frames_to_average=2,
    sample_skip=50  # No overlap
)

print(f"Overview analysis: {len(overview_frames)} frames")
```

### Example 7 (Legacy): Using heartrate_to_frame_length

```python
import numpy as np
from psafe_custom import compute_frame, heartrate_to_frame_length

# Create a physiological signal (e.g., EEG with cardiac artifact)
time_series = np.sin(np.linspace(0, 20 * np.pi, 2000)) + 0.2 * np.random.randn(2000)

# Heart rate varies from 0.9 to 1.1 Hz over time
heartrate_hz = np.linspace(0.9, 1.1, 2000)

# Sampling rate is 100 Hz
sampling_rate = 100.0

# Convert heart rate to frame lengths
frame_lengths = heartrate_to_frame_length(heartrate_hz, sampling_rate)

print(f"Frame length range: {frame_lengths.min()} to {frame_lengths.max()} samples")

# Now use with compute_frame
averaged_frames = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=100,  # Target uniform length
    frames_to_average=5,
    sample_skip=25
)

print(f"Computed {len(averaged_frames)} averaged frames from physiological signal")
```

### Example 8 (Legacy): Batch Processing Multiple Channels with DataFrame

```python
import pandas as pd
import numpy as np
from psafe_custom import compute_frames_from_df

# Create a dataframe with multiple EEG channels and heart rate
df = pd.DataFrame({
    'eeg_fp1': np.random.randn(2000) * 0.1 + np.sin(np.linspace(0, 20 * np.pi, 2000)),
    'eeg_fp2': np.random.randn(2000) * 0.1 + np.cos(np.linspace(0, 20 * np.pi, 2000)),
    'eeg_c3': np.random.randn(2000) * 0.15 + np.sin(np.linspace(0, 15 * np.pi, 2000)),
    'eeg_c4': np.random.randn(2000) * 0.15 + np.cos(np.linspace(0, 15 * np.pi, 2000)),
    'heartrate_hz': np.linspace(0.9, 1.1, 2000)  # Variable heart rate
})

print(f"Processing {len(df)} samples from 4 EEG channels")

# Process all EEG channels at once
results = compute_frames_from_df(
    dataframe=df,
    time_series_columns=['eeg_fp1', 'eeg_fp2', 'eeg_c3', 'eeg_c4'],
    heartrate_column='heartrate_hz',
    sampling_rate=100.0,
    average_frame_length=100,
    frames_to_average=10,
    sample_skip=50,
    interpolation_kind='cubic'
)

# Results is a dictionary with all processed channels
print(f"Processed {len(results)} channels")
for channel, frames in results.items():
    print(f"  {channel}: {len(frames)} averaged frames")

# Access specific channel results
fp1_frames = results['eeg_fp1']
print(f"\nFirst frame from FP1 channel has shape: {fp1_frames[0].shape}")
```

### Example 9 (Legacy): Error Handling

```python
import numpy as np
from psafe_custom import compute_frame

try:
    # Mismatched array lengths (will raise ValueError)
    time_series = np.arange(100)
    frame_lengths = np.full(50, 10)  # Wrong length!
    
    result = compute_frame(
        time_series=time_series,
        current_frame_length=frame_lengths,
        average_frame_length=10,
        frames_to_average=2,
        sample_skip=5
    )
except ValueError as e:
    print(f"Error caught: {e}")
    # Output: Error caught: time_series and current_frame_length must have equal length

try:
    # Invalid parameter (will raise ValueError)
    time_series = np.arange(100)
    frame_lengths = np.full(100, 10)
    
    result = compute_frame(
        time_series=time_series,
        current_frame_length=frame_lengths,
        average_frame_length=-10,  # Invalid!
        frames_to_average=2,
        sample_skip=5
    )
except ValueError as e:
    print(f"Error caught: {e}")
    # Output: Error caught: average_frame_length must be positive
```

---

## Type Signatures

For type checking and IDE support, all functions include comprehensive type hints:

```python
# Complete type signatures
from typing import List
import numpy as np

compute_frame(
    time_series: np.ndarray,           # Must be 1D float array
    current_frame_length: np.ndarray,  # Must be 1D int/float array
    average_frame_length: int,         # Positive integer
    frames_to_average: int,            # Positive integer
    sample_skip: int,                  # Positive integer
    interpolation_kind: str = "linear", # Interpolation method string
) -> List[np.ndarray]                  # List of 1D arrays

compute_frame_with_metadata(
    time_series: np.ndarray,
    current_frame_length: np.ndarray,
    average_frame_length: int,
    frames_to_average: int,
    sample_skip: int,
    interpolation_kind: str = "linear", # Interpolation method string
) -> tuple[List[np.ndarray], List[int]]  # Tuple of (frames, indices)
```

### Available Interpolation Methods

The `interpolation_kind` parameter accepts the following string values (from scipy.interpolate.interp1d):

- **'linear'** (default): Linear interpolation between points
- **'nearest'**: Nearest neighbor interpolation
- **'nearest-up'**: Nearest neighbor with preference for upper value
- **'zero'**: Zero-order spline (piecewise constant)
- **'slinear'**: First-order spline (equivalent to linear)
- **'quadratic'**: Second-order spline interpolation
- **'cubic'**: Third-order spline interpolation
- **'previous'**: Use previous value
- **'next'**: Use next value

**Choosing an Interpolation Method:**
- **Linear**: Good default, fast, smooth but not as accurate for curved data
- **Cubic**: Smoother results for curved signals, slower than linear
- **Quadratic**: Balance between linear and cubic
- **Nearest**: Best for discrete/step data, preserves exact values
- **Previous/Next**: For forward/backward filling patterns

### Utility Function Type Signatures

```python
from typing import Union
import numpy as np

heartrate_to_frame_length(
    heartrate: np.ndarray,        # 1D array of heart rates in Hz
    sampling_rate: float,         # Sampling rate in Hz
) -> np.ndarray                   # 1D array of frame lengths (integers)

compute_frames_from_df(
    dataframe: pd.DataFrame,           # DataFrame with time series and heart rate
    time_series_columns: list[str],    # List of column names to process
    heartrate_column: str,             # Name of heart rate column
    sampling_rate: float,              # Sampling rate in Hz
    average_frame_length: int,         # Target frame length
    frames_to_average: int,            # Number of frames to average
    sample_skip: int,                  # Skip distance
    interpolation_kind: str = "linear", # Interpolation method
) -> dict[str, list[np.ndarray]]      # Dict of column name -> list of frames
```

---

## Common Patterns for Agentic AI

### Pattern 1: Time Series Segmentation
Use when you need to break down a long time series into manageable, averaged segments.

### Pattern 2: Signal Averaging
Use when you want to reduce noise by averaging multiple signal periods together.

### Pattern 3: Feature Extraction
Extract averaged frames as features for machine learning models.

### Pattern 4: Adaptive Windowing
Use variable frame lengths to handle signals with non-uniform characteristics.

### Pattern 5: Overlap Analysis
Use small `sample_skip` values for sliding window analysis with overlap.

---

## Performance Considerations

1. **Memory Usage**: Each averaged frame requires memory. Large `average_frame_length` or many output frames will increase memory usage.

2. **Interpolation Cost**: Variable frame lengths require interpolation, which adds computational cost. Uniform frame lengths matching `average_frame_length` are fastest. Higher-order interpolation methods (cubic, quadratic) are slower than linear.

3. **Sample Skip Trade-off**: Smaller `sample_skip` produces more frames but increases computation time.

4. **Frames to Average**: Larger `frames_to_average` values require more data and produce fewer output frames but better noise reduction.

---

## Testing

Run the test suite:

```bash
pytest tests/test_compute_frame.py -v
```

Run with coverage:

```bash
pytest tests/test_compute_frame.py --cov=psafe_custom.compute_frame --cov-report=html
```

---

## Version History

### v0.1.0
- Initial release
- `compute_frame` function with full validation
- `compute_frame_with_metadata` for position tracking
- `heartrate_to_frame_length` utility function for physiological signal processing
- Comprehensive test suite
- Flexible interpolation for variable frame lengths (linear, cubic, quadratic, nearest, etc.)

---

## Future Enhancements

Potential additions for future versions:
- Additional interpolation methods (cubic, spline)
- Parallel processing for large datasets
- GPU acceleration support
- Windowing functions (Hamming, Hanning, etc.)
- Real-time streaming mode
- Support for multi-dimensional time series

---

## Contact & Support

For issues, questions, or contributions, please refer to the project repository.

---

*This documentation is designed for both human developers and agentic AI systems to understand and utilize the psafe_custom library effectively.*