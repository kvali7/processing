# psafe-custom

A custom Python library for time series frame extraction and averaging with variable-length frame support.

## Features

- **Frame Extraction**: Extract variable-length frames from time series data
- **Flexible Interpolation**: Support for multiple interpolation methods (linear, cubic, quadratic, nearest, etc.)
- **Averaging**: Compute averaged frames from multiple time series segments
- **Metadata Tracking**: Track starting positions of averaged frames
- **Utility Functions**: Heart rate to frame length conversion for physiological signals
- **Batch Processing**: Process multiple DataFrame columns at once
- **Type Safety**: Full type hints for all functions
- **Comprehensive Testing**: Extensive test suite included

## Installation

### Basic Installation

Install the package in development mode:

```bash
pip install -e .
```

### With Development Dependencies

Install with testing and formatting tools:

```bash
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0

## Quick Start

```python
import numpy as np
import pandas as pd
from psafe_custom import compute_frame, heartrate_to_frame_length, compute_frames_from_df

# Option 1: Single time series
time_series = np.sin(np.linspace(0, 10 * np.pi, 1000))
frame_lengths = np.full(1000, 50)

averaged_frames = compute_frame(
    time_series=time_series,
    current_frame_length=frame_lengths,
    average_frame_length=50,
    frames_to_average=5,
    sample_skip=25
)

# Option 2: Multiple channels from DataFrame
df = pd.DataFrame({
    'ch1': np.sin(np.linspace(0, 20 * np.pi, 1000)),
    'ch2': np.cos(np.linspace(0, 20 * np.pi, 1000)),
    'heartrate': np.full(1000, 1.0)
})

results = compute_frames_from_df(
    dataframe=df,
    time_series_columns=['ch1', 'ch2'],
    heartrate_column='heartrate',
    sampling_rate=100.0,
    average_frame_length=100,
    frames_to_average=5,
    sample_skip=25
)

print(f"Processed {len(results)} channels")
```

## Main Functions

### `compute_frame()`

Extracts frames from a time series, interpolates them to uniform length, and computes averaged frames.

**Parameters:**
- `time_series`: 1D numpy array of time series data
- `current_frame_length`: 1D numpy array of frame lengths
- `average_frame_length`: Target length for interpolated frames (int)
- `frames_to_average`: Number of frames to average together (int)
- `sample_skip`: Skip distance between frames (int)
- `interpolation_kind`: Type of interpolation ('linear', 'cubic', 'quadratic', 'nearest', etc.) - defaults to 'linear'

**Returns:**
- List of averaged frames (each is a numpy array)

### `compute_frame_with_metadata()`

Same as `compute_frame()` but also returns starting indices for each averaged frame.

**Returns:**
- Tuple of (frames, start_indices)

### `heartrate_to_frame_length()`

Converts heart rate (in Hz) to frame lengths in samples.

**Parameters:**
- `heartrate`: 1D numpy array of heart rate values in Hz
- `sampling_rate`: Sampling rate in Hz (float)

**Returns:**
- Array of frame lengths (integers)

**Example:**
```python
import numpy as np
from psafe_custom import heartrate_to_frame_length

# Variable heart rate
heartrate = np.array([1.0, 1.2, 0.8, 1.5])
frame_lengths = heartrate_to_frame_length(heartrate, 100.0)
# Result: array([100, 83, 125, 67])
```

### `compute_frames_from_df()`

Batch process multiple time series columns from a DataFrame.

**Parameters:**
- `dataframe`: DataFrame containing time series and heart rate data
- `time_series_columns`: List of column names to process
- `heartrate_column`: Name of the column with heart rate (Hz)
- `sampling_rate`: Sampling rate in Hz (float)
- `average_frame_length`: Target frame length (int)
- `frames_to_average`: Number of frames to average (int)
- `sample_skip`: Skip distance (int)
- `interpolation_kind`: Interpolation method (default: 'linear')

**Returns:**
- Dictionary mapping column names to lists of averaged frames

**Example:**
```python
import pandas as pd
import numpy as np
from psafe_custom import compute_frames_from_df

df = pd.DataFrame({
    'eeg_ch1': np.random.randn(2000),
    'eeg_ch2': np.random.randn(2000),
    'heartrate': np.full(2000, 1.0)
})

results = compute_frames_from_df(
    dataframe=df,
    time_series_columns=['eeg_ch1', 'eeg_ch2'],
    heartrate_column='heartrate',
    sampling_rate=100.0,
    average_frame_length=100,
    frames_to_average=10,
    sample_skip=50
)

# Access results: results['eeg_ch1'], results['eeg_ch2']
```

## Documentation

For comprehensive documentation, examples, and usage patterns, see:
- **[docs.md](docs.md)** - Complete API reference with examples for agentic AI
- **[example_usage.py](example_usage.py)** - Runnable examples demonstrating all features

## Development

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=psafe_custom --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_compute_frame.py -v
```

### Code Formatting

Format code with Black:
```bash
black src/
```

### Type Checking

Check types with mypy:
```bash
mypy src/
```

### Running Examples

Execute the example script:
```bash
python example_usage.py
```

## Project Structure

```
psafe_custom/
├── src/
│   └── psafe_custom/
│       ├── __init__.py
│       └── compute_frame.py
├── tests/
│   ├── __init__.py
│   ├── test_compute_frame.py
│   └── test_example.py
├── pyproject.toml
├── README.md
├── docs.md
└── example_usage.py
```

## Use Cases

- **Signal Processing**: Average periodic signals to reduce noise
- **Time Series Analysis**: Extract and analyze recurring patterns
- **Feature Extraction**: Generate uniform-length features from variable-length segments
- **Adaptive Windowing**: Handle signals with non-uniform characteristics
- **ECG/EEG Processing**: Average cardiac or neural cycles for artifact removal
- **Physiological Signal Processing**: Remove cardiac artifacts from EEG, EMG, or other biosignals

## License

MIT License