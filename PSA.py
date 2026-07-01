import numpy as np

from collections import Counter
import matplotlib.pyplot as plt


class Segment:
        def __init__(self, array, time):
            self.Array = array
            self.Time = time

def signal_chopping(signal, Fs, markers, ReferenceSignalFrequency_Hz,num_of_columns,torlaerance_ratio=0.2):
    # Apply frequency normalization on mixed_signal using target_freq
    cropped_PPG_segments = []
    idxlen_diff_array = []
    for i in range(len(markers)-1):
        start_idx = markers[i]
        end_idx = markers[i+1]
        idxlen = end_idx - start_idx
        segment = signal[start_idx:end_idx]
        # Resample segment to length 100 using linear interpolation, preserving amplitude
        # Resample each column of the segment (assuming segment is 2D: shape (N, 10))
        resampled_cols = []
        x_old = np.linspace(0, 1, len(segment))
        x_new = np.linspace(0, 1, 100)
        if num_of_columns == 1:
            # segment is (N, 1) or (N,), handle both
            if segment.ndim == 2:
                resampled_col = np.interp(x_new, x_old, segment[:, 0])
            else:
                resampled_col = np.interp(x_new, x_old, segment)
            resampled_cols.append(resampled_col)
        else:
            for col in range(num_of_columns):
                resampled_col = np.interp(x_new, x_old, segment[:, col])
                resampled_cols.append(resampled_col)
        
        segment = np.stack(resampled_cols, axis=1)  # shape (100, 10)
        time = np.floor((start_idx + end_idx) / 2 / Fs).astype(int)
        expected_len = Fs/ReferenceSignalFrequency_Hz[time]
        torlaerance = torlaerance_ratio*expected_len
        idxlen_diff = idxlen - expected_len
        idxlen_diff_array.append(idxlen_diff)
        if idxlen > expected_len - torlaerance and idxlen < expected_len + torlaerance:
            cropped_PPG_segments.append(Segment(segment, time))

    return cropped_PPG_segments, idxlen_diff_array

def plot_bar_chart(idxlen_diff_array):
    # Round the differences to nearest integer for counting occurrences
    rounded_diffs = [int(round(x)) for x in idxlen_diff_array]
    diff_counts = Counter(rounded_diffs)
    plt.figure(figsize=(10,4))
    plt.bar(diff_counts.keys(), diff_counts.values())
    plt.title('Frequency of idxlen_diff_array Values')
    plt.xlabel('idxlen - expected_len (rounded)')
    plt.ylabel('Count')
    plt.show()

def compute_DC(cropped_PPG_segments, window, t_max):
    """
    Computes averaged segments and amplitude differences for each window.

    Parameters:
    - cropped_PPG_segments: list of Segment objects, each with .Array and .Time
    - window: size of the time window (default 60)
    - t_max: maximum time to iterate over (default 2200)

    Returns:
    - AveragedSegArray: list of averaged segment arrays
    - Amp: list of amplitude difference arrays (col_max - col_min for each column)
    """
    AveragedSegArray = []
    DC = []
    for t in range(t_max):
        segments_in_window = [segment for segment in cropped_PPG_segments if t <= segment.Time < t + window]
        if segments_in_window:
            avg_segment = np.mean([segment.Array for segment in segments_in_window], axis=0)
            AveragedSegArray.append(avg_segment)
            # Find min and max for each column in avg_segment, calculate the difference, and store in an array with 10 columns
            col_min = np.min(avg_segment, axis=0)
            col_diff = col_min
            DC.append(col_diff)
        else:
            print("No segments in window")
            DC.append(-11111)
    # Replace any -11111 in DC with the nearest normal value
    # Find indices where DC == -11111
    for i in range(len(DC)):
        if isinstance(DC[i], (int, float)) and DC[i] == -11111:
            # Find nearest previous and next non -11111 values
            prev_idx = i - 1
            next_idx = i + 1
            while prev_idx >= 0 and (isinstance(DC[prev_idx], (int, float)) and DC[prev_idx] == -11111):
                prev_idx -= 1
            while next_idx < len(DC) and (isinstance(DC[next_idx], (int, float)) and DC[next_idx] == -11111):
                next_idx += 1
            prev_val = DC[prev_idx] if prev_idx >= 0 and not (isinstance(DC[prev_idx], (int, float)) and DC[prev_idx] == -11111) else None
            next_val = DC[next_idx] if next_idx < len(DC) and not (isinstance(DC[next_idx], (int, float)) and DC[next_idx] == -11111) else None
            if prev_val is not None and next_val is not None:
                # Average the two
                DC[i] = (np.array(prev_val) + np.array(next_val)) / 2
            elif prev_val is not None:
                DC[i] = np.array(prev_val)
            elif next_val is not None:
                DC[i] = np.array(next_val)
            else:
                # If all are -11111, set to zeros of length 10
                DC[i] = np.zeros(10)
    return AveragedSegArray, DC

def compute_averaged_segments_and_amplitudes(cropped_PPG_segments, window, t_max):
    """
    Computes averaged segments and amplitude differences for each window.

    Parameters:
    - cropped_PPG_segments: list of Segment objects, each with .Array and .Time
    - window: size of the time window (default 60)
    - t_max: maximum time to iterate over (default 2200)

    Returns:
    - AveragedSegArray: list of averaged segment arrays
    - Amp: list of amplitude difference arrays (col_max - col_min for each column)
    """
    AveragedSegArray = []
    Amp = []
    for t in range(t_max):
        segments_in_window = [segment for segment in cropped_PPG_segments if t <= segment.Time < t + window]
        if segments_in_window:
            avg_segment = np.mean([segment.Array for segment in segments_in_window], axis=0)
            AveragedSegArray.append(avg_segment)
            # Find min and max for each column in avg_segment, calculate the difference, and store in an array with 10 columns
            col_min = np.min(avg_segment, axis=0)
            col_max = np.max(avg_segment, axis=0)
            col_diff = col_max - col_min
            Amp.append(col_diff)
        else:
            print("No segments in window")
            Amp.append(-11111)
    # Replace any -11111 in Amp with the nearest normal value

    # Find indices where Amp == -11111
    for i in range(len(Amp)):
        if isinstance(Amp[i], (int, float)) and Amp[i] == -11111:
            # Find nearest previous and next non -11111 values
            prev_idx = i - 1
            next_idx = i + 1
            while prev_idx >= 0 and (isinstance(Amp[prev_idx], (int, float)) and Amp[prev_idx] == -11111):
                prev_idx -= 1
            while next_idx < len(Amp) and (isinstance(Amp[next_idx], (int, float)) and Amp[next_idx] == -11111):
                next_idx += 1
            prev_val = Amp[prev_idx] if prev_idx >= 0 and not (isinstance(Amp[prev_idx], (int, float)) and Amp[prev_idx] == -11111) else None
            next_val = Amp[next_idx] if next_idx < len(Amp) and not (isinstance(Amp[next_idx], (int, float)) and Amp[next_idx] == -11111) else None
            if prev_val is not None and next_val is not None:
                # Average the two
                Amp[i] = (np.array(prev_val) + np.array(next_val)) / 2
            elif prev_val is not None:
                Amp[i] = np.array(prev_val)
            elif next_val is not None:
                Amp[i] = np.array(next_val)
            else:
                # If all are -11111, set to zeros of length 10
                Amp[i] = np.zeros(10)
    return AveragedSegArray, Amp