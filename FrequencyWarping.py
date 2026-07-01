from scipy.interpolate import interp1d
import numpy as np
import processing.filters as myfilters
from scipy.signal import find_peaks
def frequency_normalize(signal, inst_freq, fs):
    t = np.arange(len(signal)) / fs
    phase = np.cumsum(inst_freq) / fs
    t_new = phase
    interp_func = interp1d(t_new, signal, kind='linear', bounds_error=False, fill_value="extrapolate")
    t_uniform = np.arange(0, t_new[-1], 1/fs)
    norm_signal = interp_func(t_uniform)
    return norm_signal, t_uniform


def unwarp_to_original_time(norm_signal, t_norm, inst_freq, fs):
    # Reconstruct the mapping from original time to warped time
    t = np.arange(len(inst_freq)) / fs
    phase = np.cumsum(inst_freq) / fs
    t_new = phase  # this is the warped time corresponding to each original t

    # Interpolate the normalized signal (which is on t_norm) back to t_new
    interp_func = interp1d(t_norm, norm_signal, kind='linear', bounds_error=False, fill_value="extrapolate")
    unwarped_signal = interp_func(t_new)
    return unwarped_signal, t
    
def frequency_warpping_signal_extraction(mixed_signal, fs, target_freq,number_of_harmonics,order=3):

    # Apply frequency normalization on mixed_signal using target_freq
    norm_signal, t_uniform = frequency_normalize(mixed_signal, target_freq, fs)
    filtered_sum = np.zeros_like(norm_signal)
    for f in range(1, number_of_harmonics+1):
        bw = 0.02 * f + 0.05
        low_cutoff = f - bw/2
        high_cutoff = f + bw/2
        filtered = myfilters.butter_bandpass_filter(norm_signal, low_cutoff, high_cutoff, fs,order=order)
        filtered_sum += filtered
    # Convert the filtered_sum (which is in normalized time) back to original time
    filtered_sum_unwarped, _ = unwarp_to_original_time(filtered_sum, t_uniform, target_freq, fs)
    filtered_sum = filtered_sum_unwarped
    return filtered_sum
def peak_detection_with_mask(signal, fs, target_freq,order=3,height=0,distance=33,wlen=20,prominence=0.0001):

    first_harmonic_signal = frequency_warpping_signal_extraction(signal, fs, target_freq, 1, order)
    signal_masked = signal.copy()
    signal_masked[first_harmonic_signal < 0] = 0
    peaks, _ = find_peaks(signal_masked, height=height, distance=distance, wlen=wlen, prominence=prominence)
    return peaks