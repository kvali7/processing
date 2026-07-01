from scipy.signal import butter, filtfilt, sosfiltfilt, freqz_sos
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="lowpass")
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="highpass")
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def unit_gain_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4, axis: int = -1
) -> np.ndarray:
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    center_freq = np.sqrt(lowcut * highcut)
    w, h = freqz_sos(sos, worN=[center_freq], fs=fs)
    center_gain = np.abs(h[0])
    sos[0, :3] /= np.sqrt(center_gain)
    y = sosfiltfilt(sos, data, axis=axis)
    return y


def unit_gain_highpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 4, axis: int = -1
) -> np.ndarray:
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    w, h = freqz_sos(sos, worN=[cutoff], fs=fs)
    cutoff_gain = np.abs(h[0])
    sos[0, :3] /= np.sqrt(cutoff_gain)
    y = sosfiltfilt(sos, data, axis=axis)
    return y


