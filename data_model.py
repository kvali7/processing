from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """One averaged Lambert frame.

    Attributes
    ----------
    template:
        Shape ``(resample_len, 10)``.  Time-domain average of ``frames_to_average``
        consecutive beats across all 10 PPG channels (WL1 cols 0-4, WL2 cols 5-9).
    crop_start_idx:
        Shape ``(frames_to_average,)``.  Start sample index (into the original
        dataset, before any HP filtering) of each accepted beat in this frame.
    crop_end_idx:
        Shape ``(frames_to_average,)``.  Exclusive end sample index of each
        accepted beat in this frame.  ``ppg[crop_start_idx[k]:crop_end_idx[k]]``
        gives the raw samples of beat k.
    dc:
        Shape ``(10,)``.  DC baseline per channel: the per-channel minimum of the
        averaged *raw* (pre-HP-filter) beat template.  Mirrors the ``col_min``
        quantity produced by ``PSA.compute_DC``.
    """

    template: np.ndarray
    crop_start_idx: np.ndarray
    crop_end_idx: np.ndarray
    dc: np.ndarray


@dataclass
class DataStream:
    """All data associated with one experiment (one sheep recording).

    Attributes
    ----------
    ppg:
        Shape ``(N, 10)``.  Raw PPG voltage channels.  Columns 0-4 are WL1
        (ch1-ch5), columns 5-9 are WL2 (ch1-ch5).  May be mutated in-place by
        pipeline steps (e.g. high-pass filtering).
    fsao2:
        Shape ``(N,)``.  Fetal SpO2 signal at the recording sample rate.
    hr:
        Shape ``(N,)``.  Heart-rate signal at the recording sample rate (units
        match the source column, e.g. beats-per-minute for MHR/FHR).
    frames:
        Averaged Lambert frames produced by ``FrameBuilder``.  Empty until the
        pipeline has run.
    extras:
        Scratch space for intermediate pipeline outputs (e.g. ``"hr_smooth"``,
        ``"peaks"``).  Values may be any numpy array.
    """

    ppg: np.ndarray
    fsao2: np.ndarray
    hr: np.ndarray
    experiment_id: str = ""
    ppg_col_names: list[str] = field(default_factory=list)
    fs: float = 80.0
    frames: list[Frame] = field(default_factory=list)
    extras: dict[str, np.ndarray] = field(default_factory=dict)
