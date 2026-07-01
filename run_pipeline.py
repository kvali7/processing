"""Run the full Lambert frame pipeline: load → process → export.

Flip between FHR and MHR by editing HEART_RATE_COL and OUT_CSV below.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing import (  # noqa: E402
    SheepDataLoader,
    Pipeline,
    HighPassFilter,
    HRSmoother,
    PeakDetector,
    FrameBuilder,
    export_to_csv,
)

# ---------------------------------------------------------------------------
# Config — edit here to switch between MHR / FHR runs
# ---------------------------------------------------------------------------
FRAMES_TO_AVERAGE = 120
# OUT_CSV = ROOT / "data" / "ppg_frames.csv"
OUT_CSV = ROOT / "data" / "ppg_frames_test.csv"

FS: float = 80.0
HP_CUTOFF_HZ: float = 1.0
HP_ORDER: int = 4
HR_MA_WINDOW_SECONDS: int = 60
HEART_RATE_COL = "FHR"
SEGMENT_TOLERANCE_RATIO: float = 0.2

SHEEP_INDICES = range(13)

# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
loader = SheepDataLoader(heart_rate_col=HEART_RATE_COL)

pipeline = Pipeline([
    HighPassFilter(cutoff_hz=HP_CUTOFF_HZ, order=HP_ORDER),
    HRSmoother(window_seconds=HR_MA_WINDOW_SECONDS),
    PeakDetector(channel_idx=3),
    FrameBuilder(
        frames_to_average=FRAMES_TO_AVERAGE,
        stride=2,
        resample_len=100,
        segment_tolerance_ratio=SEGMENT_TOLERANCE_RATIO,
    ),
])

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    streams = []
    for sheep_id in SHEEP_INDICES:
        print(f"Processing sheep_id={sheep_id}...")
        stream = loader.load(sheep_id)
        pipeline.run(stream)
        print(f"  → {len(stream.frames)} frames built")
        streams.append(stream)

    export_to_csv(streams, OUT_CSV)
