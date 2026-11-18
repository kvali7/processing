"""Run the full Lambert frame pipeline from an arbitrary CSV file.

Edit the Config section below to point at your file and name the columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing import (  # noqa: E402
    CsvDataLoader,
    Pipeline,
    HighPassFilter,
    HRSmoother,
    PeakDetector,
    FrameBuilder,
    export_to_csv,
)

# ---------------------------------------------------------------------------
# Config — edit here
# ---------------------------------------------------------------------------
INPUT_CSV = ROOT / "processing" / "data" / "patient3.csv"
OUT_CSV   = ROOT / "processing" / "data" / "ppg_frames_patient3.csv"

PPG_COL_NAMES = ["wl1_1", "wl1_2", "wl1_3", "wl1_5",
                 "wl2_1", "wl2_2", "wl2_3", "wl2_5", ]
HR_COL    = "FHR"
FSAO2_COL = None   # set to column name if available, else zeros

FRAMES_TO_AVERAGE: int = 120
FS: float = 80.0
HP_CUTOFF_HZ: float = 1.0
HP_ORDER: int = 4
HR_MA_WINDOW_SECONDS: int = 60
SEGMENT_TOLERANCE_RATIO: float = 0.2

# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
loader = CsvDataLoader(
    file_path=INPUT_CSV,
    ppg_col_names=PPG_COL_NAMES,
    hr_col=HR_COL,
    fsao2_col=FSAO2_COL,
)

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
    print(f"Processing {INPUT_CSV.name}...")
    stream = loader.load()
    pipeline.run(stream)
    print(f"  → {len(stream.frames)} frames built")
    export_to_csv([stream], OUT_CSV)
