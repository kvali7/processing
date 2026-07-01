from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from psafe_custom.apply_pasfe import (
    get_sheep_dataframe,
    get_sheep_experiment_id,
    get_sorted_ac_columns,
)
from processing.data_model import DataStream


class SheepDataLoader:
    """Load a single sheep experiment into a :class:`DataStream`.

    Parameters
    ----------
    heart_rate_col:
        Column name in the sheep dataframe to use as the HR signal.
        Typically ``"MHR"`` (maternal) or ``"FHR"`` (fetal).
    """

    def __init__(self, heart_rate_col: str = "FHR") -> None:
        self.heart_rate_col = heart_rate_col

    def load(self, sheep_id: int) -> DataStream:
        """Return a :class:`DataStream` for *sheep_id*.

        Parameters
        ----------
        sheep_id:
            Zero-based sheep index passed to ``get_sheep_dataframe``.
        """
        sheep_df = get_sheep_dataframe(sheep_id).reset_index(drop=True)

        ac_cols = get_sorted_ac_columns()
        ppg_cols = ac_cols["wl1"] + ac_cols["wl2"]  # WL1 first 5, WL2 last 5

        exp_id_raw = get_sheep_experiment_id(sheep_id)["experiment_id"]
        exp_id = str(exp_id_raw).strip()
        if not exp_id.lower().endswith(".csv"):
            exp_id = f"{exp_id}.csv"

        ppg = sheep_df[ppg_cols].to_numpy(dtype=float)       # (N, 10)
        fsao2 = sheep_df["fSpO2"].to_numpy(dtype=float)      # (N,)
        hr = sheep_df[self.heart_rate_col].to_numpy(dtype=float)  # (N,)

        return DataStream(ppg=ppg, fsao2=fsao2, hr=hr, experiment_id=exp_id, ppg_col_names=ppg_cols)


class CsvDataLoader:
    """Load any CSV into a :class:`DataStream` by column name.

    Parameters
    ----------
    file_path:
        Path to the CSV file. The stem (filename without ``.csv``) becomes the
        ``experiment_id``.
    ppg_col_names:
        Ordered list of PPG column names. Must be present in the CSV.
    hr_col:
        Column name for heart rate. Must be present in the CSV.
    fsao2_col:
        Column name for fSpO2. If ``None``, filled with zeros.
    """

    def __init__(
        self,
        file_path: str | Path,
        ppg_col_names: list[str],
        hr_col: str,
        fsao2_col: str | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.ppg_col_names = ppg_col_names
        self.hr_col = hr_col
        self.fsao2_col = fsao2_col

    def load(self) -> DataStream:
        df = pd.read_csv(self.file_path).reset_index(drop=True)
        ppg = df[self.ppg_col_names].to_numpy(dtype=float)
        hr = df[self.hr_col].to_numpy(dtype=float)
        fsao2 = (
            df[self.fsao2_col].to_numpy(dtype=float)
            if self.fsao2_col is not None
            else np.zeros(len(ppg), dtype=float)
        )
        return DataStream(
            ppg=ppg,
            fsao2=fsao2,
            hr=hr,
            experiment_id=self.file_path.stem,
            ppg_col_names=self.ppg_col_names,
        )
