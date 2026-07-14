from __future__ import annotations

import numpy as np
import pandas as pd


class SequenceStandardScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "SequenceStandardScaler":
        flat = x.reshape(-1, x.shape[-1])
        self.mean_ = flat.mean(axis=0).astype(np.float32)
        self.std_ = flat.std(axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceStandardScaler must be fit before transform")
        out = (x - self.mean_) / self.std_
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def make_per_turbine_sequences(
    table: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    target_col: str = "turbine_target",
    official_col: str = "official_target",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    if len(table) == 0:
        return (
            np.empty((0, window, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.Series([], dtype="datetime64[ns]"),
            np.empty((0,), dtype=np.int16),
        )

    work = table.sort_values("forecast_kst_dtm").copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    sequence_parts = []
    target_parts = []
    official_parts = []
    time_parts = []
    year_parts = []

    for year, year_df in work.groupby(work["forecast_kst_dtm"].dt.year, sort=True):
        year_df = year_df.sort_values("forecast_kst_dtm").reset_index(drop=True)
        values = year_df[feature_cols].to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        padded = np.concatenate(
            [np.repeat(values[:1], window - 1, axis=0), values],
            axis=0,
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=window,
            axis=0,
        ).transpose(0, 2, 1)
        sequence_parts.append(np.ascontiguousarray(windows, dtype=np.float32))
        target_parts.append(
            pd.to_numeric(year_df.get(target_col, np.nan), errors="coerce").to_numpy(np.float32)
        )
        official_parts.append(
            pd.to_numeric(year_df.get(official_col, np.nan), errors="coerce").to_numpy(np.float32)
        )
        time_parts.append(year_df["forecast_kst_dtm"].to_numpy())
        year_parts.append(np.full(len(year_df), int(year), dtype=np.int16))

    return (
        np.concatenate(sequence_parts, axis=0),
        np.concatenate(target_parts),
        np.concatenate(official_parts),
        pd.Series(np.concatenate(time_parts)),
        np.concatenate(year_parts),
    )
