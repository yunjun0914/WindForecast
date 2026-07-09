import numpy as np
import pandas as pd

from utils.preprocessing import TIME_KEY_COLS
from utils.tree_feature_profiles import FEATURE_PROFILE_AGGRESSIVE_MINIMAL_CONTEXT_V1, build_tree_features


SEQNN_V1_FEATURES = [
    "sin_doy",
    "cos_doy",
    "sin_hod",
    "cos_hod",
    "gfs_ws100_speed",
    "gfs_ws850_speed",
    "gfs_ws10_speed",
    "ldaps_ws50_max_speed",
    "ldaps_ws10_speed",
    "gfs_surface_0_gust",
    "phys_gfs_gust_factor",
    "phys_gfs_gust_minus_ws10",
    "phys_gfs_air_density_x_gfs_ws850_speed_cube",
    "phys_gfs_air_density_x_gfs_ws100_speed_cube",
    "phys_ldaps_air_density_x_ldaps_ws50_max_speed_cube",
    "phys_shear_gfs_850_100",
    "phys_shear_gfs_100_10",
    "phys_shear_ldaps_50max_10",
    "phys_gfs_ws850_grid_max",
    "phys_gfs_ws850_grid_p90",
    "phys_gfs_ws100_grid_max",
    "phys_ldaps_ws50max_grid_max",
    "phys_ldaps_ws50max_grid_p90",
    "forecast_lead_hours",
    "forecast_lead_mod24_sin",
    "forecast_lead_mod24_cos",
    "data_available_hod_sin",
    "data_available_hod_cos",
    "gfs_ws850_dir_cos",
    "gfs_ws850_dir_sin",
    "gfs_ws100_dir_cos",
    "gfs_ws100_dir_sin",
    "ldaps_ws50max_dir_cos",
    "ldaps_ws50max_dir_sin",
]


YEARS = [2022, 2023, 2024]


def available_seqnn_features(weather, feature_names=SEQNN_V1_FEATURES):
    return [col for col in feature_names if col in weather.columns]


def build_seqnn_weather(ldaps, gfs, group, feature_profile=FEATURE_PROFILE_AGGRESSIVE_MINIMAL_CONTEXT_V1):
    weather = build_tree_features(ldaps, gfs, group, feature_profile=feature_profile)
    cols = available_seqnn_features(weather)
    if not cols:
        raise ValueError("no SeqNN features were found in the weather table")
    return weather[[*TIME_KEY_COLS, *cols]].copy()


def build_group_table(weather, labels, group):
    labels = labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    table = weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    table = table.dropna(subset=[group]).reset_index(drop=True)
    table["year"] = pd.to_datetime(table["forecast_kst_dtm"]).dt.year
    return table.rename(columns={group: "target"})


def split_year_fold(table, pred_year):
    train_years = [year for year in YEARS if year != pred_year]
    train = table[table["year"].isin(train_years)].copy().reset_index(drop=True)
    val = table[table["year"] == pred_year].copy().reset_index(drop=True)
    return train, val, train_years


def make_sequences(table, feature_cols, window=24, target_col="target"):
    if len(table) == 0:
        return (
            np.empty((0, window, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.Series([], dtype="datetime64[ns]"),
        )

    sequences = []
    targets = []
    times = []
    work = table.sort_values("forecast_kst_dtm").copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])

    for _, year_df in work.groupby(work["forecast_kst_dtm"].dt.year, sort=True):
        year_df = year_df.sort_values("forecast_kst_dtm").reset_index(drop=True)
        values = year_df[feature_cols].to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        y = year_df[target_col].to_numpy(dtype=np.float32) if target_col in year_df.columns else None

        for idx in range(len(year_df)):
            start = max(0, idx - window + 1)
            chunk = values[start : idx + 1]
            if len(chunk) < window:
                pad = np.repeat(chunk[:1], window - len(chunk), axis=0)
                chunk = np.vstack([pad, chunk])
            sequences.append(chunk)
            if y is not None:
                targets.append(y[idx])
            times.append(year_df.loc[idx, "forecast_kst_dtm"])

    x = np.asarray(sequences, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32) if targets else np.empty((len(x),), dtype=np.float32)
    return x, target, pd.Series(times)


class SequenceStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        flat = x.reshape(-1, x.shape[-1])
        self.mean_ = flat.mean(axis=0).astype(np.float32)
        self.std_ = flat.std(axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_).astype(np.float32)
        return self

    def transform(self, x):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceStandardScaler must be fit before transform")
        out = (x - self.mean_) / self.std_
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(self, x):
        return self.fit(x).transform(x)
