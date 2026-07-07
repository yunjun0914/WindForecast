import re

import numpy as np
import pandas as pd


LDAPS_LEVELS = {
    "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "ws50max": ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
    "ws50min": ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin"),
    "ws5bl": ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
}

GFS_LEVELS = {
    "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "ws80": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
    "ws100": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    "wspbl": ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
    "ws850": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
    "ws700": ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
    "ws500": ("isobaricInhPa_500_u", "isobaricInhPa_500_v"),
}

# Selected from results/effective_wind_location_group_candidates.csv.
# These are the stable top single-feature signals for group1/group2/group3 SCADA ws
# on a 2022-2023 -> 2024 holdout.
COMMON_EFFECTIVE_WIND_CANDIDATES = [
    "gfs__grid1__ws850",
    "gfs__grid4__ws850",
    "gfs__grid5__ws850",
    "ldaps__grid12__ws10",
    "ldaps__grid13__ws10",
    "ldaps__grid7__ws10",
    "ldaps__grid13__ws50max",
    "ldaps__grid12__ws50max",
]


def _speed(u, v):
    return np.sqrt(u**2 + v**2)


def _sanitize(name):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name)


def _parse_candidate(candidate):
    source, grid_part, level = candidate.split("__")
    return source, int(grid_part.replace("grid", "")), level


def _candidate_series(df, candidate):
    source, grid_id, level = _parse_candidate(candidate)
    levels = LDAPS_LEVELS if source == "ldaps" else GFS_LEVELS
    work = df.copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    work = work[work["grid_id"].astype(int) == grid_id].copy()
    if level == "gust":
        value = work["surface_0_gust"].to_numpy(float)
    else:
        u_col, v_col = levels[level]
        value = _speed(work[u_col].to_numpy(float), work[v_col].to_numpy(float))
    out = pd.DataFrame({"forecast_kst_dtm": work["forecast_kst_dtm"], f"eff_{_sanitize(candidate)}": value})
    return out.groupby("forecast_kst_dtm", as_index=False).mean()


def build_effective_wind_features(ldaps_df, gfs_df, candidates=None):
    candidates = COMMON_EFFECTIVE_WIND_CANDIDATES if candidates is None else candidates
    frames = []
    for candidate in candidates:
        source, _, _ = _parse_candidate(candidate)
        source_df = ldaps_df if source == "ldaps" else gfs_df
        frames.append(_candidate_series(source_df, candidate))

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="forecast_kst_dtm", how="inner")

    out = out.sort_values("forecast_kst_dtm").ffill().fillna(0).reset_index(drop=True)
    eff_cols = [col for col in out.columns if col.startswith("eff_")]
    for col in eff_cols:
        out[f"{col}_sq"] = out[col] ** 2
        out[f"{col}_cube"] = out[col] ** 3

    pairs = [
        ("eff_gfs__grid1__ws850", "eff_ldaps__grid13__ws10"),
        ("eff_gfs__grid1__ws850", "eff_ldaps__grid12__ws10"),
        ("eff_gfs__grid1__ws850", "eff_gfs__grid5__ws850"),
        ("eff_ldaps__grid12__ws10", "eff_ldaps__grid13__ws10"),
        ("eff_ldaps__grid13__ws50max", "eff_ldaps__grid13__ws10"),
    ]
    for left, right in pairs:
        if left not in out.columns or right not in out.columns:
            continue
        safe_right = out[right].replace(0, np.nan)
        out[f"{left}_over_{right}"] = (out[left] / safe_right).replace([np.inf, -np.inf], np.nan).fillna(0)
        out[f"{left}_minus_{right}"] = out[left] - out[right]
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def add_effective_wind_features(weather, ldaps_df, gfs_df, candidates=None):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    eff = build_effective_wind_features(ldaps_df, gfs_df, candidates=candidates)
    return out.merge(eff, on="forecast_kst_dtm", how="left").sort_values("forecast_kst_dtm").ffill().fillna(0)
