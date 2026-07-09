import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from evaluate_hub_height_features import add_hub_height_features
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from utils.metrics import TARGET_COLS
from utils.power_curve import GROUP_TURBINE_PREFIXES


RESULTS_DIR = Path("results")


CANDIDATE_COLS = [
    "gfs_ws100_speed",
    "gfs_ws80_speed",
    "gfs_ws850_speed",
    "ldaps_ws10_speed",
    "ldaps_ws50_max_speed",
    "ldaps_ws50_min_speed",
    "phys_hub_gfs117_blend_speed",
    "phys_hub_ldaps117_from_10_alpha014_speed",
    "phys_hub_ldaps117_blend_speed",
    "phys_hub_mix117_blend_speed",
]


def scada_group_wind(scada, group):
    out = scada.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["kst_dtm"]).dt.floor("h")
    ws_cols = [f"{prefix}_ws" for prefix in GROUP_TURBINE_PREFIXES[group]]
    hourly = out.groupby("forecast_kst_dtm")[ws_cols].mean()
    values = hourly.to_numpy(float)
    return pd.DataFrame(
        {
            "forecast_kst_dtm": hourly.index,
            "scada_ws_mean": np.nanmean(values, axis=1),
            "scada_ws_std": np.nanstd(values, axis=1),
            "scada_ws_p50": np.nanpercentile(values, 50, axis=1),
        }
    ).dropna()


def score_alignment(df, group, col):
    part = df[["forecast_kst_dtm", "year", "scada_ws_mean", col]].dropna().copy()
    rows = []
    for label, sub in [("all", part), *[(str(year), p) for year, p in part.groupby("year")]]:
        if len(sub) < 100:
            continue
        pred = sub[col].to_numpy(float)
        actual = sub["scada_ws_mean"].to_numpy(float)
        err = pred - actual
        corr = float(np.corrcoef(pred, actual)[0, 1]) if np.std(pred) > 1e-9 and np.std(actual) > 1e-9 else np.nan
        rows.append(
            {
                "group": group,
                "year": label,
                "feature": col,
                "corr": corr,
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "pred_mean": float(np.mean(pred)),
                "scada_mean": float(np.mean(actual)),
                "n": len(sub),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="hub_height_scada_alignment_v1")
    args = parser.parse_args()

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    rows = []
    for group in TARGET_COLS:
        weather = add_hub_height_features(build_all_meteo_compact_v2(ldaps, gfs, group), group)
        scada_ws = scada_group_wind(scada_by_group[group], group)
        merged = weather.merge(scada_ws, on="forecast_kst_dtm", how="inner")
        merged["year"] = pd.to_datetime(merged["forecast_kst_dtm"]).dt.year
        cols = [col for col in CANDIDATE_COLS if col in merged.columns]
        print(f"{group}: rows={len(merged)} candidate_cols={len(cols)}")
        for col in cols:
            rows.extend(score_alignment(merged, group, col))

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / f"{args.stem}.csv", index=False, encoding="utf-8-sig")
    print("\n=== all-year top by group ===")
    print(
        result[result["year"].eq("all")]
        .sort_values(["group", "mae"], ascending=[True, True])
        .groupby("group", as_index=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
