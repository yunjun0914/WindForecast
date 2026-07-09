import argparse
import re
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")


def classify_feature(feature: str) -> str:
    f = feature.lower()

    if f.startswith("power_curve"):
        return "power_curve"
    if any(token in f for token in ["doy", "hod", "hour_day", "hdw", "lead_hour"]):
        return "calendar_time"

    if "air_density_x" in f and "cube" in f:
        return "density_x_wind_cube"
    if "air_density" in f or "rho" in f:
        return "air_density"

    if any(token in f for token in ["shear", "veer"]):
        return "wind_shear_veer"
    if ("ws" in f or "wind" in f) and any(token in f for token in ["regime", "sigmoid", "_cube", "_sq"]):
        return "wind_regime_poly"
    if ("ws" in f or "wind" in f or "gust" in f) and any(token in f for token in ["diff1", "diff3", "trend", "ramp"]):
        return "wind_weather_trend"

    if any(token in f for token in ["near_", "southwest", "axis", "upwind", "downwind", "gradient"]):
        return "wind_site_spatial"
    if "grid_" in f and ("ws" in f or "wind" in f):
        return "wind_grid_stats"

    if "gust" in f or "pbl" in f or "xbl" in f or "blh" in f:
        return "gust_boundary_layer"
    if ("ws" in f or "wind" in f) and (f.startswith("ldaps_") or f.startswith("gfs_") or f.startswith("phys_") or f.startswith("met_")):
        return "wind_raw_or_speed"
    if any(token in f for token in ["_u", "_v", "isobaricinhpa", "heightaboveground"]):
        return "wind_raw_or_speed"

    if any(token in f for token in ["dswrf", "ndnsw", "ndnlw", "radiation", "dlwrf", "ulwrf"]):
        return "radiation"
    if any(token in f for token in ["cloud", "hcc", "mcc", "lcc", "tcc"]):
        return "cloud"
    if any(token in f for token in ["precip", "prate", "lsrate", "lsprate", "ncpcp", "rain"]):
        return "precipitation"

    if any(token in f for token in ["pressure", "prmsl", "surface_0_sp"]):
        return "pressure"
    if any(token in f for token in ["lapse", "temperature"]) or re.search(r"(^|_)t($|_)", f):
        return "temperature_lapse"
    if "humidity" in f or re.search(r"(^|_)r($|_)", f) or re.search(r"(^|_)q($|_)", f):
        return "humidity"

    if f.startswith("met_"):
        return "meteo_other"
    if f.startswith("phys_"):
        return "physics_other"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance-csv", default="results/feature_importance_v1_tree_lgbm_best.csv")
    parser.add_argument("--stem", default="feature_family_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = pd.read_csv(args.importance_csv, encoding="utf-8-sig")
    imp["family"] = imp["feature"].map(classify_feature)

    by_group = (
        imp.groupby(["group", "family"], as_index=False)
        .agg(
            n_features=("feature", "nunique"),
            gain=("gain", "sum"),
            split=("split", "sum"),
            gain_norm_sum=("gain_norm", "sum"),
            best_gain_rank=("gain_rank", "min"),
            median_gain_rank=("gain_rank", "median"),
        )
        .sort_values(["group", "gain_norm_sum"], ascending=[True, False])
    )
    overall = (
        by_group.groupby("family", as_index=False)
        .agg(
            groups_present=("group", "nunique"),
            n_features=("n_features", "mean"),
            mean_gain_norm=("gain_norm_sum", "mean"),
            min_gain_norm=("gain_norm_sum", "min"),
            max_gain_norm=("gain_norm_sum", "max"),
            best_rank_any_group=("best_gain_rank", "min"),
            worst_best_rank=("best_gain_rank", "max"),
            median_rank_mean=("median_gain_rank", "mean"),
        )
        .sort_values("mean_gain_norm", ascending=False)
    )

    tagged_path = RESULTS_DIR / f"{args.stem}_tagged_features.csv"
    group_path = RESULTS_DIR / f"{args.stem}_by_group.csv"
    overall_path = RESULTS_DIR / f"{args.stem}_overall.csv"
    imp.to_csv(tagged_path, index=False, encoding="utf-8-sig")
    by_group.to_csv(group_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

    print("=== family overall ===")
    print(overall.to_string(index=False))
    print("\n=== family by group top/bottom ===")
    for group, part in by_group.groupby("group"):
        print(f"\n[{group}] top")
        print(part.head(12).to_string(index=False))
        print(f"\n[{group}] bottom")
        print(part.tail(8).to_string(index=False))
    print(f"\nsaved {tagged_path}")
    print(f"saved {group_path}")
    print(f"saved {overall_path}")


if __name__ == "__main__":
    main()
