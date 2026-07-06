import pandas as pd
from lightgbm import LGBMRegressor

from evaluate_tree_feature_blocks import (
    GROUP_CAPACITY_KWH,
    GROUP_N_TURBINES,
    HUB_HEIGHT_PROXY_COL,
    fit_group_power_curve_before,
    score_one,
    split_fold,
)
from evaluate_pinn_effective_wind_teacher import add_lead_feature
from utils.power_curve import add_power_curve_feature
from utils.preprocessing import FARM_CENTROID, TIME_KEY_COLS, build_weather_features, haversine_km

RESULTS_PATH = "results/tree_meteo_feature_block_fast_scores.csv"

FOLDS = [
    {"fold": "year_2024", "fold_type": "yearly", "val_start": "2024-01-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
    {"fold": "q2024_1", "fold_type": "quarter", "val_start": "2024-01-01 01:00:00", "val_end": "2024-04-01 01:00:00"},
    {"fold": "q2024_2", "fold_type": "quarter", "val_start": "2024-04-01 01:00:00", "val_end": "2024-07-01 01:00:00"},
    {"fold": "q2024_3", "fold_type": "quarter", "val_start": "2024-07-01 01:00:00", "val_end": "2024-10-01 01:00:00"},
    {"fold": "q2024_4", "fold_type": "quarter", "val_start": "2024-10-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
BLOCKS = ["baseline", "thermo", "radiation_cloud", "all_meteo", "lead_all_meteo"]

LDAPS_WIND_RAW = {
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
    "heightAboveGround_5_XBLWS",
    "heightAboveGround_5_YBLWS",
}
GFS_WIND_RAW = {
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "planetaryBoundaryLayer_0_u",
    "planetaryBoundaryLayer_0_v",
    "isobaricInhPa_850_u",
    "isobaricInhPa_850_v",
    "isobaricInhPa_700_u",
    "isobaricInhPa_700_v",
    "isobaricInhPa_500_u",
    "isobaricInhPa_500_v",
    "surface_0_gust",
}


def nearest_gfs_grid_id(df):
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = haversine_km(FARM_CENTROID[0], FARM_CENTROID[1], grids["latitude"], grids["longitude"])
    return grids.loc[dist.idxmin(), "grid_id"]


def build_meteo_features(ldaps, gfs):
    ldaps_cols = [c for c in ldaps.columns if c not in TIME_KEY_COLS + ["grid_id", "latitude", "longitude"] and c not in LDAPS_WIND_RAW]
    ldaps_agg = ldaps.groupby(TIME_KEY_COLS, as_index=False)[ldaps_cols].mean()
    ldaps_agg = ldaps_agg.rename(columns={c: f"met_ldaps_{c}" for c in ldaps_cols})

    grid_id = nearest_gfs_grid_id(gfs)
    gfs_sel = gfs[gfs["grid_id"] == grid_id].copy()
    gfs_cols = [c for c in gfs_sel.columns if c not in TIME_KEY_COLS + ["grid_id", "latitude", "longitude"] and c not in GFS_WIND_RAW]
    gfs_agg = gfs_sel[TIME_KEY_COLS + gfs_cols].rename(columns={c: f"met_gfs_{c}" for c in gfs_cols})

    out = ldaps_agg.merge(gfs_agg, on="forecast_kst_dtm", how="inner", suffixes=("", "_gfs_dup"))
    out = out.drop(columns=[c for c in out.columns if c.endswith("_gfs_dup")])
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out = add_lead_feature(out, ldaps)
    return out.sort_values("forecast_kst_dtm").ffill().fillna(0).reset_index(drop=True)


def add_meteo_block(weather, meteo, block):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    met = meteo.copy()
    met["forecast_kst_dtm"] = pd.to_datetime(met["forecast_kst_dtm"])
    if block == "baseline":
        return out

    cols = ["forecast_kst_dtm"]
    if "lead" in block:
        cols += ["lead_hour"]
    if block == "thermo":
        keys = ["_2_t", "_2_dpt", "_2_r", "_2_q", "_2_2t", "_2_2d", "_2_2r", "_2_2sh", "_sp", "_prmsl", "_850_t", "_850_r", "_700_t", "_500_t", "_blh"]
        cols += [c for c in met.columns if any(k in c for k in keys)]
    elif block == "radiation_cloud":
        keys = ["SW", "LW", "dswrf", "dlwrf", "cloud", "Cloud", "hcc", "mcc", "lcc", "tcc", "CDC", "prate", "tp", "lsrate", "snol", "SNOM"]
        cols += [c for c in met.columns if any(k in c for k in keys)]
    elif "all_meteo" in block:
        cols += [c for c in met.columns if c not in TIME_KEY_COLS]

    cols = [c for c in dict.fromkeys(cols) if c in met.columns]
    return out.merge(met[cols], on="forecast_kst_dtm", how="left").sort_values("forecast_kst_dtm").ffill().fillna(0)


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    weather_base = build_weather_features(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)

    rows = []
    for fold in FOLDS:
        print(f"\n=== {fold['fold']} ===")
        for block in BLOCKS:
            group_scores = []
            for group, capacity in GROUP_CAPACITY_KWH.items():
                curve = fit_group_power_curve_before(scada_by_group[group], group, fold["val_start"])
                if curve is None:
                    continue
                weather = add_meteo_block(weather_base, meteo, block)
                weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve, GROUP_N_TURBINES[group])
                x_train, x_val, y_train, y_val = split_fold(weather, labels, group, fold["val_start"], fold["val_end"])
                if len(x_train) < 1000 or len(x_val) < 200:
                    continue
                model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
                model.fit(x_train, y_train)
                pred = model.predict(x_val)
                score, nmae, ficr = score_one(y_val, pred, capacity)
                rows.append({**fold, "feature_block": block, "group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(y_val)})
                group_scores.append(score)
            if group_scores:
                mean_score = sum(group_scores) / len(group_scores)
                rows.append({**fold, "feature_block": block, "group": "mean", "score": mean_score, "nmae": None, "ficr": None, "n": None})
                print(f"{block}: {mean_score:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary = (
        results[results["group"] == "mean"]
        .groupby(["fold_type", "feature_block"], as_index=False)
        .agg(mean_score=("score", "mean"), worst_fold=("score", "min"), std_score=("score", "std"), n_folds=("score", "count"))
        .sort_values(["fold_type", "mean_score"], ascending=[True, False])
    )
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
