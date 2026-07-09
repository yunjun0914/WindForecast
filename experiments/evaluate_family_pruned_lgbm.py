import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from analyze_feature_families import classify_feature
from predict_power_lgbm_best import clean_params, sample_weight
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from tune_power_lgbm_hyperparams import filter_scada_years, filter_weather_years, parse_list, score_one
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]

VARIANTS = {
    "full": [],
    "drop_tiny_nonwind": ["power_curve", "precipitation", "meteo_other", "physics_other", "air_density"],
    "drop_tiny_plus_cloud": ["power_curve", "precipitation", "meteo_other", "physics_other", "air_density", "cloud"],
    "drop_selected_low": [
        "power_curve",
        "precipitation",
        "meteo_other",
        "wind_regime_poly",
    ],
    "drop_tiny_plus_regime": [
        "power_curve",
        "precipitation",
        "meteo_other",
        "physics_other",
        "air_density",
        "wind_regime_poly",
    ],
    "drop_tiny_cloud_regime": [
        "power_curve",
        "precipitation",
        "meteo_other",
        "physics_other",
        "air_density",
        "cloud",
        "wind_regime_poly",
    ],
    "core_interpretable": [
        "power_curve",
        "precipitation",
        "meteo_other",
        "physics_other",
        "air_density",
        "cloud",
        "wind_regime_poly",
    ],
}


def load_data():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    return ldaps, gfs, labels, scada_by_group


def feature_columns(weather):
    return [col for col in weather.columns if col not in TIME_KEY_COLS]


def add_calendar_interactions(weather):
    out = weather.copy()
    dt = pd.to_datetime(out["forecast_kst_dtm"])
    doy_angle = 2.0 * np.pi * dt.dt.dayofyear.to_numpy(float) / 365.25
    hod_angle = 2.0 * np.pi * dt.dt.hour.to_numpy(float) / 24.0

    out["cal_doy2_sin"] = np.sin(2.0 * doy_angle)
    out["cal_doy2_cos"] = np.cos(2.0 * doy_angle)
    out["cal_doy3_sin"] = np.sin(3.0 * doy_angle)
    out["cal_doy3_cos"] = np.cos(3.0 * doy_angle)
    out["cal_hod2_sin"] = np.sin(2.0 * hod_angle)
    out["cal_hod2_cos"] = np.cos(2.0 * hod_angle)

    sin_doy = out["sin_doy"] if "sin_doy" in out else np.sin(doy_angle)
    cos_doy = out["cos_doy"] if "cos_doy" in out else np.cos(doy_angle)
    sin_hod = out["sin_hod"] if "sin_hod" in out else np.sin(hod_angle)
    cos_hod = out["cos_hod"] if "cos_hod" in out else np.cos(hod_angle)
    out["cal_doy_hod_ss"] = sin_doy * sin_hod
    out["cal_doy_hod_sc"] = sin_doy * cos_hod
    out["cal_doy_hod_cs"] = cos_doy * sin_hod
    out["cal_doy_hod_cc"] = cos_doy * cos_hod
    return out


def build_family_map(tagged_csv=None):
    family_map = {}
    if tagged_csv and Path(tagged_csv).exists():
        tagged = pd.read_csv(tagged_csv, encoding="utf-8-sig")
        for row in tagged[["feature", "family"]].drop_duplicates().itertuples(index=False):
            family_map[row.feature] = row.family
    return family_map


def select_features(all_features, drop_families, family_map):
    selected = []
    rows = []
    drop_families = set(drop_families)
    for feature in all_features:
        family = family_map.get(feature, classify_feature(feature))
        keep = family not in drop_families
        rows.append({"feature": feature, "family": family, "keep": keep})
        if keep:
            selected.append(feature)
    return selected, pd.DataFrame(rows)


def score_predictions(scores):
    fold_means = (
        scores.groupby(["variant", "pred_year"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    )
    summary = (
        fold_means.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
        )
        .sort_values("mean_score", ascending=False)
    )
    group_summary = (
        scores.groupby(["variant", "group"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            n_features=("n_features", "max"),
            n_folds=("score", "count"),
        )
    )
    return summary, group_summary, fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--variants", default=",".join(VARIANTS.keys()))
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--family-csv", default="results/feature_family_v2_tagged_features.csv")
    parser.add_argument("--stem", default="family_pruned_lgbm_v1")
    parser.add_argument("--calendar-interactions", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    variant_names = parse_list(args.variants)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    family_map = build_family_map(args.family_csv)
    ldaps, gfs, labels, scada_by_group = load_data()

    feature_cache = {}
    for group in groups:
        print(f"build all_meteo_compact_v2 {group}")
        weather = build_all_meteo_compact_v2(ldaps, gfs, group)
        if args.calendar_interactions:
            weather = add_calendar_interactions(weather)
        feature_cache[group] = weather

    score_rows = []
    pred_parts = []
    feature_rows = []
    for group in groups:
        row = best[best["group"] == group]
        if row.empty:
            continue
        best_row = row.iloc[0]
        base_weather = feature_cache[group]

        for pred_year in YEARS:
            train_years = [year for year in YEARS if year != pred_year]
            scada = filter_scada_years(scada_by_group[group], train_years)
            if len(scada) == 0:
                continue
            train_weather_base = filter_weather_years(base_weather, train_years)
            val_weather_base = filter_weather_years(base_weather, [pred_year])
            train_weather, val_weather = add_power_curve_feature_oof(
                train_weather_base,
                val_weather_base,
                scada,
                group,
                HUB_HEIGHT_PROXY_COL,
                GROUP_N_TURBINES[group],
            )
            x_train_full, y_train = build_group_dataset(train_weather, labels, group)
            x_val_full, y_val = build_group_dataset(val_weather, labels, group)
            if len(x_train_full) < 1000 or len(x_val_full) < 200:
                print(f"{group} pred_year={pred_year}: skip insufficient rows train={len(x_train_full)} val={len(x_val_full)}")
                continue
            val_times = (
                val_weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
                .dropna(subset=[group])["forecast_kst_dtm"]
                .reset_index(drop=True)
            )
            min_output_ratio = float(best_row["min_output_ratio"])
            train_mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * min_output_ratio
            y_used = y_train.loc[train_mask].reset_index(drop=True)
            all_features = feature_columns(train_weather)

            for variant in variant_names:
                if variant not in VARIANTS:
                    raise ValueError(f"unknown variant: {variant}. choices={list(VARIANTS)}")
                selected, feature_df = select_features(all_features, VARIANTS[variant], family_map)
                if not selected:
                    continue
                x_train = x_train_full.loc[train_mask].reset_index(drop=True).reindex(columns=selected, fill_value=0)
                x_val = x_val_full.reindex(columns=selected, fill_value=0)

                model = LGBMRegressor(**clean_params(best_row))
                weights = sample_weight(y_used, group, best_row["weight_policy"])
                model.fit(x_train, y_used, sample_weight=weights)
                pred = model.predict(x_val)
                score, nmae, ficr = score_one(y_val, pred, group)
                score_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n_features": len(selected),
                        "n_used": len(y_used),
                        "n_val": len(y_val),
                    }
                )
                pred_parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": pd.to_datetime(val_times).to_numpy(),
                            "pred_year": pred_year,
                            "train_years": ",".join(map(str, train_years)),
                            "variant": variant,
                            "group": group,
                            "actual": y_val.to_numpy(float),
                            "pred": pred,
                        }
                    )
                )
                kept_count = int(feature_df["keep"].sum())
                for family, part in feature_df.groupby("family"):
                    feature_rows.append(
                        {
                            "variant": variant,
                            "group": group,
                            "pred_year": pred_year,
                            "family": family,
                            "keep": bool(part["keep"].iloc[0]),
                            "n_features": len(part),
                            "n_kept_total": kept_count,
                        }
                    )
                print(f"{group} {variant} pred_year={pred_year}: score={score:.5f} ficr={ficr:.5f} features={len(selected)}")

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    feature_summary = pd.DataFrame(feature_rows).drop_duplicates()
    summary, group_summary, fold_means = score_predictions(scores)

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_summary.csv", index=False, encoding="utf-8-sig")
    fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(RESULTS_DIR / f"{args.stem}_feature_families.csv", index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== group summary ===")
    print(
        group_summary.sort_values(["group", "mean_score"], ascending=[True, False])
        .groupby("group", as_index=False)
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
