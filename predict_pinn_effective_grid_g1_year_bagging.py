import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from utils.effective_wind_features import add_effective_wind_features
from utils.meteo_features import add_meteo_block, build_meteo_features
from utils.metrics import GROUP_CAPACITY_KWH
import utils.pinn_effective_pipeline as pfte
from utils.pinn_effective_pipeline import (
    apply_extended_teacher_crossfit,
    blend_weather,
    build_extended_pinn_weather,
    filter_forecast_years,
    filter_label_years,
    filter_scada_years,
)
from utils.pinn_scada_teacher_config import apply_best_scada_teacher_pinn_hparams


DEFAULT_OUTPUT = "results/submission_pinn_effective_grid_g1_year_bagging.csv"
DEFAULT_FOLD_OUTPUT = "results/pinn_effective_grid_g1_year_bagging_fold_stats.csv"


def build_weather_base(ldaps, gfs, use_effective_grid):
    weather = build_extended_pinn_weather(ldaps, gfs)
    weather = add_meteo_block(weather, build_meteo_features(ldaps, gfs), "all_meteo")
    if use_effective_grid:
        weather = add_effective_wind_features(weather, ldaps, gfs)
    return weather


def teacher_train_test(weather_train, weather_test, scada_df, group, v_mode):
    return apply_extended_teacher_crossfit(weather_train, weather_test, scada_df, group, v_mode)


def build_weather_for_years(train_years):
    ldaps_train_all = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train_all = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")

    ldaps_train = filter_forecast_years(ldaps_train_all, train_years)
    gfs_train = filter_forecast_years(gfs_train_all, train_years)

    canonical_train = build_weather_base(ldaps_train, gfs_train, use_effective_grid=False)
    canonical_test = build_weather_base(ldaps_test, gfs_test, use_effective_grid=False)
    effective_train = build_weather_base(ldaps_train, gfs_train, use_effective_grid=True)
    effective_test = build_weather_base(ldaps_test, gfs_test, use_effective_grid=True)

    scada_vestas = filter_scada_years(pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"), train_years)
    scada_unison = filter_scada_years(pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"), train_years)

    g1_train, g1_test = teacher_train_test(
        effective_train, effective_test, scada_vestas, pfte.GROUP1, "cubic"
    )
    g2_train, g2_test = teacher_train_test(
        canonical_train, canonical_test, scada_vestas, pfte.GROUP2, "p90"
    )

    g3_unison_train, g3_unison_test = teacher_train_test(
        canonical_train, canonical_test, scada_unison, pfte.GROUP3, "p90"
    )
    g3_vestas_train, g3_vestas_test = teacher_train_test(
        canonical_train, canonical_test, scada_vestas, pfte.GROUP2, "p90"
    )
    g3_train = blend_weather("effective_g1_group3_canonical_mix", g3_unison_train, g3_vestas_train, 0.30)
    g3_test = blend_weather("effective_g1_group3_canonical_mix", g3_unison_test, g3_vestas_test, 0.30)

    return {
        "train": {pfte.GROUP1: g1_train, pfte.GROUP2: g2_train, pfte.GROUP3: g3_train},
        "test": {pfte.GROUP1: g1_test, pfte.GROUP2: g2_test, pfte.GROUP3: g3_test},
    }


def predict_fold(train_years):
    print(f"\n=== effective-g1 year-bag fold train_years={train_years} ===")
    labels = filter_label_years(pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig"), train_years)
    weather = build_weather_for_years(train_years)

    vestas_model, vestas_bias = pfte.train_full_pinn(
        "vestas",
        {pfte.GROUP1: weather["train"][pfte.GROUP1], pfte.GROUP2: weather["train"][pfte.GROUP2]},
        labels,
    )
    unison_model, unison_bias = pfte.train_full_pinn("unison", {pfte.GROUP3: weather["train"][pfte.GROUP3]}, labels)

    pred = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(weather["test"][pfte.GROUP1]["forecast_kst_dtm"])})
    pred[pfte.GROUP1] = np.clip(
        pfte.predict_pinn(vestas_model, vestas_bias[pfte.GROUP1], pfte.GROUP1, weather["test"][pfte.GROUP1]),
        0,
        GROUP_CAPACITY_KWH[pfte.GROUP1],
    )
    pred[pfte.GROUP2] = np.clip(
        pfte.predict_pinn(vestas_model, vestas_bias[pfte.GROUP2], pfte.GROUP2, weather["test"][pfte.GROUP2]),
        0,
        GROUP_CAPACITY_KWH[pfte.GROUP2],
    )
    pred[pfte.GROUP3] = np.clip(
        pfte.predict_pinn(unison_model, unison_bias[pfte.GROUP3], pfte.GROUP3, weather["test"][pfte.GROUP3]),
        0,
        GROUP_CAPACITY_KWH[pfte.GROUP3],
    )
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-stats-output", default=DEFAULT_FOLD_OUTPUT)
    parser.add_argument("--years", default="2022,2023,2024")
    parser.add_argument("--teacher-backend", default=pfte.TEACHER_BACKEND, choices=["rf_oob", "lgbm_time_oof"])
    args = parser.parse_args()

    apply_best_scada_teacher_pinn_hparams(pfte)
    pfte.TEACHER_BACKEND = args.teacher_backend
    years = [int(year.strip()) for year in args.years.split(",") if year.strip()]
    folds = list(combinations(years, len(years) - 1))
    if len(folds) != 3:
        raise ValueError(f"expected 3 leave-one-year-out folds, got years={years} folds={folds}")

    fold_preds = []
    fold_stats = []
    for train_years in folds:
        pred = predict_fold(list(train_years))
        fold_preds.append(pred)
        stats = pred[pfte.GROUPS].agg(["min", "max", "mean"]).reset_index().rename(columns={"index": "stat"})
        stats.insert(0, "train_years", ",".join(str(y) for y in train_years))
        fold_stats.append(stats)

    base_time = fold_preds[0]["forecast_kst_dtm"].reset_index(drop=True)
    for idx, pred in enumerate(fold_preds[1:], start=1):
        if not base_time.equals(pred["forecast_kst_dtm"].reset_index(drop=True)):
            raise ValueError(f"test time mismatch for fold {idx}")

    prediction = pd.DataFrame({"forecast_kst_dtm": base_time})
    for group in pfte.GROUPS:
        stacked = np.vstack([pred[group].to_numpy() for pred in fold_preds])
        prediction[group] = np.clip(stacked.mean(axis=0), 0, GROUP_CAPACITY_KWH[group])

    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"])
    merged = submission[["forecast_id", "forecast_kst_dtm"]].merge(prediction, on="forecast_kst_dtm", how="left")
    if merged[pfte.GROUPS].isna().any().any():
        missing = merged[merged[pfte.GROUPS].isna().any(axis=1)].head()
        raise ValueError(f"submission has missing predictions:\n{missing}")
    merged = merged[["forecast_id", "forecast_kst_dtm", *pfte.GROUPS]]
    if len(merged) != len(submission):
        raise ValueError(f"row count mismatch: {len(merged)} vs {len(submission)}")

    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    pd.concat(fold_stats, ignore_index=True).to_csv(args.fold_stats_output, index=False, encoding="utf-8-sig")
    print(f"\nsaved {args.output}: {merged.shape}")
    print(merged[pfte.GROUPS].agg(["min", "max", "mean"]).to_string())
    print(f"saved {args.fold_stats_output}")
    return merged


if __name__ == "__main__":
    main()
