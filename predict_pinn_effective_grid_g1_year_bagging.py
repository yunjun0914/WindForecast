import argparse
from itertools import combinations

import numpy as np
import pandas as pd

import train_pinn as tp
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


def teacher_train_valid_test(weather_train, weather_valid, weather_test, scada_df, group, v_mode):
    if weather_valid is None:
        train_teacher, test_teacher = teacher_train_test(weather_train, weather_test, scada_df, group, v_mode)
        return train_teacher, None, test_teacher

    valid_len = len(weather_valid)
    pred_weather = pd.concat([weather_valid, weather_test], ignore_index=True)
    train_teacher, pred_teacher = teacher_train_test(weather_train, pred_weather, scada_df, group, v_mode)
    valid_teacher = pred_teacher.iloc[:valid_len].reset_index(drop=True)
    test_teacher = pred_teacher.iloc[valid_len:].reset_index(drop=True)
    return train_teacher, valid_teacher, test_teacher


def build_weather_for_years(train_years, valid_year=None):
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
    if valid_year is not None:
        ldaps_valid = filter_forecast_years(ldaps_train_all, [valid_year])
        gfs_valid = filter_forecast_years(gfs_train_all, [valid_year])
        canonical_valid = build_weather_base(ldaps_valid, gfs_valid, use_effective_grid=False)
        effective_valid = build_weather_base(ldaps_valid, gfs_valid, use_effective_grid=True)

    scada_vestas = filter_scada_years(pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"), train_years)
    scada_unison = filter_scada_years(pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"), train_years)

    g1_train, g1_valid, g1_test = teacher_train_valid_test(
        effective_train,
        effective_valid if valid_year is not None else None,
        effective_test,
        scada_vestas,
        pfte.GROUP1,
        "cubic",
    )
    g2_train, g2_valid, g2_test = teacher_train_valid_test(
        canonical_train,
        canonical_valid if valid_year is not None else None,
        canonical_test,
        scada_vestas,
        pfte.GROUP2,
        "p90",
    )

    g3_unison_train, g3_unison_valid, g3_unison_test = teacher_train_valid_test(
        canonical_train,
        canonical_valid if valid_year is not None else None,
        canonical_test,
        scada_unison,
        pfte.GROUP3,
        "p90",
    )
    g3_vestas_train, g3_vestas_valid, g3_vestas_test = teacher_train_valid_test(
        canonical_train,
        canonical_valid if valid_year is not None else None,
        canonical_test,
        scada_vestas,
        pfte.GROUP2,
        "p90",
    )
    g3_train = blend_weather("effective_g1_group3_canonical_mix", g3_unison_train, g3_vestas_train, 0.30)
    g3_test = blend_weather("effective_g1_group3_canonical_mix", g3_unison_test, g3_vestas_test, 0.30)
    if valid_year is not None:
        g3_valid = blend_weather("effective_g1_group3_canonical_mix", g3_unison_valid, g3_vestas_valid, 0.30)

    out = {
        "train": {pfte.GROUP1: g1_train, pfte.GROUP2: g2_train, pfte.GROUP3: g3_train},
        "test": {pfte.GROUP1: g1_test, pfte.GROUP2: g2_test, pfte.GROUP3: g3_test},
    }
    if valid_year is not None:
        out["valid"] = {pfte.GROUP1: g1_valid, pfte.GROUP2: g2_valid, pfte.GROUP3: g3_valid}
    return out


def predict_fold(train_years, all_years, early_stop_kwargs=None):
    early_stop_kwargs = {} if early_stop_kwargs is None else early_stop_kwargs
    valid_years = sorted(set(all_years) - set(train_years))
    if len(valid_years) != 1:
        raise ValueError(f"expected one held-out valid year, got train_years={train_years} valid_years={valid_years}")
    valid_year = valid_years[0]
    print(f"\n=== effective-g1 year-bag fold train_years={train_years} valid_year={valid_year} ===")
    labels_all = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels = filter_label_years(labels_all, train_years)
    valid_labels = filter_label_years(labels_all, [valid_year])
    weather = build_weather_for_years(train_years, valid_year=valid_year)

    vestas_model, vestas_bias = pfte.train_full_pinn(
        "vestas",
        {pfte.GROUP1: weather["train"][pfte.GROUP1], pfte.GROUP2: weather["train"][pfte.GROUP2]},
        labels,
        valid_weather_by_group={pfte.GROUP1: weather["valid"][pfte.GROUP1], pfte.GROUP2: weather["valid"][pfte.GROUP2]},
        valid_labels=valid_labels,
        **early_stop_kwargs,
    )
    unison_model, unison_bias = pfte.train_full_pinn(
        "unison",
        {pfte.GROUP3: weather["train"][pfte.GROUP3]},
        labels,
        valid_weather_by_group={pfte.GROUP3: weather["valid"][pfte.GROUP3]},
        valid_labels=valid_labels,
        **early_stop_kwargs,
    )

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
    parser.add_argument("--use-dow-bias", action="store_true")
    parser.add_argument("--dow-l2", type=float, default=None)
    parser.add_argument("--use-moy-bias", action="store_true")
    parser.add_argument("--moy-l2", type=float, default=None)
    parser.add_argument("--use-train-hour-bias", action="store_true")
    parser.add_argument("--hour-l2", type=float, default=None)
    parser.add_argument("--hour-l1", type=float, default=None)
    parser.add_argument("--hour-prox-start-epoch", type=int, default=None)
    parser.add_argument("--use-train-year-bias", action="store_true")
    parser.add_argument("--year-l2", type=float, default=None)
    parser.add_argument("--use-scada-wd-correction", action="store_true")
    parser.add_argument("--wd-amplitude", type=float, default=None)
    parser.add_argument("--wd-l2", type=float, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--no-stage1-early-stopping", action="store_true")
    parser.add_argument("--no-stage2-early-stopping", action="store_true")
    parser.add_argument("--stage1-patience", type=int, default=pfte.STAGE1_PATIENCE)
    parser.add_argument("--stage2-patience", type=int, default=pfte.STAGE2_PATIENCE)
    parser.add_argument("--stage1-eval-interval", type=int, default=pfte.STAGE1_EVAL_INTERVAL)
    parser.add_argument("--stage2-eval-interval", type=int, default=pfte.STAGE2_EVAL_INTERVAL)
    args = parser.parse_args()

    apply_best_scada_teacher_pinn_hparams(pfte)
    if args.stage1_epochs is not None:
        pfte.STAGE1_EPOCHS = args.stage1_epochs
    if args.stage2_epochs is not None:
        pfte.STAGE2_EPOCHS = args.stage2_epochs
    pfte.TEACHER_BACKEND = args.teacher_backend
    pfte.USE_DOW_BIAS = args.use_dow_bias
    pfte.USE_MOY_BIAS = args.use_moy_bias
    pfte.USE_TRAIN_ONLY_HOUR_BIAS = args.use_train_hour_bias
    pfte.USE_TRAIN_ONLY_YEAR_BIAS = args.use_train_year_bias
    tp.USE_DOW_BIAS = args.use_dow_bias
    tp.USE_MOY_BIAS = args.use_moy_bias
    tp.USE_TRAIN_ONLY_HOUR_BIAS = args.use_train_hour_bias
    tp.USE_TRAIN_ONLY_YEAR_BIAS = args.use_train_year_bias
    pfte.USE_SCADA_WD_CORRECTION = args.use_scada_wd_correction
    tp.USE_SCADA_WD_CORRECTION = args.use_scada_wd_correction
    if args.wd_amplitude is not None:
        pfte.SCADA_WD_AMPLITUDE = args.wd_amplitude
        tp.SCADA_WD_AMPLITUDE = args.wd_amplitude
    if args.wd_l2 is not None:
        pfte.LAMBDA["wd"] = args.wd_l2
    if args.use_dow_bias:
        pfte.LAMBDA["dow"] = args.dow_l2 if args.dow_l2 is not None else pfte.LAMBDA["hod"]
    if args.use_moy_bias and args.moy_l2 is not None:
        pfte.LAMBDA["moy"] = args.moy_l2
    if args.use_train_hour_bias:
        if args.hour_l2 is not None:
            pfte.LAMBDA["hour"] = args.hour_l2
        if args.hour_l1 is not None:
            pfte.LAMBDA["hour_l1"] = args.hour_l1
        if args.hour_prox_start_epoch is not None:
            pfte.LAMBDA["hour_prox_start_epoch"] = args.hour_prox_start_epoch
    if args.use_train_year_bias and args.year_l2 is not None:
        pfte.LAMBDA["year"] = args.year_l2
    years = [int(year.strip()) for year in args.years.split(",") if year.strip()]
    folds = list(combinations(years, len(years) - 1))
    if len(folds) != 3:
        raise ValueError(f"expected 3 leave-one-year-out folds, got years={years} folds={folds}")
    early_stop_kwargs = {
        "early_stopping": not args.no_early_stopping,
        "stage1_early_stopping": not args.no_stage1_early_stopping,
        "stage2_early_stopping": not args.no_stage2_early_stopping,
        "stage1_patience": args.stage1_patience,
        "stage2_patience": args.stage2_patience,
        "stage1_eval_interval": args.stage1_eval_interval,
        "stage2_eval_interval": args.stage2_eval_interval,
    }

    fold_preds = []
    fold_stats = []
    for train_years in folds:
        pred = predict_fold(list(train_years), years, early_stop_kwargs=early_stop_kwargs)
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
