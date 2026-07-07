from itertools import combinations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.effective_wind_features import add_effective_wind_features
from utils.meteo_features import add_meteo_block, build_meteo_features
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
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


RESULTS_PATH = "results/pinn_effective_grid_g1_year_bagging_oof_scores.csv"
OOF_PATH = "results/pinn_effective_grid_g1_year_bagging_oof_predictions.csv"
GROUP1 = "kpx_group_1"
GROUP2 = "kpx_group_2"
GROUP3 = "kpx_group_3"
GROUPS = [GROUP1, GROUP2, GROUP3]


def build_weather_base(ldaps, gfs, use_effective_grid):
    weather = build_extended_pinn_weather(ldaps, gfs)
    weather = add_meteo_block(weather, build_meteo_features(ldaps, gfs), "all_meteo")
    if use_effective_grid:
        weather = add_effective_wind_features(weather, ldaps, gfs)
    return weather


def teacher_train_pred(weather_train, weather_pred, scada_df, group, v_mode):
    return apply_extended_teacher_crossfit(weather_train, weather_pred, scada_df, group, v_mode)


def build_weather_for_fold(train_years, pred_year):
    ldaps_all = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_all = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")

    ldaps_train = filter_forecast_years(ldaps_all, train_years)
    gfs_train = filter_forecast_years(gfs_all, train_years)
    ldaps_pred = filter_forecast_years(ldaps_all, [pred_year])
    gfs_pred = filter_forecast_years(gfs_all, [pred_year])

    canonical_train = build_weather_base(ldaps_train, gfs_train, use_effective_grid=False)
    canonical_pred = build_weather_base(ldaps_pred, gfs_pred, use_effective_grid=False)
    effective_train = build_weather_base(ldaps_train, gfs_train, use_effective_grid=True)
    effective_pred = build_weather_base(ldaps_pred, gfs_pred, use_effective_grid=True)

    scada_vestas = filter_scada_years(pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"), train_years)
    scada_unison = filter_scada_years(pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"), train_years)

    g1_train, g1_pred = teacher_train_pred(
        effective_train, effective_pred, scada_vestas, GROUP1, "cubic"
    )
    g2_train, g2_pred = teacher_train_pred(
        canonical_train, canonical_pred, scada_vestas, GROUP2, "p90"
    )

    g3_unison_train, g3_unison_pred = teacher_train_pred(
        canonical_train, canonical_pred, scada_unison, GROUP3, "p90"
    )
    g3_vestas_train, g3_vestas_pred = teacher_train_pred(
        canonical_train, canonical_pred, scada_vestas, GROUP2, "p90"
    )
    g3_train = blend_weather("effective_g1_group3_canonical_mix", g3_unison_train, g3_vestas_train, 0.30)
    g3_pred = blend_weather("effective_g1_group3_canonical_mix", g3_unison_pred, g3_vestas_pred, 0.30)

    return {
        "train": {GROUP1: g1_train, GROUP2: g2_train, GROUP3: g3_train},
        "pred": {GROUP1: g1_pred, GROUP2: g2_pred, GROUP3: g3_pred},
    }


def predict_oof_fold(train_years, pred_year):
    print(f"\n=== effective-g1 OOF fold train_years={train_years} pred_year={pred_year} ===")
    labels_train = filter_label_years(pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig"), train_years)
    weather = build_weather_for_fold(train_years, pred_year)

    vestas_model, vestas_bias = pfte.train_full_pinn(
        "vestas",
        {GROUP1: weather["train"][GROUP1], GROUP2: weather["train"][GROUP2]},
        labels_train,
    )
    unison_model, unison_bias = pfte.train_full_pinn("unison", {GROUP3: weather["train"][GROUP3]}, labels_train)

    pred = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(weather["pred"][GROUP1]["forecast_kst_dtm"])})
    pred[GROUP1] = np.clip(
        pfte.predict_pinn(vestas_model, vestas_bias[GROUP1], GROUP1, weather["pred"][GROUP1]),
        0,
        GROUP_CAPACITY_KWH[GROUP1],
    )
    pred[GROUP2] = np.clip(
        pfte.predict_pinn(vestas_model, vestas_bias[GROUP2], GROUP2, weather["pred"][GROUP2]),
        0,
        GROUP_CAPACITY_KWH[GROUP2],
    )
    pred[GROUP3] = np.clip(
        pfte.predict_pinn(unison_model, unison_bias[GROUP3], GROUP3, weather["pred"][GROUP3]),
        0,
        GROUP_CAPACITY_KWH[GROUP3],
    )
    pred["pred_year"] = pred_year
    pred["train_years"] = ",".join(str(year) for year in train_years)
    return pred


def score_fold(pred, labels_all):
    truth = labels_all.copy()
    truth["kst_dtm"] = pd.to_datetime(truth["kst_dtm"])
    merged = pred.merge(truth, left_on="forecast_kst_dtm", right_on="kst_dtm", suffixes=("_pred", "_actual"))
    rows = []
    for group in GROUPS:
        actual = merged[f"{group}_actual"]
        forecast = merged[f"{group}_pred"]
        valid = actual.notna()
        if int(valid.sum()) == 0:
            nmae, ficr, score = np.nan, np.nan, np.nan
        else:
            nmae, ficr = group_nmae_ficr(actual[valid], forecast[valid], GROUP_CAPACITY_KWH[group])
            score = 0.5 * (1 - nmae) + 0.5 * ficr
        rows.append(
            {
                "pred_year": int(pred["pred_year"].iloc[0]),
                "train_years": pred["train_years"].iloc[0],
                "stage": f"group__{group}",
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": int(valid.sum()),
            }
        )
    group_df = pd.DataFrame(rows)
    rows.append(
        {
            "pred_year": int(pred["pred_year"].iloc[0]),
            "train_years": pred["train_years"].iloc[0],
            "stage": "fold_mean",
            "score": group_df["score"].mean(skipna=True),
            "nmae": group_df["nmae"].mean(skipna=True),
            "ficr": group_df["ficr"].mean(skipna=True),
            "n_rows": int(group_df["n_rows"].replace(0, np.nan).min()),
        }
    )
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-backend", default=pfte.TEACHER_BACKEND, choices=["rf_oob", "lgbm_time_oof"])
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    apply_best_scada_teacher_pinn_hparams(pfte)
    pfte.TEACHER_BACKEND = args.teacher_backend
    stem = args.stem or (
        "pinn_effective_grid_g1_year_bagging"
        if args.teacher_backend == "rf_oob"
        else f"pinn_effective_grid_g1_year_bagging_{args.teacher_backend}"
    )
    results_path = f"results/{stem}_oof_scores.csv"
    oof_path = f"results/{stem}_oof_predictions.csv"
    years = [2022, 2023, 2024]
    labels_all = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    oof_preds = []
    score_rows = []
    for train_years in combinations(years, len(years) - 1):
        pred_year = sorted(set(years) - set(train_years))[0]
        pred = predict_oof_fold(list(train_years), pred_year)
        oof_preds.append(pred)
        score_rows.append(score_fold(pred, labels_all))

    scores = pd.concat(score_rows, ignore_index=True)
    fold_means = scores[scores["stage"] == "fold_mean"]
    overall = fold_means.agg({"score": "mean", "nmae": "mean", "ficr": "mean"}).to_dict()
    scores = pd.concat(
        [
            scores,
            pd.DataFrame(
                [
                    {
                        "pred_year": "all",
                        "train_years": "leave_one_year_out",
                        "stage": "overall_mean",
                        "score": overall["score"],
                        "nmae": overall["nmae"],
                        "ficr": overall["ficr"],
                        "n_rows": int(fold_means["n_rows"].sum()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    oof = pd.concat(oof_preds, ignore_index=True)
    scores.to_csv(results_path, index=False, encoding="utf-8-sig")
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    print("\n=== effective-g1 OOF scores ===")
    print(scores.to_string(index=False))
    print(f"saved {results_path}")
    print(f"saved {oof_path}")
    return scores


if __name__ == "__main__":
    main()
