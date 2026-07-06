import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from evaluate_tree_feature_block_submission_style import score_one, split_time
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from predict_tree_feature_block import CV_SPLITTER, MODELS, ensemble_fit_predict
from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, build_weather_features

VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/tree_meteo_submission_style_scores.csv"


def evaluate_block(block, weather_base, meteo, labels, scada_by_group):
    weather = add_meteo_block(weather_base, meteo, block)
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val = split_time(group_weather, labels, group, VAL_START)

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)

        raw_val = ensemble_fit_predict(x_train, y_train, x_val)
        per_group[group] = {"y": y_val.to_numpy(), "raw": raw_val, "capacity": capacity}
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))

    rows = []
    for group, data in per_group.items():
        capacity = data["capacity"]
        for variant, pred in {
            "raw": data["raw"],
            "pooled_isotonic": calibrator.predict(data["raw"] / capacity) * capacity,
        }.items():
            score, nmae, ficr = score_one(data["y"], pred, capacity)
            rows.append({"feature_block": block, "variant": variant, "group": group, "score": score, "nmae": nmae, "ficr": ficr})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="baseline,all_meteo")
    args = parser.parse_args()

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
    for block in [b.strip() for b in args.blocks.split(",") if b.strip()]:
        print(f"\n=== {block} ===")
        block_rows = evaluate_block(block, weather_base, meteo, labels, scada_by_group)
        rows.extend(block_rows)
        tmp = pd.DataFrame(block_rows)
        means = tmp.groupby("variant")["score"].mean().sort_values(ascending=False)
        print(means.round(4).to_string())

    results = pd.DataFrame(rows)
    means = results.groupby(["feature_block", "variant"], as_index=False).agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    results = pd.concat([results, means.assign(group="mean")[results.columns]], ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(means.sort_values("score", ascending=False).to_string(index=False))
    return results


if __name__ == "__main__":
    main()
