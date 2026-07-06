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
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from utils.power_curve import add_power_curve_feature
from utils.preprocessing import build_weather_features
from utils.spatial_features import add_group_spatial_features


RESULTS_PATH = "results/tree_spatial_feature_fast_scores.csv"
FOLDS = [
    {"fold": "year_2024", "fold_type": "yearly", "val_start": "2024-01-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
    {"fold": "q2024_1", "fold_type": "quarter", "val_start": "2024-01-01 01:00:00", "val_end": "2024-04-01 01:00:00"},
    {"fold": "q2024_2", "fold_type": "quarter", "val_start": "2024-04-01 01:00:00", "val_end": "2024-07-01 01:00:00"},
    {"fold": "q2024_3", "fold_type": "quarter", "val_start": "2024-07-01 01:00:00", "val_end": "2024-10-01 01:00:00"},
    {"fold": "q2024_4", "fold_type": "quarter", "val_start": "2024-10-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
MODES = ["none", "nearest", "weighted", "upwind", "ldaps", "gfs", "all"]


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

    weather_base = add_meteo_block(build_weather_features(ldaps, gfs), build_meteo_features(ldaps, gfs), "all_meteo")
    spatial_cache = {
        group: {mode: (weather_base if mode == "none" else add_group_spatial_features(weather_base, ldaps, gfs, group, mode))}
        for group in GROUP_CAPACITY_KWH
        for mode in []
    }
    spatial_cache = {group: {} for group in GROUP_CAPACITY_KWH}
    for group in GROUP_CAPACITY_KWH:
        spatial_cache[group]["none"] = weather_base
        for mode in MODES:
            if mode == "none":
                continue
            print(f"build spatial {group} {mode}")
            spatial_cache[group][mode] = add_group_spatial_features(weather_base, ldaps, gfs, group, mode)

    rows = []
    for fold in FOLDS:
        print(f"\n=== {fold['fold']} ===")
        for mode in MODES:
            group_scores = []
            for group, capacity in GROUP_CAPACITY_KWH.items():
                curve = fit_group_power_curve_before(scada_by_group[group], group, fold["val_start"])
                if curve is None:
                    continue
                weather = add_power_curve_feature(spatial_cache[group][mode], HUB_HEIGHT_PROXY_COL, curve, GROUP_N_TURBINES[group])
                x_train, x_val, y_train, y_val = split_fold(weather, labels, group, fold["val_start"], fold["val_end"])
                if len(x_train) < 1000 or len(x_val) < 200:
                    continue
                model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
                model.fit(x_train, y_train)
                pred = model.predict(x_val)
                score, nmae, ficr = score_one(y_val, pred, capacity)
                rows.append({**fold, "spatial_mode": mode, "group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(y_val)})
                group_scores.append(score)
            if group_scores:
                mean_score = sum(group_scores) / len(group_scores)
                rows.append({**fold, "spatial_mode": mode, "group": "mean", "score": mean_score, "nmae": None, "ficr": None, "n": None})
                print(f"{mode}: {mean_score:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary = (
        results[results["group"] == "mean"]
        .groupby(["fold_type", "spatial_mode"], as_index=False)
        .agg(mean_score=("score", "mean"), worst_fold=("score", "min"), std_score=("score", "std"), n_folds=("score", "count"))
        .sort_values(["fold_type", "mean_score"], ascending=[True, False])
    )
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
