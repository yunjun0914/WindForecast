import pandas as pd
from lightgbm import LGBMRegressor

from evaluate_tree_feature_blocks import (
    GROUP_CAPACITY_KWH,
    GROUP_N_TURBINES,
    HUB_HEIGHT_PROXY_COL,
    add_feature_block,
    build_extended_pinn_weather,
    build_weather_features,
    fit_group_power_curve_before,
    score_one,
    split_fold,
)
from utils.power_curve import add_power_curve_feature

RESULTS_PATH = "results/tree_feature_block_fast_scores.csv"

FOLDS = [
    {"fold": "year_2024", "fold_type": "yearly", "val_start": "2024-01-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
    {"fold": "q2024_1", "fold_type": "quarter", "val_start": "2024-01-01 01:00:00", "val_end": "2024-04-01 01:00:00"},
    {"fold": "q2024_2", "fold_type": "quarter", "val_start": "2024-04-01 01:00:00", "val_end": "2024-07-01 01:00:00"},
    {"fold": "q2024_3", "fold_type": "quarter", "val_start": "2024-07-01 01:00:00", "val_end": "2024-10-01 01:00:00"},
    {"fold": "q2024_4", "fold_type": "quarter", "val_start": "2024-10-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
BLOCKS = ["baseline", "lead", "ldaps", "gfs", "lead_ldaps_gfs"]


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
    weather_ext = build_extended_pinn_weather(ldaps, gfs)

    rows = []
    for fold in FOLDS:
        print(f"\n=== {fold['fold']} ===")
        for block in BLOCKS:
            group_scores = []
            for group, capacity in GROUP_CAPACITY_KWH.items():
                curve = fit_group_power_curve_before(scada_by_group[group], group, fold["val_start"])
                if curve is None:
                    continue
                weather = add_power_curve_feature(weather_base, HUB_HEIGHT_PROXY_COL, curve, GROUP_N_TURBINES[group])
                weather = add_feature_block(weather, weather_ext, block)
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
                rows.append({**fold, "feature_block": block, "group": "mean", "score": sum(group_scores) / len(group_scores), "nmae": None, "ficr": None, "n": None})
                print(f"{block}: {sum(group_scores) / len(group_scores):.4f}")

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
