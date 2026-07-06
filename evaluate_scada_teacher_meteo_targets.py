import pandas as pd

from evaluate_pinn_effective_wind_teacher import (
    EXT_TARGETS,
    build_extended_pinn_weather,
    build_extended_scada_targets,
    extended_feature_cols,
)
from evaluate_scada_teacher_effective_targets import GROUP_SCADA, split_fit_eval
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features

RESULTS_PATH = "results/scada_teacher_meteo_target_scores.csv"


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    scada_by_name = {
        "vestas": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "unison": pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"),
    }

    weather_ext = build_extended_pinn_weather(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    weather_meteo = add_meteo_block(weather_ext, meteo, "all_meteo")

    feature_sets = {
        "extended": (weather_ext, extended_feature_cols(weather_ext)),
        "extended_meteo": (weather_meteo, extended_feature_cols(weather_meteo)),
    }
    target_cols = ["scada_ws_mean", "scada_ws_p75", "scada_ws_p90", "scada_ws_cubic"]
    folds = [
        ("year_2023", "2023-01-01 01:00:00", "2024-01-01 01:00:00"),
        ("year_2024", "2024-01-01 01:00:00", "2025-01-01 01:00:00"),
    ]

    rows = []
    for group, scada_name in GROUP_SCADA.items():
        targets = build_extended_scada_targets(scada_by_name[scada_name], group)
        for fold, val_start, val_end in folds:
            for feature_set, (weather, feature_cols) in feature_sets.items():
                for row in split_fit_eval(weather, targets, feature_cols, target_cols, val_start, val_end, "rf"):
                    rows.append({"group": group, "fold": fold, "feature_set": feature_set, "model": "rf", **row})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(results.sort_values(["fold", "group", "target", "feature_set"]).to_string(index=False))
    print("\n=== Mean R2 ===")
    summary = (
        results.groupby(["fold", "feature_set", "target"], as_index=False)
        .agg(r2=("r2", "mean"), mae=("mae", "mean"))
        .sort_values(["fold", "target", "r2"], ascending=[True, True, False])
    )
    print(summary.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
