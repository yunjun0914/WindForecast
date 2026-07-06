import pandas as pd

from evaluate_group3_effective_teacher_mix import blend_weather
from evaluate_pinn_active_filtered_teacher import apply_teacher as apply_active_teacher
from evaluate_pinn_active_filtered_teacher import fit_teacher as fit_active_teacher
from evaluate_pinn_meteo_teacher_model_blend import (
    GROUP1,
    GROUP2,
    GROUP3,
    build_extended_pinn_weather,
    build_meteo_features,
    add_meteo_block,
    teacher_weather,
)
from train_pinn import train_manufacturer


RESULTS_PATH = "results/pinn_active_filtered_hybrid_scores.csv"
VAL_START = "2024-01-01 01:00:00"


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    weather = build_extended_pinn_weather(ldaps, gfs)
    weather = add_meteo_block(weather, build_meteo_features(ldaps, gfs), "all_meteo")

    g1 = teacher_weather(weather, scada_vestas, GROUP1, "cubic", "rf_hist_gbr_avg")
    g2 = teacher_weather(weather, scada_vestas, GROUP2, "p90", "rf_hist_gbr_avg")

    g3_unison_teacher = fit_active_teacher(weather, scada_unison, GROUP3, fit_before=VAL_START)
    g3_unison = apply_active_teacher(weather, g3_unison_teacher, "filtered_mix_cubic_p90")
    g3_vestas_teacher = fit_active_teacher(weather, scada_vestas, GROUP2, fit_before=VAL_START)
    g3_vestas = apply_active_teacher(weather, g3_vestas_teacher, "filtered_mix_cubic_p90")
    g3 = blend_weather("group3_active_filtered_hybrid_mix", g3_unison, g3_vestas, 0.30)

    weather_by_manufacturer = {
        "vestas": {GROUP1: g1, GROUP2: g2},
        "unison": {GROUP3: g3},
    }

    rows = []
    for manufacturer, weather_by_group in weather_by_manufacturer.items():
        _, _, stage1, stage2 = train_manufacturer(manufacturer, weather_by_group, labels, verbose=False, save=False)
        stage1["stage"] = "physics_only"
        stage2["stage"] = "with_bias"
        for frame in [stage1, stage2]:
            frame["variant"] = "rf_hist_gbr_g12_active_filtered_g3"
            frame["manufacturer"] = manufacturer
            rows.append(frame)

    results = pd.concat(rows, ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(results[results["stage"] == "with_bias"].to_string(index=False))
    print("\n=== mean ===")
    print(results[results["stage"] == "with_bias"].agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean")))
    return results


if __name__ == "__main__":
    main()
