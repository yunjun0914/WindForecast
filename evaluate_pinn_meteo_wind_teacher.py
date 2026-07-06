import argparse

import pandas as pd

from evaluate_pinn_effective_wind_teacher import (
    apply_extended_teacher,
    fit_extended_teacher,
    run_variant,
)
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from evaluate_pinn_effective_wind_teacher import build_extended_pinn_weather
from utils.pinn_data import GROUP_MANUFACTURER

RESULTS_PATH = "results/pinn_meteo_wind_teacher_scores.csv"
BASE_GROUP_SCADA = {
    "kpx_group_1": "vestas",
    "kpx_group_2": "vestas",
    "kpx_group_3": "unison",
}


def build_variant_weather(variant, weather_meteo, scada_by_name):
    by_group = {}
    for group, scada_name in BASE_GROUP_SCADA.items():
        teacher = fit_extended_teacher(weather_meteo, scada_by_name[scada_name], group, fit_before="2024-01-01 01:00:00")
        by_group[group] = apply_extended_teacher(weather_meteo, teacher, variant)

    return {
        "vestas": {group: by_group[group] for group in by_group if GROUP_MANUFACTURER[group] == "vestas"},
        "unison": {group: by_group[group] for group in by_group if GROUP_MANUFACTURER[group] == "unison"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="cubic,p90,mix_cubic_p90")
    parser.add_argument("--stage1-epochs", type=int, default=500)
    parser.add_argument("--stage2-epochs", type=int, default=2000)
    args = parser.parse_args()

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_by_name = {
        "vestas": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "unison": pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"),
    }

    weather_ext = build_extended_pinn_weather(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    weather_meteo = add_meteo_block(weather_ext, meteo, "all_meteo")

    results = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(f"\n=== variant: meteo_{variant} ===")
        weather_by_manufacturer = build_variant_weather(variant, weather_meteo, scada_by_name)
        result = run_variant(
            f"meteo_{variant}",
            weather_by_manufacturer,
            labels,
            args.stage1_epochs,
            args.stage2_epochs,
        )
        results.append(result)
        print(result[result["stage"] == "with_bias"].to_string(index=False))

    final = pd.concat(results, ignore_index=True)
    final.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    with_bias = final[final["stage"] == "with_bias"]
    summary = with_bias.groupby("variant").agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    print("\n=== mean with_bias summary ===")
    print(summary.sort_values("score", ascending=False).to_string())
    return final


if __name__ == "__main__":
    main()
