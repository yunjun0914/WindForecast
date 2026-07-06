import numpy as np
import pandas as pd

from evaluate_group3_pinn_teacher_transfer import train_single_group
from evaluate_pinn_effective_wind_teacher import (
    build_extended_pinn_weather,
    fit_extended_teacher,
    apply_extended_teacher,
)

RESULTS_PATH = "results/group3_effective_teacher_mix_scores.csv"


def blend_weather(name, weather_a, weather_b, weight_a):
    out = weather_a.copy()
    blend_cols = ["v", "v_std", "scada_ws_mean", "scada_ws_std", "scada_ws_p10", "scada_ws_p50", "scada_ws_p75", "scada_ws_p90", "scada_ws_cubic"]
    for col in blend_cols:
        if col in weather_a.columns and col in weather_b.columns:
            out[col] = weight_a * weather_a[col].to_numpy() + (1 - weight_a) * weather_b[col].to_numpy()
    out["teacher_recipe"] = name
    return out


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    weather_ext = build_extended_pinn_weather(ldaps, gfs)
    unison_cubic = apply_extended_teacher(
        weather_ext,
        fit_extended_teacher(weather_ext, scada_unison, "kpx_group_3"),
        "cubic",
    )
    vestas_g2_cubic = apply_extended_teacher(
        weather_ext,
        fit_extended_teacher(weather_ext, scada_vestas, "kpx_group_2"),
        "cubic",
    )
    unison_p90 = apply_extended_teacher(
        weather_ext,
        fit_extended_teacher(weather_ext, scada_unison, "kpx_group_3"),
        "p90",
    )
    vestas_g2_p90 = apply_extended_teacher(
        weather_ext,
        fit_extended_teacher(weather_ext, scada_vestas, "kpx_group_2"),
        "p90",
    )

    variants = {
        "unison_cubic": unison_cubic,
        "mix30_unison70_vestas_cubic": blend_weather("mix30_unison70_vestas_cubic", unison_cubic, vestas_g2_cubic, 0.30),
        "mix50_unison50_vestas_cubic": blend_weather("mix50_unison50_vestas_cubic", unison_cubic, vestas_g2_cubic, 0.50),
        "mix30_unison70_vestas_p90": blend_weather("mix30_unison70_vestas_p90", unison_p90, vestas_g2_p90, 0.30),
    }

    rows = []
    for name, weather in variants.items():
        print(f"\n=== {name} ===")
        result = train_single_group("unison", weather, labels, seed=42, verbose=False)
        rows.append(
            {
                "variant": name,
                "stage2_score": result["stage2_score"],
                "stage2_nmae": result["stage2_nmae"],
                "stage2_ficr": result["stage2_ficr"],
            }
        )
        print(rows[-1])

    results = pd.DataFrame(rows).sort_values("stage2_score", ascending=False)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== Summary ===")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
