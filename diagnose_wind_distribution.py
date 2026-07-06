import pandas as pd
import torch

from models.pinn import PowerCurvePINN
from utils.pinn_data import (
    C_MAX_BY_MANUFACTURER,
    apply_wind_speed_correction,
    build_pinn_weather,
    fit_wind_speed_correction,
)
from utils.pinn_physics import MANUFACTURER_AREA


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_pinn_weather(ldaps, gfs)
    print(
        "raw v_std p50/p90/p99:",
        [round(float(weather["v_std"].quantile(q)), 3) for q in [0.5, 0.9, 0.99]],
    )

    for manufacturer, scada_path in [
        ("vestas", "data/train/scada_vestas_train.csv"),
        ("unison", "data/train/scada_unison_train.csv"),
    ]:
        scada = pd.read_csv(scada_path, encoding="utf-8-sig")
        correction = fit_wind_speed_correction(weather, scada, manufacturer)
        corrected = apply_wind_speed_correction(weather, correction)
        model = PowerCurvePINN(
            C_MAX_BY_MANUFACTURER[manufacturer],
            MANUFACTURER_AREA[manufacturer],
        )
        model.load_state_dict(torch.load(f"results/pinn_{manufacturer}_stage2.pt", map_location="cpu"))
        print(
            manufacturer,
            "v_std p50/p90/p99",
            [round(float(corrected["v_std"].quantile(q)), 3) for q in [0.5, 0.9, 0.99]],
            "sigma floor/scale",
            round(float(model.wind_dist.floor), 3),
            round(float(model.wind_dist.scale), 3),
            "tau",
            round(float(model.response.tau), 3),
        )
        correction_cols, _, _, coef = correction
        print(
            manufacturer,
            "wind correction standardized coef",
            {col: round(float(coef[i + 1]), 3) for i, col in enumerate(correction_cols)},
            "intercept",
            round(float(coef[0]), 3),
        )


if __name__ == "__main__":
    main()
