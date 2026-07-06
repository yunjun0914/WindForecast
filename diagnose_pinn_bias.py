import numpy as np
import pandas as pd
import torch

from models.pinn import PowerCurvePINN
from train_pinn import DEVICE, group_prediction, load_training_data, time_split
from utils.pinn_data import (
    C_MAX_BY_MANUFACTURER,
    GROUP_MANUFACTURER,
    GROUP_N_TURBINES,
    build_group_pinn_dataset,
)
from utils.metrics import GROUP_CAPACITY_KWH
from utils.pinn_physics import MANUFACTURER_AREA


def rms(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x**2))


def main():
    corrected_weather, labels = load_training_data()

    for manufacturer, weather in corrected_weather.items():
        model = PowerCurvePINN(
            C_MAX_BY_MANUFACTURER[manufacturer],
            MANUFACTURER_AREA[manufacturer],
        ).to(DEVICE)
        model.load_state_dict(
            torch.load(f"results/pinn_{manufacturer}_stage2.pt", map_location=DEVICE)
        )
        model.eval()

        for group, group_manufacturer in GROUP_MANUFACTURER.items():
            if group_manufacturer != manufacturer:
                continue

            ds = build_group_pinn_dataset(weather, labels, group)
            train_df, val_df = time_split(ds)

            with torch.no_grad():
                train_pred = group_prediction(
                    model, train_df, GROUP_N_TURBINES[group], bias=None
                ).cpu().numpy()
                val_pred = group_prediction(
                    model, val_df, GROUP_N_TURBINES[group], bias=None
                ).cpu().numpy()

            train_resid = train_df["y"].to_numpy() - train_pred
            val_resid = val_df["y"].to_numpy() - val_pred
            train_hod = (
                pd.DataFrame({"hod": train_df["hod"], "resid": train_resid})
                .groupby("hod")["resid"]
                .mean()
                .reindex(range(24))
                .to_numpy()
            )
            val_hod = (
                pd.DataFrame({"hod": val_df["hod"], "resid": val_resid})
                .groupby("hod")["resid"]
                .mean()
                .reindex(range(24))
                .to_numpy()
            )
            bias_state = torch.load(f"results/pinn_{group}_bias.pt", map_location="cpu")
            bias_ratio = bias_state["hod_bias.weight"].squeeze().numpy()
            capacity = float(bias_state.get("capacity", GROUP_CAPACITY_KWH[group]))
            bias = bias_ratio * capacity
            hour_bias = None
            if "hour_bias.weight" in bias_state:
                hour_bias = bias_state["hour_bias.weight"].squeeze().numpy() * capacity

            train_hod_bias = bias[train_df["hod"].to_numpy()]
            val_hod_bias = bias[val_df["hod"].to_numpy()]
            train_after_hod = train_resid - train_hod_bias
            val_after_hod = val_resid - val_hod_bias
            if hour_bias is not None:
                train_after_all = train_after_hod - hour_bias
            else:
                train_after_all = train_after_hod

            print(f"\n{group}")
            print(
                "train hod residual min/max/rms:",
                round(float(np.nanmin(train_hod)), 1),
                round(float(np.nanmax(train_hod)), 1),
                round(float(rms(train_hod)), 1),
            )
            print(
                "val hod residual min/max/rms:",
                round(float(np.nanmin(val_hod)), 1),
                round(float(np.nanmax(val_hod)), 1),
                round(float(rms(val_hod)), 1),
            )
            print(
                "learned bias min/max/rms:",
                round(float(np.nanmin(bias)), 1),
                round(float(np.nanmax(bias)), 1),
                round(float(rms(bias)), 1),
            )
            if hour_bias is not None:
                print(
                    "train-only hour bias min/max/rms:",
                    round(float(np.nanmin(hour_bias)), 1),
                    round(float(np.nanmax(hour_bias)), 1),
                    round(float(rms(hour_bias)), 1),
                )
            print("train residual rms after hod:", round(float(rms(train_after_hod)), 1))
            print("train residual rms after all train bias:", round(float(rms(train_after_all)), 1))
            print("val residual rms after hod:", round(float(rms(val_after_hod)), 1))
            print("corr(train_hod, bias):", round(float(np.corrcoef(train_hod, bias)[0, 1]), 3))
            print("corr(val_hod, bias):", round(float(np.corrcoef(val_hod, bias)[0, 1]), 3))
            print("train_hod:", np.round(train_hod, 1).tolist())
            print("val_hod:  ", np.round(val_hod, 1).tolist())
            print("bias:     ", np.round(bias, 1).tolist())


if __name__ == "__main__":
    main()
