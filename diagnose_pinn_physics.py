import numpy as np
import pandas as pd
import torch

from models.pinn import PowerCurvePINN, TurbineGroupBias
from train_pinn import DEVICE, group_prediction, load_training_data, time_split
from utils.metrics import GROUP_CAPACITY_KWH
from utils.pinn_data import (
    C_MAX_BY_MANUFACTURER,
    GROUP_MANUFACTURER,
    GROUP_N_TURBINES,
    build_group_pinn_dataset,
)
from utils.pinn_physics import MANUFACTURER_AREA


def summarize(name, values):
    values = np.asarray(values, dtype=float)
    return {
        "name": name,
        "min": np.nanmin(values),
        "p01": np.nanpercentile(values, 1),
        "p50": np.nanpercentile(values, 50),
        "p99": np.nanpercentile(values, 99),
        "max": np.nanmax(values),
        "n_neg": int(np.sum(values < 0)),
        "n": int(values.size),
    }


def print_summary(row):
    print(
        f"{row['name']}: min={row['min']:.3f} p01={row['p01']:.3f} "
        f"p50={row['p50']:.3f} p99={row['p99']:.3f} max={row['max']:.3f} "
        f"neg={row['n_neg']}/{row['n']}"
    )


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

        groups = [g for g, m in GROUP_MANUFACTURER.items() if m == manufacturer]
        print(f"\n== {manufacturer} ==")

        for group in groups:
            ds = build_group_pinn_dataset(weather, labels, group)
            train_df, val_df = time_split(ds)
            bias = TurbineGroupBias(
                GROUP_CAPACITY_KWH[group],
                n_train_rows=len(train_df),
                n_train_years=len(train_df["forecast_kst_dtm"].dt.year.unique()),
            ).to(DEVICE)
            bias.load_state_dict(torch.load(f"results/pinn_{group}_bias.pt", map_location=DEVICE))

            for split_name, df in [("train", train_df), ("val", val_df)]:
                v = torch.tensor(df["v"].to_numpy(), dtype=torch.float32, device=DEVICE)
                rho = torch.tensor(df["rho"].to_numpy(), dtype=torch.float32, device=DEVICE)
                doy = torch.tensor(df["doy"].to_numpy(), dtype=torch.float32, device=DEVICE)
                moy = torch.tensor(df["moy"].to_numpy(), dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    c_eff = model.c_eff(v, doy, moy).cpu().numpy()
                    p_phys_kw = (model.p_phys(v, rho, doy, moy) / 1000.0).cpu().numpy()
                    pred_no_bias = group_prediction(
                        model, df, GROUP_N_TURBINES[group], bias=None
                    ).cpu().numpy()
                    pred_with_calendar = group_prediction(
                        model, df, GROUP_N_TURBINES[group], bias=bias
                    ).cpu().numpy()

                print(f"\n{group} {split_name}")
                print_summary(summarize("C_eff", c_eff))
                print_summary(summarize("P_phys_per_turbine_kW", p_phys_kw))
                print_summary(summarize("P_group_no_bias_kWh", pred_no_bias))
                print_summary(summarize("P_group_calendar_bias_kWh", pred_with_calendar))

                neg = np.where(pred_with_calendar < 0)[0]
                if len(neg):
                    sample = df.iloc[neg[:5]][["forecast_kst_dtm", "v", "rho", "doy", "moy", "hod"]].copy()
                    sample["pred_with_calendar"] = pred_with_calendar[neg[:5]]
                    print("negative prediction sample:")
                    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
