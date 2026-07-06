import pandas as pd
import torch

from models.pinn import PowerCurvePINN, TurbineGroupBias
from train_pinn import DEVICE, group_prediction, load_training_data, time_split
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.pinn_data import (
    C_MAX_BY_MANUFACTURER,
    GROUP_MANUFACTURER,
    GROUP_N_TURBINES,
    build_group_pinn_dataset,
)
from utils.pinn_physics import MANUFACTURER_AREA


def main():
    corrected_weather, labels = load_training_data()
    rows = []

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
            train_years = train_df["forecast_kst_dtm"].dt.year.unique()
            bias = TurbineGroupBias(
                GROUP_CAPACITY_KWH[group],
                n_train_rows=len(train_df),
                n_train_years=len(train_years),
            ).to(DEVICE)
            bias.load_state_dict(torch.load(f"results/pinn_{group}_bias.pt", map_location=DEVICE))
            bias.eval()

            with torch.no_grad():
                raw = group_prediction(model, val_df, GROUP_N_TURBINES[group], bias=bias)
                clipped = torch.clamp(raw, min=0.0, max=GROUP_CAPACITY_KWH[group])

            y = val_df["y"].to_numpy()
            for name, pred in [("raw", raw), ("clamped", clipped)]:
                nmae, ficr = group_nmae_ficr(y, pred.cpu().numpy(), GROUP_CAPACITY_KWH[group])
                score = 0.5 * (1 - nmae) + 0.5 * ficr
                rows.append(
                    {
                        "group": group,
                        "prediction": name,
                        "nmae": nmae,
                        "ficr": ficr,
                        "score": score,
                    }
                )

    results = pd.DataFrame(rows)
    print(results)
    print("\nMean score:")
    print(results.groupby("prediction")["score"].mean().sort_values(ascending=False))
    return results


if __name__ == "__main__":
    main()
