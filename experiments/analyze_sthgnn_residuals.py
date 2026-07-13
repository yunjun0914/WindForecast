from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/spatiotemporal_hetero_gnn_bigru_v1_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/spatiotemporal_hetero_gnn_bigru_v1_residual_correlations.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.predictions, encoding="utf-8-sig")
    keys = ["forecast_kst_dtm", "group", "pred_year", "actual"]
    wide = table.pivot_table(index=keys, columns="variant", values="prediction").reset_index()
    records = []
    variants = [
        column
        for column in wide.columns
        if column not in keys
    ]
    for (group, year), fold in wide.groupby(["group", "pred_year"]):
        valid = fold["actual"].ge(0.10 * GROUP_CAPACITY_KWH[group])
        residual = {
            variant: fold.loc[valid, variant].to_numpy(float)
            - fold.loc[valid, "actual"].to_numpy(float)
            for variant in variants
        }
        for left_index, left in enumerate(variants):
            for right in variants[left_index + 1 :]:
                records.append(
                    {
                        "group": group,
                        "pred_year": int(year),
                        "left": left,
                        "right": right,
                        "rows": int(valid.sum()),
                        "residual_correlation": float(
                            np.corrcoef(residual[left], residual[right])[0, 1]
                        ),
                    }
                )
    output = pd.DataFrame(records)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(output.groupby(["left", "right"])["residual_correlation"].mean().to_string())


if __name__ == "__main__":
    main()
