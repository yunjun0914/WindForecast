from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, pooled_oof_summary


KEYS = ["forecast_kst_dtm", "group", "pred_year"]
OUTPUT_BINS = (
    ("low_10_30", 0.10, 0.30),
    ("middle_30_80", 0.30, 0.80),
    ("shoulder_80_90", 0.80, 0.90),
    ("peak_90_100", 0.90, np.inf),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-prediction", type=Path, required=True)
    parser.add_argument("--median-prediction", type=Path, required=True)
    parser.add_argument("--max-prediction", type=Path, required=True)
    parser.add_argument("--baseline-prediction", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--stem", default="temporal_representation_experts_v1")
    return parser.parse_args()


def load_prediction(path: Path, name: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
    if "variant" in table.columns:
        variants = table["variant"].dropna().unique()
        if "baseline_retrained" in variants:
            table = table.loc[table["variant"].eq("baseline_retrained")]
        elif len(variants) != 1:
            raise ValueError(f"{name} contains ambiguous variants: {variants}")
    required = [*KEYS, "official_target", "pred"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{name} missing prediction columns: {missing}")
    table = table[required].copy()
    if table.duplicated(KEYS).any():
        raise ValueError(f"{name} contains duplicate prediction keys")
    return table.rename(
        columns={"official_target": f"actual__{name}", "pred": name}
    )


def align_predictions(paths: dict[str, Path]) -> pd.DataFrame:
    aligned = None
    for name, path in paths.items():
        current = load_prediction(path, name)
        aligned = current if aligned is None else aligned.merge(
            current,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    if aligned is None:
        raise ValueError("No prediction paths supplied")
    actual_columns = [column for column in aligned if column.startswith("actual__")]
    reference = aligned[actual_columns[0]].to_numpy(float)
    for column in actual_columns[1:]:
        if not np.allclose(reference, aligned[column], equal_nan=True):
            raise ValueError(f"Target mismatch in {column}")
    aligned["official_target"] = reference
    return aligned.drop(columns=actual_columns)


def to_long(aligned: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    parts = []
    for variant in variants:
        part = aligned[KEYS + ["official_target", variant]].rename(
            columns={variant: "pred"}
        )
        part["variant"] = variant
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def restricted_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    capacity: float,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, capacity)
    error = np.abs(prediction - actual) / capacity
    unit_price = np.select(
        [error <= 0.06, error <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    nmae = float(error.mean())
    ficr = float(np.sum(actual * unit_price) / np.sum(actual * 4.0))
    return {
        "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
        "nmae": nmae,
        "ficr": ficr,
        "hit6": float(np.mean(error <= 0.06)),
        "hit8": float(np.mean(error <= 0.08)),
        "n_rows": int(len(actual)),
    }


def output_bin_metrics(
    aligned: pd.DataFrame,
    variants: list[str],
) -> pd.DataFrame:
    rows = []
    for variant in variants:
        for bin_name, lower, upper in OUTPUT_BINS:
            group_rows = []
            for group in TARGET_COLS:
                capacity = float(GROUP_CAPACITY_KWH[group])
                part = aligned.loc[aligned["group"].eq(group)]
                ratio = part["official_target"] / capacity
                keep = ratio.ge(lower) & ratio.lt(upper)
                selected = part.loc[keep]
                if selected.empty:
                    continue
                group_rows.append(
                    restricted_metrics(
                        selected["official_target"].to_numpy(float),
                        selected[variant].to_numpy(float),
                        capacity,
                    )
                )
            row: dict[str, str | int | float] = {
                "variant": variant,
                "output_bin": bin_name,
                "n_groups": len(group_rows),
            }
            for metric in ["score", "nmae", "ficr", "hit6", "hit8"]:
                row[metric] = float(np.mean([item[metric] for item in group_rows]))
            row["n_rows"] = int(sum(item["n_rows"] for item in group_rows))
            rows.append(row)
    return pd.DataFrame(rows)


def residual_correlations(
    aligned: pd.DataFrame,
    variants: list[str],
) -> pd.DataFrame:
    parts = []
    for group_name, part in [("all", aligned), *aligned.groupby("group", sort=True)]:
        capacity = part["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
        residuals = pd.DataFrame(
            {
                variant: (
                    part[variant].to_numpy(float)
                    - part["official_target"].to_numpy(float)
                )
                / capacity
                for variant in variants
            }
        )
        correlation = residuals.corr()
        for left_index, left in enumerate(variants):
            for right in variants[left_index + 1 :]:
                parts.append(
                    {
                        "group": group_name,
                        "left": left,
                        "right": right,
                        "residual_correlation": float(correlation.loc[left, right]),
                    }
                )
    return pd.DataFrame(parts)


def main() -> None:
    args = parse_args()
    paths = {
        "temporal_min": args.min_prediction,
        "temporal_median": args.median_prediction,
        "temporal_max": args.max_prediction,
    }
    if args.baseline_prediction is not None:
        paths = {"original_baseline": args.baseline_prediction, **paths}
    aligned = align_predictions(paths)
    aligned["equal_min_median_max"] = aligned[
        ["temporal_min", "temporal_median", "temporal_max"]
    ].mean(axis=1)
    variants = [*paths, "equal_min_median_max"]
    long = to_long(aligned, variants)
    summary, groups = pooled_oof_summary(long)
    bins = output_bin_metrics(aligned, variants)
    correlations = residual_correlations(aligned, variants)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_dir / args.stem
    aligned.to_csv(
        f"{prefix}_aligned.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        f"{prefix}_summary.csv", index=False, encoding="utf-8-sig"
    )
    groups.to_csv(
        f"{prefix}_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    bins.to_csv(
        f"{prefix}_output_bins.csv", index=False, encoding="utf-8-sig"
    )
    correlations.to_csv(
        f"{prefix}_residual_correlations.csv", index=False, encoding="utf-8-sig"
    )
    print("=== pooled OOF ===")
    print(summary.to_string(index=False))
    print("\n=== output bins ===")
    print(bins.to_string(index=False))
    print("\n=== residual correlations ===")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
