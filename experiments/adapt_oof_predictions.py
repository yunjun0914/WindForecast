import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


YEARS = [2022, 2023, 2024]
REQUIRED_STANDARD_COLUMNS = [
    "forecast_kst_dtm",
    "pred_year",
    "train_years",
    "model_family",
    "model_name",
    "group",
    "actual",
    "pred",
    "is_clipped",
]


def train_years_for_pred_year(pred_year, years=YEARS):
    pred_year = int(pred_year)
    return ",".join(str(year) for year in years if year != pred_year)


def load_labels():
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    return labels[["forecast_kst_dtm", *TARGET_COLS]]


def infer_is_clipped(pred, group):
    values = np.asarray(pred, dtype=float)
    capacity = GROUP_CAPACITY_KWH[group]
    return bool(np.nanmin(values) >= -1e-8 and np.nanmax(values) <= capacity + 1e-8)


def standardize_common(df, model_family, model_name, default_train_years=True):
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["pred_year"] = out["pred_year"].astype(int)
    if "train_years" not in out.columns and default_train_years:
        out["train_years"] = out["pred_year"].map(train_years_for_pred_year)
    out["model_family"] = model_family
    out["model_name"] = model_name
    return out


def adapt_pinn_wide(input_path, model_family, model_name):
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    labels = load_labels()
    merged = df.merge(labels, on="forecast_kst_dtm", how="left", suffixes=("_pred", "_actual"))
    rows = []
    for group in TARGET_COLS:
        pred_col = f"{group}_pred"
        actual_col = f"{group}_actual"
        if pred_col not in merged.columns or actual_col not in merged.columns:
            raise ValueError(f"missing columns for {group}: {pred_col}, {actual_col}")
        part = pd.DataFrame(
            {
                "forecast_kst_dtm": merged["forecast_kst_dtm"],
                "pred_year": merged["pred_year"].astype(int),
                "train_years": merged["train_years"].astype(str),
                "model_family": model_family,
                "model_name": model_name,
                "group": group,
                "actual": merged[actual_col].astype(float),
                "pred": merged[pred_col].astype(float),
                "is_clipped": infer_is_clipped(merged[pred_col], group),
            }
        )
        part = part.dropna(subset=["actual"]).reset_index(drop=True)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out


def adapt_tree_long(input_path, model_family, model_name, train_policy=None, source_model=None):
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if train_policy is not None and "train_policy" in df.columns:
        df = df[df["train_policy"] == train_policy].copy()
    if source_model is not None and "model" in df.columns:
        df = df[df["model"] == source_model].copy()
    required = {"forecast_kst_dtm", "pred_year", "train_years", "group", "actual", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required tree columns: {sorted(missing)}")
    out = standardize_common(df, model_family, model_name)
    out["actual"] = out["actual"].astype(float)
    out["pred"] = out["pred"].astype(float)
    out["is_clipped"] = out.groupby("group")["pred"].transform(
        lambda s: infer_is_clipped(s, str(out.loc[s.index[0], "group"]))
    )
    return out[REQUIRED_STANDARD_COLUMNS]


def adapt_blend_aligned(input_path, model_family, model_name, tree_weight):
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required = {"forecast_kst_dtm", "pred_year", "group", "actual", "tree_pred", "pinn_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required blend columns: {sorted(missing)}")
    out = standardize_common(df, model_family, model_name)
    out["actual"] = out["actual"].astype(float)
    out["pred"] = (1.0 - tree_weight) * out["pinn_pred"].astype(float) + tree_weight * out["tree_pred"].astype(float)
    clipped = []
    for group, part in out.groupby("group"):
        capacity = GROUP_CAPACITY_KWH[group]
        pred = np.clip(part["pred"].to_numpy(dtype=float), 0, capacity)
        out.loc[part.index, "pred"] = pred
        clipped.append((group, True))
    out["is_clipped"] = True
    return out[REQUIRED_STANDARD_COLUMNS]


def validate_standard_oof(df):
    missing = set(REQUIRED_STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing standard columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("standard OOF output is empty")
    out = df[REQUIRED_STANDARD_COLUMNS].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["pred_year"] = out["pred_year"].astype(int)
    out["train_years"] = out["train_years"].astype(str)
    out["model_family"] = out["model_family"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["group"] = out["group"].astype(str)
    out["actual"] = out["actual"].astype(float)
    out["pred"] = out["pred"].astype(float)
    out["is_clipped"] = out["is_clipped"].astype(bool)
    if out["actual"].isna().any():
        missing_actual = int(out["actual"].isna().sum())
        raise ValueError(f"missing actual values after adaptation: {missing_actual}")
    if not np.isfinite(out["pred"].to_numpy(dtype=float)).all():
        raise ValueError("non-finite predictions after adaptation")
    bad_groups = sorted(set(out["group"]) - set(TARGET_COLS))
    if bad_groups:
        raise ValueError(f"unknown target groups: {bad_groups}")
    return out.sort_values(["model_name", "pred_year", "group", "forecast_kst_dtm"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Convert legacy OOF prediction files to the standard long schema.")
    parser.add_argument("--kind", required=True, choices=["pinn_wide", "tree_long", "blend_aligned"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-policy", default=None, help="Optional filter for legacy tree predictions.")
    parser.add_argument("--source-model", default=None, help="Optional filter for legacy tree model column.")
    parser.add_argument("--tree-weight", type=float, default=0.5, help="Tree weight for blend_aligned files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.kind == "pinn_wide":
        out = adapt_pinn_wide(input_path, args.model_family, args.model_name)
    elif args.kind == "tree_long":
        out = adapt_tree_long(input_path, args.model_family, args.model_name, args.train_policy, args.source_model)
    elif args.kind == "blend_aligned":
        out = adapt_blend_aligned(input_path, args.model_family, args.model_name, args.tree_weight)
    else:
        raise ValueError(f"unknown kind: {args.kind}")

    out = validate_standard_oof(out)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"saved {output_path}: {out.shape}")
    print(out.groupby(["model_family", "model_name", "group"]).size().rename("rows").reset_index().to_string(index=False))
    return out


if __name__ == "__main__":
    main()
