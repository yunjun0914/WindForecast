import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")
DEFAULT_INPUT = "results/pinn_lgbmteacher_powerlgbm_v2_l1_blend_aligned_predictions.csv"
GROUPS = TARGET_COLS


def score_arrays(actual, pred, group):
    nmae, ficr = group_nmae_ficr(actual, pred, GROUP_CAPACITY_KWH[group])
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def score_frame(df, pred_col):
    rows = []
    for group, part in df.groupby("group"):
        score, nmae, ficr = score_arrays(part["actual"], part[pred_col], group)
        rows.append({"group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(part), "pred_col": pred_col})
    out = pd.DataFrame(rows)
    rows.append(
        {
            "group": "mean",
            "score": out["score"].mean(),
            "nmae": out["nmae"].mean(),
            "ficr": out["ficr"].mean(),
            "n": int(out["n"].sum()),
            "pred_col": pred_col,
        }
    )
    return pd.DataFrame(rows)


def score_frame_by_fold(df, pred_col):
    rows = []
    for pred_year, fold in df.groupby("pred_year"):
        for group, part in fold.groupby("group"):
            score, nmae, ficr = score_arrays(part["actual"], part[pred_col], group)
            rows.append(
                {
                    "pred_year": pred_year,
                    "group": group,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n": len(part),
                    "pred_col": pred_col,
                }
            )
        fold_rows = [row for row in rows if row["pred_year"] == pred_year and row["pred_col"] == pred_col]
        rows.append(
            {
                "pred_year": pred_year,
                "group": "fold_mean",
                "score": float(np.mean([row["score"] for row in fold_rows])),
                "nmae": float(np.mean([row["nmae"] for row in fold_rows])),
                "ficr": float(np.mean([row["ficr"] for row in fold_rows])),
                "n": int(sum(row["n"] for row in fold_rows)),
                "pred_col": pred_col,
            }
        )
    fold_mean = pd.DataFrame(rows)
    folds = fold_mean[fold_mean["group"].eq("fold_mean")]
    rows.append(
        {
            "pred_year": "all",
            "group": "overall_fold_mean",
            "score": folds["score"].mean(),
            "nmae": folds["nmae"].mean(),
            "ficr": folds["ficr"].mean(),
            "n": int(folds["n"].sum()),
            "pred_col": pred_col,
        }
    )
    return pd.DataFrame(rows)


def official_like_score(df, pred_col):
    rows = []
    for group, part in df.groupby("group"):
        score, nmae, ficr = score_arrays(part["actual"], part[pred_col], group)
        rows.append((score, nmae, ficr))
    if not rows:
        return np.nan, np.nan, np.nan
    arr = np.asarray(rows, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


def add_base_columns(df, tree_weight):
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["capacity"] = out["group"].map(GROUP_CAPACITY_KWH).astype(float)
    out["blend_pred"] = (1.0 - tree_weight) * out["pinn_pred"] + tree_weight * out["tree_pred"]
    out["blend_pred"] = out["blend_pred"].clip(lower=0.0, upper=out["capacity"])
    out["pred_ratio"] = out["blend_pred"] / out["capacity"]
    out["actual_ratio"] = out["actual"] / out["capacity"]
    out["residual"] = out["actual"] - out["blend_pred"]
    out["residual_ratio"] = out["residual"] / out["capacity"]
    out["error_rate"] = out["residual_ratio"].abs()
    out["tree_error_rate"] = (out["tree_pred"] - out["actual"]).abs() / out["capacity"]
    out["pinn_error_rate"] = (out["pinn_pred"] - out["actual"]).abs() / out["capacity"]
    out["tree_pinn_gap"] = (out["tree_pred"] - out["pinn_pred"]) / out["capacity"]
    out["abs_tree_pinn_gap"] = out["tree_pinn_gap"].abs()
    out["hit6"] = out["error_rate"] <= 0.06
    out["hit8"] = out["error_rate"] <= 0.08
    out["pinn_hit8"] = out["pinn_error_rate"] <= 0.08
    out["tree_hit8"] = out["tree_error_rate"] <= 0.08
    out["price"] = np.select([out["error_rate"] <= 0.06, out["error_rate"] <= 0.08], [4.0, 3.0], default=0.0)
    out["settlement_weight"] = out["actual"] * 4.0
    return out[out["actual"] >= out["capacity"] * 0.10].reset_index(drop=True)


def weighted_hit(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = weights.sum()
    if denom <= 0:
        return np.nan
    return float((values * weights).sum() / denom)


def summarize_bins(df, by_cols):
    rows = []
    for key, part in df.groupby(by_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(by_cols, key))
        row.update(
            {
                "n": len(part),
                "actual_mean": part["actual_ratio"].mean(),
                "pred_mean": part["pred_ratio"].mean(),
                "resid_median": part["residual_ratio"].median(),
                "resid_p25": part["residual_ratio"].quantile(0.25),
                "resid_p75": part["residual_ratio"].quantile(0.75),
                "error_mean": part["error_rate"].mean(),
                "hit6": part["hit6"].mean(),
                "hit8": part["hit8"].mean(),
                "weighted_hit6": weighted_hit(part["hit6"], part["actual"]),
                "weighted_hit8": weighted_hit(part["hit8"], part["actual"]),
                "pinn_better_rate": (part["pinn_error_rate"] < part["tree_error_rate"]).mean(),
                "tree_better_rate": (part["tree_error_rate"] < part["pinn_error_rate"]).mean(),
                "pinn_only_hit8": ((part["pinn_hit8"]) & (~part["tree_hit8"])).mean(),
                "tree_only_hit8": ((part["tree_hit8"]) & (~part["pinn_hit8"])).mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def choose_best_delta(train_df, subset_mask=None, grid=None):
    if grid is None:
        grid = np.arange(-0.08, 0.0801, 0.0025)
    if subset_mask is None:
        subset_mask = np.ones(len(train_df), dtype=bool)
    part = train_df.loc[subset_mask].copy()
    if len(part) == 0:
        return 0.0, np.nan
    best_delta = 0.0
    best_score = -np.inf
    base_pred = part["blend_pred"].to_numpy(float)
    capacity = part["capacity"].to_numpy(float)
    for delta in grid:
        part["_tmp_pred"] = np.clip(base_pred + delta * capacity, 0.0, capacity)
        score, _, _ = official_like_score(part, "_tmp_pred")
        if score > best_score:
            best_score = score
            best_delta = float(delta)
    return best_delta, float(best_score)


def evaluate_delta(train_df, test_df, mode, grid):
    rows = []
    pred = test_df["blend_pred"].to_numpy(float).copy()
    deltas = []
    if mode == "global_delta":
        delta, train_score = choose_best_delta(train_df, grid=grid)
        pred = pred + delta * test_df["capacity"].to_numpy(float)
        deltas.append({"target": "global", "delta": delta, "train_score": train_score, "n_train": len(train_df)})
    elif mode == "group_delta":
        for group in GROUPS:
            train_mask = train_df["group"].eq(group)
            test_mask = test_df["group"].eq(group).to_numpy()
            delta, train_score = choose_best_delta(train_df, train_mask.to_numpy(), grid)
            pred[test_mask] = pred[test_mask] + delta * test_df.loc[test_mask, "capacity"].to_numpy(float)
            deltas.append({"target": group, "delta": delta, "train_score": train_score, "n_train": int(train_mask.sum())})
    elif mode == "pred_bin_delta":
        for group in GROUPS:
            for bin_label in test_df["pred_ratio_bin"].cat.categories:
                train_mask = train_df["group"].eq(group) & train_df["pred_ratio_bin"].eq(bin_label)
                test_mask = (test_df["group"].eq(group) & test_df["pred_ratio_bin"].eq(bin_label)).to_numpy()
                if train_mask.sum() < 250 or test_mask.sum() == 0:
                    delta, train_score = 0.0, np.nan
                else:
                    delta, train_score = choose_best_delta(train_df, train_mask.to_numpy(), grid)
                pred[test_mask] = pred[test_mask] + delta * test_df.loc[test_mask, "capacity"].to_numpy(float)
                deltas.append(
                    {
                        "target": f"{group}|{bin_label}",
                        "delta": delta,
                        "train_score": train_score,
                        "n_train": int(train_mask.sum()),
                    }
                )
    else:
        raise ValueError(mode)

    out = test_df.copy()
    out[f"{mode}_pred"] = np.clip(pred, 0.0, out["capacity"].to_numpy(float))
    score, nmae, ficr = official_like_score(out, f"{mode}_pred")
    rows.append(
        {
            "mode": mode,
            "pred_year": int(test_df["pred_year"].iloc[0]),
            "score": score,
            "nmae": nmae,
            "ficr": ficr,
            "n": len(out),
            "deltas": ";".join(f"{d['target']}={d['delta']:+.4f}" for d in deltas),
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(deltas).assign(mode=mode, pred_year=int(test_df["pred_year"].iloc[0]))


def evaluate_weight_selection(train_df, test_df, weights):
    best_weight = None
    best_train_score = -np.inf
    for weight in weights:
        train_tmp = train_df.copy()
        train_tmp["_tmp_pred"] = np.clip(
            (1.0 - weight) * train_tmp["pinn_pred"] + weight * train_tmp["tree_pred"],
            0.0,
            train_tmp["capacity"],
        )
        score, _, _ = official_like_score(train_tmp, "_tmp_pred")
        if score > best_train_score:
            best_train_score = score
            best_weight = float(weight)

    out = test_df.copy()
    out["selected_weight_pred"] = np.clip(
        (1.0 - best_weight) * out["pinn_pred"] + best_weight * out["tree_pred"],
        0.0,
        out["capacity"],
    )
    score, nmae, ficr = official_like_score(out, "selected_weight_pred")
    return {
        "mode": "crossyear_selected_tree_weight",
        "pred_year": int(test_df["pred_year"].iloc[0]),
        "tree_weight": best_weight,
        "train_score": best_train_score,
        "score": score,
        "nmae": nmae,
        "ficr": ficr,
        "n": len(out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-csv", default=DEFAULT_INPUT)
    parser.add_argument("--tree-weight", type=float, default=0.5)
    parser.add_argument("--stem", default="ficr_postprocess_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.aligned_csv, encoding="utf-8-sig")
    df = add_base_columns(raw, args.tree_weight)

    score_parts = []
    fold_score_parts = []
    for col in ["pinn_pred", "tree_pred", "blend_pred"]:
        score_parts.append(score_frame(df, col))
        fold_score_parts.append(score_frame_by_fold(df, col))
    model_scores = pd.concat(score_parts, ignore_index=True)
    model_fold_scores = pd.concat(fold_score_parts, ignore_index=True)

    err_bins = [-np.inf, 0.04, 0.055, 0.06, 0.08, 0.10, 0.15, np.inf]
    err_labels = ["<=4", "4-5.5", "5.5-6", "6-8", "8-10", "10-15", ">15"]
    pred_bins = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, np.inf]
    pred_labels = ["10-20", "20-35", "35-50", "50-65", "65-80", "80-95", "95+"]
    gap_bins = [-0.001, 0.02, 0.04, 0.06, 0.10, 0.15, np.inf]
    gap_labels = ["0-2", "2-4", "4-6", "6-10", "10-15", "15+"]
    df["error_band"] = pd.cut(df["error_rate"], bins=err_bins, labels=err_labels)
    df["pred_ratio_bin"] = pd.cut(df["pred_ratio"], bins=pred_bins, labels=pred_labels)
    df["gap_bin"] = pd.cut(df["abs_tree_pinn_gap"], bins=gap_bins, labels=gap_labels)

    error_band_summary = summarize_bins(df, ["group", "error_band"])
    pred_bin_summary = summarize_bins(df, ["group", "pred_ratio_bin"])
    gap_bin_summary = summarize_bins(df, ["group", "gap_bin"])

    near_threshold = df[df["error_rate"].between(0.055, 0.085)].copy()
    near_threshold_summary = summarize_bins(near_threshold, ["group", "pred_ratio_bin"])

    delta_grid = np.arange(-0.08, 0.0801, 0.0025)
    cal_rows = []
    delta_rows = []
    weight_rows = []
    for pred_year in sorted(df["pred_year"].unique()):
        train_df = df[df["pred_year"] != pred_year].reset_index(drop=True)
        test_df = df[df["pred_year"] == pred_year].reset_index(drop=True)
        for mode in ["global_delta", "group_delta", "pred_bin_delta"]:
            cal, deltas = evaluate_delta(train_df, test_df, mode, delta_grid)
            cal_rows.append(cal)
            delta_rows.append(deltas)
        weight_rows.append(evaluate_weight_selection(train_df, test_df, np.arange(0.0, 1.0001, 0.05)))

    crossyear_calibration = pd.concat(cal_rows, ignore_index=True)
    crossyear_deltas = pd.concat(delta_rows, ignore_index=True)
    crossyear_weights = pd.DataFrame(weight_rows)

    model_scores.to_csv(RESULTS_DIR / f"{args.stem}_model_scores.csv", index=False, encoding="utf-8-sig")
    model_fold_scores.to_csv(RESULTS_DIR / f"{args.stem}_model_fold_scores.csv", index=False, encoding="utf-8-sig")
    error_band_summary.to_csv(RESULTS_DIR / f"{args.stem}_error_band_summary.csv", index=False, encoding="utf-8-sig")
    pred_bin_summary.to_csv(RESULTS_DIR / f"{args.stem}_pred_ratio_bins.csv", index=False, encoding="utf-8-sig")
    gap_bin_summary.to_csv(RESULTS_DIR / f"{args.stem}_disagreement_bins.csv", index=False, encoding="utf-8-sig")
    near_threshold_summary.to_csv(RESULTS_DIR / f"{args.stem}_near_threshold_bins.csv", index=False, encoding="utf-8-sig")
    crossyear_calibration.to_csv(RESULTS_DIR / f"{args.stem}_crossyear_calibration.csv", index=False, encoding="utf-8-sig")
    crossyear_deltas.to_csv(RESULTS_DIR / f"{args.stem}_crossyear_deltas.csv", index=False, encoding="utf-8-sig")
    crossyear_weights.to_csv(RESULTS_DIR / f"{args.stem}_crossyear_weights.csv", index=False, encoding="utf-8-sig")

    print("=== model scores ===")
    print(model_fold_scores[model_fold_scores["group"].eq("overall_fold_mean")].to_string(index=False))
    print("\n=== cross-year delta calibration ===")
    print(
        crossyear_calibration.groupby("mode", as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), worst=("score", "min"))
        .sort_values("score", ascending=False)
        .to_string(index=False)
    )
    print("\n=== cross-year selected tree weight ===")
    print(crossyear_weights.to_string(index=False))
    print("\n=== high-disagreement bins ===")
    print(gap_bin_summary.sort_values(["group", "gap_bin"]).to_string(index=False))


if __name__ == "__main__":
    main()
