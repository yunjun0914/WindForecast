import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")
DEFAULT_INPUT = "results/pinn_lgbmteacher_powerlgbm_v2_l1_blend_aligned_predictions.csv"
GROUPS = TARGET_COLS


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def add_base_columns(df, tree_weight):
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["capacity"] = out["group"].map(GROUP_CAPACITY_KWH).astype(float)
    out["pinn_ratio"] = out["pinn_pred"] / out["capacity"]
    out["tree_ratio"] = out["tree_pred"] / out["capacity"]
    out["blend_pred"] = (1.0 - tree_weight) * out["pinn_pred"] + tree_weight * out["tree_pred"]
    out["blend_pred"] = out["blend_pred"].clip(lower=0.0, upper=out["capacity"])
    out["blend_ratio"] = out["blend_pred"] / out["capacity"]
    out["gap_ratio"] = out["tree_ratio"] - out["pinn_ratio"]
    out["abs_gap_ratio"] = out["gap_ratio"].abs()
    out["pred_min_ratio"] = out[["pinn_ratio", "tree_ratio"]].min(axis=1)
    out["pred_max_ratio"] = out[["pinn_ratio", "tree_ratio"]].max(axis=1)
    out["residual_ratio"] = (out["actual"] - out["blend_pred"]) / out["capacity"]
    out = pd.concat([out, pd.get_dummies(out["group"], prefix="group", dtype=float)], axis=1)
    return out[out["actual"] >= out["capacity"] * 0.10].reset_index(drop=True)


def feature_columns(df):
    cols = [
        "pinn_ratio",
        "tree_ratio",
        "blend_ratio",
        "gap_ratio",
        "abs_gap_ratio",
        "pred_min_ratio",
        "pred_max_ratio",
    ]
    cols += [col for col in df.columns if col.startswith("group_")]
    return cols


def score_group(actual, pred, group):
    nmae, ficr = group_nmae_ficr(actual, pred, GROUP_CAPACITY_KWH[group])
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def score_fold(df, pred_col, variant):
    rows = []
    pred_year = int(df["pred_year"].iloc[0])
    for group, part in df.groupby("group"):
        score, nmae, ficr = score_group(part["actual"], part[pred_col], group)
        rows.append(
            {
                "variant": variant,
                "pred_year": pred_year,
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(part),
            }
        )
    group_rows = list(rows)
    rows.append(
        {
            "variant": variant,
            "pred_year": pred_year,
            "group": "fold_mean",
            "score": float(np.mean([row["score"] for row in group_rows])),
            "nmae": float(np.mean([row["nmae"] for row in group_rows])),
            "ficr": float(np.mean([row["ficr"] for row in group_rows])),
            "n": int(sum(row["n"] for row in group_rows)),
        }
    )
    return rows


def score_summary(scores):
    fold = scores[scores["group"].eq("fold_mean")]
    return (
        fold.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
        )
        .sort_values("mean_score", ascending=False)
    )


def make_model(alpha, seed):
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        n_estimators=800,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=250,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
    )


def choose_global_delta(train_df, grid):
    best_delta = 0.0
    best_score = -np.inf
    base = train_df["blend_pred"].to_numpy(float)
    capacity = train_df["capacity"].to_numpy(float)
    for delta in grid:
        tmp = train_df.copy()
        tmp["_pred"] = np.clip(base + delta * capacity, 0.0, capacity)
        fold_scores = []
        for group, part in tmp.groupby("group"):
            score, _, _ = score_group(part["actual"], part["_pred"], group)
            fold_scores.append(score)
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score = score
            best_delta = float(delta)
    return best_delta, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-csv", default=DEFAULT_INPUT)
    parser.add_argument("--tree-weight", type=float, default=0.6)
    parser.add_argument("--alphas", default="0.45,0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--clips", default="0.01,0.02,0.03,0.05")
    parser.add_argument("--fixed-deltas", default="0.01,0.015,0.0175,0.02,0.025,0.03")
    parser.add_argument("--stem", default="quantile_residual_calibration_v1_w60")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.aligned_csv, encoding="utf-8-sig")
    df = add_base_columns(raw, args.tree_weight)
    features = feature_columns(df)
    alphas = parse_float_list(args.alphas)
    clips = parse_float_list(args.clips)
    fixed_deltas = parse_float_list(args.fixed_deltas)

    score_rows = []
    delta_rows = []
    pred_parts = []
    for pred_year in sorted(df["pred_year"].unique()):
        train = df[df["pred_year"] != pred_year].reset_index(drop=True)
        test = df[df["pred_year"] == pred_year].reset_index(drop=True)

        baseline = test.copy()
        baseline["baseline_pred"] = baseline["blend_pred"]
        score_rows.extend(score_fold(baseline, "baseline_pred", "baseline_blend"))

        for fixed_delta in fixed_deltas:
            fixed = test.copy()
            fixed["fixed_delta_pred"] = np.clip(
                fixed["blend_pred"] + fixed_delta * fixed["capacity"], 0.0, fixed["capacity"]
            )
            variant = f"fixed_delta_{int(round(fixed_delta * 10000)):04d}"
            score_rows.extend(score_fold(fixed, "fixed_delta_pred", variant))
            delta_rows.append(
                {
                    "pred_year": int(pred_year),
                    "variant": variant,
                    "alpha": np.nan,
                    "clip": np.nan,
                    "global_delta": fixed_delta,
                    "train_score": np.nan,
                    "delta_mean": fixed_delta,
                    "delta_std": 0.0,
                    "delta_p05": fixed_delta,
                    "delta_p50": fixed_delta,
                    "delta_p95": fixed_delta,
                }
            )

        grid = np.arange(-0.08, 0.0801, 0.0025)
        global_delta, global_train_score = choose_global_delta(train, grid)
        global_df = test.copy()
        global_df["global_delta_pred"] = np.clip(
            global_df["blend_pred"] + global_delta * global_df["capacity"], 0.0, global_df["capacity"]
        )
        score_rows.extend(score_fold(global_df, "global_delta_pred", "global_delta"))
        delta_rows.append(
            {
                "pred_year": int(pred_year),
                "variant": "global_delta",
                "alpha": np.nan,
                "clip": np.nan,
                "global_delta": global_delta,
                "train_score": global_train_score,
                "delta_mean": global_delta,
                "delta_std": 0.0,
                "delta_p05": global_delta,
                "delta_p50": global_delta,
                "delta_p95": global_delta,
            }
        )

        x_train = train[features]
        y_train = train["residual_ratio"]
        x_test = test[features]
        weights = np.clip(train["actual"] / train["capacity"], 0.1, None)

        for alpha in alphas:
            model = make_model(alpha, seed=2026070800 + int(alpha * 1000) + int(pred_year))
            model.fit(x_train, y_train, sample_weight=weights)
            raw_delta = model.predict(x_test)
            for clip in clips:
                delta = np.clip(raw_delta, -clip, clip)
                variant = f"q{int(round(alpha * 100)):02d}_clip{int(round(clip * 1000)):03d}"
                part = test.copy()
                part["delta_ratio"] = delta
                part["quantile_pred"] = np.clip(part["blend_pred"] + delta * part["capacity"], 0.0, part["capacity"])
                score_rows.extend(score_fold(part, "quantile_pred", variant))
                delta_rows.append(
                    {
                        "pred_year": int(pred_year),
                        "variant": variant,
                        "alpha": alpha,
                        "clip": clip,
                        "global_delta": np.nan,
                        "train_score": np.nan,
                        "delta_mean": float(np.mean(delta)),
                        "delta_std": float(np.std(delta)),
                        "delta_p05": float(np.quantile(delta, 0.05)),
                        "delta_p50": float(np.quantile(delta, 0.50)),
                        "delta_p95": float(np.quantile(delta, 0.95)),
                    }
                )
                pred_parts.append(
                    part[
                        [
                            "forecast_kst_dtm",
                            "pred_year",
                            "group",
                            "actual",
                            "pinn_pred",
                            "tree_pred",
                            "blend_pred",
                            "delta_ratio",
                            "quantile_pred",
                        ]
                    ].assign(variant=variant)
                )

    scores = pd.DataFrame(score_rows)
    summary = score_summary(scores)
    deltas = pd.DataFrame(delta_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(RESULTS_DIR / f"{args.stem}_deltas.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")

    print("=== summary ===")
    print(summary.head(15).to_string(index=False))
    print("\n=== deltas top variants ===")
    top_variants = summary.head(8)["variant"].tolist()
    print(deltas[deltas["variant"].isin(top_variants)].to_string(index=False))
    print(f"\nfeatures={features}")


if __name__ == "__main__":
    main()
