import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr


RESULTS_DIR = Path("results")
KEYS = ["forecast_kst_dtm", "pred_year", "group"]
DEFAULT_ALIGNED = "results/pinn_lgbmteacher_powerlgbm_v2_l1_blend_aligned_predictions.csv"
DEFAULT_XGB = "results/power_xgb_v1_4_best_predictions.csv"
DEFAULT_EXTRA = "results/power_extra_v1_8_best_predictions.csv"
DEFAULT_PRUNED = "results/family_pruned_lgbm_selected_low_v1_predictions.csv"


def load_base(path, tree_weight):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["capacity"] = df["group"].map(GROUP_CAPACITY_KWH).astype(float)
    df["base_pred"] = (1.0 - tree_weight) * df["pinn_pred"] + tree_weight * df["tree_pred"]
    df["base_pred"] = df["base_pred"].clip(lower=0.0, upper=df["capacity"])
    return df[KEYS + ["actual", "capacity", "pinn_pred", "tree_pred", "base_pred"]]


def load_component(path, name, filters=None):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    if filters:
        for col, value in filters.items():
            df = df[df[col].eq(value)]
    if "pred" not in df.columns:
        return None
    return df[KEYS + ["pred"]].rename(columns={"pred": name}).drop_duplicates(KEYS)


def add_components(base, args):
    out = base.copy()
    components = {
        "xgb_pred": load_component(args.xgb_csv, "xgb_pred"),
        "extra_pred": load_component(args.extra_csv, "extra_pred"),
        "pruned_pred": load_component(args.pruned_csv, "pruned_pred", filters={"variant": "drop_selected_low"}),
    }
    for name, comp in components.items():
        if comp is None:
            continue
        out = out.merge(comp, on=KEYS, how="left")
        out[name] = out[name].fillna(out["base_pred"])
        out[name] = out[name].clip(lower=0.0, upper=out["capacity"])
    return out


def add_features(df):
    out = df.copy()
    pred_cols = [col for col in ["pinn_pred", "tree_pred", "xgb_pred", "extra_pred", "pruned_pred", "base_pred"] if col in out]
    for col in pred_cols:
        out[f"{col}_ratio"] = out[col].to_numpy(float) / out["capacity"].to_numpy(float)

    ratio_cols = [f"{col}_ratio" for col in pred_cols]
    out["pred_mean_ratio"] = out[ratio_cols].mean(axis=1)
    out["pred_std_ratio"] = out[ratio_cols].std(axis=1).fillna(0.0)
    out["pred_min_ratio"] = out[ratio_cols].min(axis=1)
    out["pred_max_ratio"] = out[ratio_cols].max(axis=1)
    out["tree_minus_pinn_ratio"] = out["tree_pred_ratio"] - out["pinn_pred_ratio"]
    if "xgb_pred_ratio" in out:
        out["xgb_minus_base_ratio"] = out["xgb_pred_ratio"] - out["base_pred_ratio"]
    if "pruned_pred_ratio" in out:
        out["pruned_minus_base_ratio"] = out["pruned_pred_ratio"] - out["base_pred_ratio"]

    dt = pd.to_datetime(out["forecast_kst_dtm"])
    hod = 2.0 * np.pi * dt.dt.hour.to_numpy(float) / 24.0
    doy = 2.0 * np.pi * dt.dt.dayofyear.to_numpy(float) / 365.25
    out["sin_hod"] = np.sin(hod)
    out["cos_hod"] = np.cos(hod)
    out["sin_doy"] = np.sin(doy)
    out["cos_doy"] = np.cos(doy)
    out["target_ratio"] = out["actual"].to_numpy(float) / out["capacity"].to_numpy(float)
    return out


def model_specs():
    return {
        "ridge_a001": Ridge(alpha=0.01),
        "ridge_a01": Ridge(alpha=0.1),
        "ridge_a1": Ridge(alpha=1.0),
        "ridge_a10": Ridge(alpha=10.0),
        "ridge_a100": Ridge(alpha=100.0),
        "elastic_a001_l1_005": ElasticNet(alpha=0.001, l1_ratio=0.05, max_iter=10000, random_state=7),
        "elastic_a005_l1_010": ElasticNet(alpha=0.005, l1_ratio=0.10, max_iter=10000, random_state=7),
        "huber_a0001": HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500),
        "huber_a001": HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=500),
        "positive_linear": LinearRegression(positive=True),
    }


def make_pipeline_for(model, use_group_onehot):
    numeric = [
        "pinn_pred_ratio",
        "tree_pred_ratio",
        "base_pred_ratio",
        "pred_mean_ratio",
        "pred_std_ratio",
        "pred_min_ratio",
        "pred_max_ratio",
        "tree_minus_pinn_ratio",
        "sin_hod",
        "cos_hod",
        "sin_doy",
        "cos_doy",
    ]
    for optional in ["xgb_pred_ratio", "extra_pred_ratio", "pruned_pred_ratio", "xgb_minus_base_ratio", "pruned_minus_base_ratio"]:
        if optional not in numeric:
            numeric.append(optional)
    categorical = ["group"] if use_group_onehot else []
    transformers = [("num", StandardScaler(), numeric)]
    if use_group_onehot:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    pre = ColumnTransformer(transformers, remainder="drop")
    return make_pipeline(pre, clone(model)), numeric + categorical


def score_fold(part, pred_col, variant):
    rows = []
    pred_year = int(part["pred_year"].iloc[0])
    for group, group_part in part.groupby("group"):
        pred = np.clip(group_part[pred_col].to_numpy(float), 0.0, GROUP_CAPACITY_KWH[group])
        metric_part = group_part[group_part["actual"] >= group_part["capacity"] * 0.10]
        metric_pred = pred[group_part["actual"].to_numpy(float) >= group_part["capacity"].to_numpy(float) * 0.10]
        if len(metric_part) == 0:
            continue
        nmae, ficr = group_nmae_ficr(metric_part["actual"], metric_pred, GROUP_CAPACITY_KWH[group])
        score = 0.5 * (1.0 - nmae) + 0.5 * ficr
        rows.append(
            {
                "variant": variant,
                "pred_year": pred_year,
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(metric_part),
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


def summarize(scores):
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
        .sort_values(["mean_score", "mean_nmae"], ascending=[False, True])
    )


def evaluate_variant(df, name, model, scope, train_metric_only):
    pred_col = f"pred_{name}_{scope}_{'metric' if train_metric_only else 'all'}"
    out = df.copy()
    out[pred_col] = out["base_pred"]
    use_group_onehot = scope == "global"
    metric_mask = out["actual"] >= out["capacity"] * 0.10

    for pred_year in sorted(out["pred_year"].dropna().unique()):
        eval_mask = out["pred_year"].eq(pred_year)
        if scope == "global":
            train_mask = ~eval_mask
            if train_metric_only:
                train_mask &= metric_mask
            if train_mask.sum() < 500:
                continue
            pipe, features = make_pipeline_for(model, use_group_onehot=True)
            pipe.fit(out.loc[train_mask, features], out.loc[train_mask, "target_ratio"])
            pred_ratio = pipe.predict(out.loc[eval_mask, features])
            out.loc[eval_mask, pred_col] = np.clip(pred_ratio, 0.0, 1.0) * out.loc[eval_mask, "capacity"].to_numpy(float)
            continue

        for group in sorted(out["group"].dropna().unique()):
            group_mask = out["group"].eq(group)
            train_mask = group_mask & ~eval_mask
            if train_metric_only:
                train_mask &= metric_mask
            local_eval = group_mask & eval_mask
            if train_mask.sum() < 300 or local_eval.sum() == 0:
                continue
            pipe, features = make_pipeline_for(model, use_group_onehot=False)
            pipe.fit(out.loc[train_mask, features], out.loc[train_mask, "target_ratio"])
            pred_ratio = pipe.predict(out.loc[local_eval, features])
            out.loc[local_eval, pred_col] = np.clip(pred_ratio, 0.0, 1.0) * out.loc[local_eval, "capacity"].to_numpy(float)

    variant = f"{name}_{scope}_{'metric' if train_metric_only else 'all'}"
    score_rows = []
    for _, fold in out.groupby("pred_year"):
        score_rows.extend(score_fold(fold, pred_col, variant))
    pred_out = out[KEYS + ["actual", "capacity", "base_pred", pred_col]].rename(columns={pred_col: "pred"})
    pred_out["variant"] = variant
    return score_rows, pred_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-csv", default=DEFAULT_ALIGNED)
    parser.add_argument("--xgb-csv", default=DEFAULT_XGB)
    parser.add_argument("--extra-csv", default=DEFAULT_EXTRA)
    parser.add_argument("--pruned-csv", default=DEFAULT_PRUNED)
    parser.add_argument("--tree-weight", type=float, default=0.6)
    parser.add_argument("--stem", default="oof_stacker_v1_w60")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = add_features(add_components(load_base(args.aligned_csv, args.tree_weight), args))

    score_rows = []
    pred_parts = []
    for model_name, model in model_specs().items():
        for scope in ["global", "group"]:
            for train_metric_only in [False, True]:
                rows, preds = evaluate_variant(df, model_name, model, scope, train_metric_only)
                score_rows.extend(rows)
                pred_parts.append(preds)
                fold_scores = [row["score"] for row in rows if row["group"] == "fold_mean"]
                print(f"{model_name} {scope} metric_only={train_metric_only}: score={np.mean(fold_scores):.5f}")

    scores = pd.DataFrame(score_rows)
    summary = summarize(scores)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n=== top stackers ===")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
