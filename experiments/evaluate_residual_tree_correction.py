import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor

import _bootstrap  # noqa: F401
from tune_power_lgbm_hyperparams import parse_list, prepare_fold_cache, sample_weight
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")
KEY = ["forecast_kst_dtm", "pred_year", "group"]
XGB_PARAM_COLS = [
    "random_state",
    "n_jobs",
    "objective",
    "tree_method",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "gamma",
]
EXTRA_PARAM_COLS = [
    "random_state",
    "n_jobs",
    "n_estimators",
    "min_samples_leaf",
    "max_features",
    "max_depth",
    "bootstrap",
]


def clean_xgb_params(row):
    params = {col: row[col] for col in XGB_PARAM_COLS}
    for col in ["random_state", "n_jobs", "n_estimators", "max_depth"]:
        params[col] = int(params[col])
    for col in ["learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma"]:
        params[col] = float(params[col])
    return params


def clean_extra_params(row):
    params = {col: row[col] for col in EXTRA_PARAM_COLS}
    for col in ["random_state", "n_jobs", "n_estimators", "min_samples_leaf"]:
        params[col] = int(params[col])
    params["max_features"] = float(params["max_features"])
    params["max_depth"] = None if pd.isna(params["max_depth"]) else int(float(params["max_depth"]))
    if isinstance(params["bootstrap"], str):
        params["bootstrap"] = params["bootstrap"].lower() == "true"
    else:
        params["bootstrap"] = bool(params["bootstrap"])
    return params


def read_best_params(path, model_type):
    best = pd.read_csv(path, encoding="utf-8-sig")
    params = {}
    policies = {}
    for row in best.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        group = row_s["group"]
        if model_type == "xgb":
            params[group] = clean_xgb_params(row_s)
        elif model_type in {"extra", "rf"}:
            params[group] = clean_extra_params(row_s)
        else:
            raise ValueError(f"unknown model_type: {model_type}")
        policies[group] = {
            "min_output_ratio": float(row_s["min_output_ratio"]),
            "weight_policy": str(row_s["weight_policy"]),
        }
    return params, policies


def make_model(model_type, params):
    if model_type == "xgb":
        return XGBRegressor(**params)
    if model_type == "extra":
        return ExtraTreesRegressor(**params)
    if model_type == "rf":
        return RandomForestRegressor(**params)
    raise ValueError(f"unknown model_type: {model_type}")


def load_base_predictions(pinn_path, tree_path):
    tree = pd.read_csv(tree_path, encoding="utf-8-sig")
    tree["forecast_kst_dtm"] = pd.to_datetime(tree["forecast_kst_dtm"])
    tree = tree[KEY + ["actual", "pred"]].rename(columns={"pred": "tree_pred"})

    pinn = pd.read_csv(pinn_path, encoding="utf-8-sig")
    pinn["forecast_kst_dtm"] = pd.to_datetime(pinn["forecast_kst_dtm"])
    pinn_long = pinn.melt(
        id_vars=["forecast_kst_dtm", "pred_year", "train_years"],
        value_vars=TARGET_COLS,
        var_name="group",
        value_name="pinn_pred",
    )[KEY + ["pinn_pred"]]

    base = tree.merge(pinn_long, on=KEY, how="inner")
    base["base_pred"] = 0.5 * base["pinn_pred"] + 0.5 * base["tree_pred"]
    base["residual"] = base["actual"] - base["base_pred"]
    return base


def score_group(actual, pred, group):
    capacity = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    nmae, ficr = group_nmae_ficr(actual, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def score_predictions(df, pred_col):
    rows = []
    for (pred_year, group), part in df.groupby(["pred_year", "group"]):
        score, nmae, ficr = score_group(part["actual"], part[pred_col], group)
        rows.append({"pred_year": pred_year, "group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(part)})
    scores = pd.DataFrame(rows)
    fold_means = scores.groupby("pred_year", as_index=False)[["score", "nmae", "ficr"]].mean()
    group_means = scores.groupby("group", as_index=False)[["score", "nmae", "ficr"]].mean()
    summary = {
        "mean_score": fold_means["score"].mean(),
        "mean_nmae": fold_means["nmae"].mean(),
        "mean_ficr": fold_means["ficr"].mean(),
        "worst_fold": fold_means["score"].min(),
        "std_score": fold_means["score"].std(),
    }
    return scores, fold_means, group_means, summary


def fit_residual_oof(model_type, groups, cache, base, params_by_group, policies_by_group):
    pred_parts = []
    for group in groups:
        group_base = base[base["group"] == group][["forecast_kst_dtm", "actual", "base_pred", "residual"]]
        capacity = GROUP_CAPACITY_KWH[group]
        params = params_by_group[group]
        policy = policies_by_group[group]
        print(f"\n=== residual {model_type} {group}: policy={policy} ===")

        for fold in cache[group]:
            train_meta = pd.DataFrame(
                {
                    "orig_pos": np.arange(len(fold["y_train"])),
                    "forecast_kst_dtm": pd.to_datetime(fold["time_train"]).to_numpy(),
                    "actual_train": fold["y_train"].to_numpy(float),
                }
            ).merge(group_base, on="forecast_kst_dtm", how="inner")
            val_meta = pd.DataFrame(
                {
                    "orig_pos": np.arange(len(fold["y_val"])),
                    "forecast_kst_dtm": pd.to_datetime(fold["time_val"]).to_numpy(),
                    "actual": fold["y_val"].to_numpy(float),
                }
            ).merge(group_base[["forecast_kst_dtm", "base_pred"]], on="forecast_kst_dtm", how="inner")

            min_output = capacity * policy["min_output_ratio"]
            train_mask = train_meta["actual_train"].to_numpy(float) >= min_output
            used = train_meta.loc[train_mask].copy()
            if len(used) < 500:
                continue

            x_train = fold["x_train"].iloc[used["orig_pos"].to_numpy(int)]
            y_resid = used["residual"].to_numpy(float)
            actual_used = used["actual_train"].to_numpy(float)
            weights = sample_weight(actual_used, group, policy["weight_policy"])

            model = make_model(model_type, params)
            model.fit(x_train, y_resid, sample_weight=weights)
            x_val = fold["x_val"].iloc[val_meta["orig_pos"].to_numpy(int)]
            resid_pred = model.predict(x_val)

            pred_parts.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": val_meta["forecast_kst_dtm"].to_numpy(),
                        "pred_year": fold["pred_year"],
                        "train_years": fold["train_years"],
                        "group": group,
                        "actual": val_meta["actual"].to_numpy(float),
                        "base_pred": val_meta["base_pred"].to_numpy(float),
                        f"resid_{model_type}": resid_pred,
                    }
                )
            )
            print(f"{group} pred_year={fold['pred_year']}: n_train={len(used)} n_val={len(val_meta)}")
    return pd.concat(pred_parts, ignore_index=True)


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def evaluate_corrections(df, alphas, clips):
    pred_df = df.copy()
    residual_cols = [col for col in pred_df.columns if col.startswith("resid_") and col != "resid_mean"]
    if len(residual_cols) >= 2:
        pred_df["resid_mean"] = pred_df[residual_cols].mean(axis=1)
        residual_cols.append("resid_mean")
    elif "resid_mean" in pred_df.columns:
        residual_cols.append("resid_mean")

    summary_rows = []
    group_rows = []
    fold_rows = []

    pred_df["pred_base"] = pred_df["base_pred"]
    _, folds, groups, summary = score_predictions(pred_df, "pred_base")
    summary_rows.append({"variant": "base", "alpha": 0.0, "clip_ratio": 0.0, **summary})
    groups.insert(0, "clip_ratio", 0.0)
    groups.insert(0, "alpha", 0.0)
    groups.insert(0, "variant", "base")
    group_rows.append(groups)
    folds.insert(0, "clip_ratio", 0.0)
    folds.insert(0, "alpha", 0.0)
    folds.insert(0, "variant", "base")
    fold_rows.append(folds)

    for residual_col in residual_cols:
        for alpha in alphas:
            if alpha == 0:
                continue
            for clip_ratio in clips:
                corrected = pred_df[residual_col].to_numpy(float)
                if clip_ratio > 0:
                    caps = pred_df["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
                    corrected = np.clip(corrected, -caps * clip_ratio, caps * clip_ratio)
                pred_col = f"pred_{residual_col}_a{alpha:g}_c{clip_ratio:g}"
                pred_df[pred_col] = pred_df["base_pred"].to_numpy(float) + alpha * corrected
                _, folds, groups, summary = score_predictions(pred_df, pred_col)
                summary_rows.append({"variant": residual_col, "alpha": alpha, "clip_ratio": clip_ratio, **summary})
                groups.insert(0, "clip_ratio", clip_ratio)
                groups.insert(0, "alpha", alpha)
                groups.insert(0, "variant", residual_col)
                group_rows.append(groups)
                folds.insert(0, "clip_ratio", clip_ratio)
                folds.insert(0, "alpha", alpha)
                folds.insert(0, "variant", residual_col)
                fold_rows.append(folds)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_score", ascending=False)
    group_df = pd.concat(group_rows, ignore_index=True)
    fold_df = pd.concat(fold_rows, ignore_index=True)
    return pred_df, summary_df, group_df, fold_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--pinn-oof", default="results/pinn_effective_grid_g1_year_bagging_lgbm_time_oof_oof_predictions.csv")
    parser.add_argument("--tree-oof", default="results/power_lgbm_best_v2_l1_predictions.csv")
    parser.add_argument("--xgb-best", default="results/power_xgb_v1_4_best.csv")
    parser.add_argument("--extra-best", default="results/power_extra_v1_8_best.csv")
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--clips", default="0,0.03,0.05,0.08,0.12")
    parser.add_argument("--prediction-input", default=None)
    parser.add_argument("--stem", default="residual_tree_correction_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.prediction_input:
        df = pd.read_csv(args.prediction_input, encoding="utf-8-sig")
    else:
        groups = parse_list(args.groups)
        base = load_base_predictions(args.pinn_oof, args.tree_oof)
        cache = prepare_fold_cache(groups)

        xgb_params, xgb_policies = read_best_params(args.xgb_best, "xgb")
        extra_params, extra_policies = read_best_params(args.extra_best, "extra")
        xgb_pred = fit_residual_oof("xgb", groups, cache, base, xgb_params, xgb_policies)
        extra_pred = fit_residual_oof("extra", groups, cache, base, extra_params, extra_policies)

        df = xgb_pred.merge(
            extra_pred[KEY + ["resid_extra"]],
            on=KEY,
            how="inner",
        )
    pred_df, summary_df, group_df, fold_df = evaluate_corrections(
        df,
        alphas=parse_float_list(args.alphas),
        clips=parse_float_list(args.clips),
    )

    pred_df.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(RESULTS_DIR / f"{args.stem}_group_scores.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(RESULTS_DIR / f"{args.stem}_fold_scores.csv", index=False, encoding="utf-8-sig")

    print("\n=== residual correction summary top 20 ===")
    print(summary_df.head(20).to_string(index=False))
    print(f"saved results/{args.stem}_summary.csv")


if __name__ == "__main__":
    main()
