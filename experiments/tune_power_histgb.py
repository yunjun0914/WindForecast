import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import _bootstrap  # noqa: F401
from analyze_feature_families import classify_feature
from tune_power_lgbm_hyperparams import (
    log_uniform,
    parse_list,
    prepare_fold_cache,
    sample_weight,
    score_one,
)
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


RESULTS_DIR = Path("results")
DEFAULT_DROP_FAMILIES = {
    "selected_low": ["power_curve", "precipitation", "meteo_other", "wind_regime_poly"],
    "none": [],
}


def sample_loss(rng):
    name = rng.choice(["squared_error", "absolute_error", "quantile"])
    if name == "quantile":
        return name, float(rng.choice([0.45, 0.50, 0.55, 0.60, 0.65]))
    return name, None


def sample_params(rng, seed):
    loss, quantile = sample_loss(rng)
    params = {
        "loss": loss,
        "quantile": quantile,
        "learning_rate": log_uniform(rng, 0.015, 0.090),
        "max_iter": int(rng.integers(350, 1401)),
        "max_leaf_nodes": int(rng.choice([15, 24, 31, 48, 63, 96, 127])),
        "max_depth": rng.choice([None, 4, 5, 6, 8, 10, 12]),
        "min_samples_leaf": int(rng.choice([20, 40, 60, 80, 120, 180, 260])),
        "l2_regularization": log_uniform(rng, 1e-4, 60.0),
        "max_features": float(rng.uniform(0.55, 1.0)),
        "max_bins": int(rng.choice([127, 255])),
        "early_stopping": False,
        "random_state": int(seed % (2**32 - 1)),
    }
    if params["loss"] != "quantile":
        params["quantile"] = None
    return params


def sample_policy(rng):
    return {
        "min_output_ratio": float(rng.choice([0.00, 0.03, 0.05, 0.10, 0.15])),
        "weight_policy": rng.choice(["none", "metric_x2", "actual_sqrt"]),
    }


def clean_params(row):
    params = {
        "loss": row["loss"],
        "quantile": None if pd.isna(row.get("quantile")) else float(row["quantile"]),
        "learning_rate": float(row["learning_rate"]),
        "max_iter": int(row["max_iter"]),
        "max_leaf_nodes": int(row["max_leaf_nodes"]),
        "max_depth": None if pd.isna(row["max_depth"]) else int(row["max_depth"]),
        "min_samples_leaf": int(row["min_samples_leaf"]),
        "l2_regularization": float(row["l2_regularization"]),
        "max_features": float(row["max_features"]),
        "max_bins": int(row["max_bins"]),
        "early_stopping": False,
        "random_state": int(row["random_state"]),
    }
    if params["loss"] != "quantile":
        params["quantile"] = None
    return params


def build_family_map(path):
    if not path or not Path(path).exists():
        return {}
    tagged = pd.read_csv(path, encoding="utf-8-sig")
    return dict(tagged[["feature", "family"]].drop_duplicates().itertuples(index=False, name=None))


def selected_columns(columns, drop_families, family_map):
    drop_families = set(drop_families)
    selected = []
    for col in columns:
        family = family_map.get(col, classify_feature(col))
        if family not in drop_families:
            selected.append(col)
    return selected


def evaluate_trial(group, folds, params, policy, selected_by_group):
    rows = []
    pred_rows = []
    capacity = GROUP_CAPACITY_KWH[group]
    for fold in folds:
        train_mask = fold["y_train"].to_numpy(float) >= capacity * policy["min_output_ratio"]
        cols = selected_by_group[group]
        x_train = fold["x_train"].loc[train_mask, cols]
        y_train = fold["y_train"].loc[train_mask]
        if len(y_train) < 500:
            continue

        model = HistGradientBoostingRegressor(**params)
        weights = sample_weight(y_train, group, policy["weight_policy"])
        model.fit(x_train, y_train, sample_weight=weights)
        pred = model.predict(fold["x_val"].loc[:, cols])
        score, nmae, ficr = score_one(fold["y_val"], pred, group)
        rows.append(
            {
                "group": group,
                "pred_year": fold["pred_year"],
                "train_years": fold["train_years"],
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_train": len(y_train),
                "n_val": len(fold["y_val"]),
                "n_features": len(cols),
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(fold["time_val"]).to_numpy(),
                    "pred_year": fold["pred_year"],
                    "train_years": fold["train_years"],
                    "train_policy": "best_histgb",
                    "model": "histgb",
                    "group": group,
                    "actual": fold["y_val"].to_numpy(float),
                    "pred": pred,
                }
            )
        )
    return rows, pred_rows


def summarize_best(best):
    return (
        best.groupby("model_type", as_index=False)
        .agg(
            mean_score=("mean_score", "mean"),
            mean_nmae=("mean_nmae", "mean"),
            mean_ficr=("mean_ficr", "mean"),
            worst_group=("mean_score", "min"),
        )
        .sort_values("mean_score", ascending=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--drop-preset", default="none", choices=sorted(DEFAULT_DROP_FAMILIES))
    parser.add_argument("--drop-families", default="")
    parser.add_argument("--family-csv", default="results/feature_family_v2_tagged_features.csv")
    parser.add_argument("--stem", default="power_histgb_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    drop_families = list(DEFAULT_DROP_FAMILIES[args.drop_preset])
    drop_families.extend(parse_list(args.drop_families))
    drop_families = sorted(set(drop_families))

    cache = prepare_fold_cache(groups)
    family_map = build_family_map(args.family_csv)
    selected_by_group = {}
    for group in groups:
        if not cache.get(group):
            continue
        all_cols = list(cache[group][0]["x_train"].columns)
        selected_by_group[group] = selected_columns(all_cols, drop_families, family_map)
        print(f"{group}: features={len(selected_by_group[group])}/{len(all_cols)} drop={drop_families}")

    rng = np.random.default_rng(args.seed)
    trial_rows = []
    fold_rows = []
    trial_pred_parts = {}
    for group in groups:
        folds = cache[group]
        print(f"\n=== tune histgb {group}: folds={len(folds)} trials={args.trials} ===")
        for trial in range(args.trials):
            params = sample_params(rng, args.seed * 1000 + trial)
            policy = sample_policy(rng)
            rows, pred_rows = evaluate_trial(group, folds, params, policy, selected_by_group)
            if not rows:
                continue
            scores = pd.DataFrame(rows)
            summary = {
                "model_type": "histgb",
                "group": group,
                "trial": trial,
                "mean_score": scores["score"].mean(),
                "mean_nmae": scores["nmae"].mean(),
                "mean_ficr": scores["ficr"].mean(),
                "worst_fold": scores["score"].min(),
                "std_score": scores["score"].std(ddof=0),
                "drop_preset": args.drop_preset,
                "drop_families": ",".join(drop_families),
                **policy,
                **params,
            }
            trial_rows.append(summary)
            for row in rows:
                fold_rows.append({**{"model_type": "histgb", "trial": trial}, **policy, **params, **row})
            trial_pred_parts[(group, trial)] = pred_rows
            print(
                f"{group} trial={trial:03d} score={summary['mean_score']:.5f} "
                f"nmae={summary['mean_nmae']:.5f} ficr={summary['mean_ficr']:.5f} "
                f"loss={params['loss']} q={params['quantile']} policy={policy}"
            )

    trials = pd.DataFrame(trial_rows).sort_values(["group", "mean_score"], ascending=[True, False])
    folds = pd.DataFrame(fold_rows)
    best = trials.groupby("group", as_index=False).head(1).reset_index(drop=True)

    best_pred_parts = []
    for row in best.itertuples():
        for pred_df in trial_pred_parts[(row.group, int(row.trial))]:
            best_pred_parts.append(pred_df)
    predictions = pd.concat(best_pred_parts, ignore_index=True) if best_pred_parts else pd.DataFrame()
    summary = summarize_best(best)

    trials_path = RESULTS_DIR / f"{args.stem}_trials.csv"
    folds_path = RESULTS_DIR / f"{args.stem}_folds.csv"
    best_path = RESULTS_DIR / f"{args.stem}_best.csv"
    pred_path = RESULTS_DIR / f"{args.stem}_best_predictions.csv"
    summary_path = RESULTS_DIR / f"{args.stem}_best_summary.csv"
    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== best ===")
    print(best[["model_type", "group", "trial", "mean_score", "mean_nmae", "mean_ficr", "worst_fold", "loss", "quantile"]].to_string(index=False))
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"saved {trials_path}")
    print(f"saved {pred_path}")


if __name__ == "__main__":
    main()
