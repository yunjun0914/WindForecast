import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS
from utils.tree_feature_profiles import FEATURE_PROFILE_FULL_V2, FEATURE_PROFILES, build_tree_features


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]


def parse_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def log_uniform(rng, low, high):
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def sample_params(rng, trial_seed, search_space):
    if search_space == "focused_l1":
        return {
            "random_state": trial_seed,
            "n_jobs": -1,
            "verbose": -1,
            "objective": "regression_l1",
            "n_estimators": int(rng.integers(550, 2001)),
            "learning_rate": log_uniform(rng, 0.012, 0.055),
            "num_leaves": int(rng.choice([16, 24, 32, 48, 64, 96, 128])),
            "max_depth": int(rng.choice([4, 5, 6, 7, 8, 10, -1])),
            "min_child_samples": int(rng.choice([40, 60, 80, 120, 160, 220, 300])),
            "subsample": float(rng.uniform(0.60, 0.98)),
            "colsample_bytree": float(rng.uniform(0.58, 0.98)),
            "reg_alpha": log_uniform(rng, 1e-4, 5.0),
            "reg_lambda": log_uniform(rng, 1e-3, 20.0),
            "min_split_gain": log_uniform(rng, 1e-5, 0.2),
        }
    if search_space != "broad":
        raise ValueError(f"unknown search_space: {search_space}")
    return {
        "random_state": trial_seed,
        "n_jobs": -1,
        "verbose": -1,
        "objective": rng.choice(["regression", "regression_l1", "huber"]),
        "n_estimators": int(rng.integers(500, 1801)),
        "learning_rate": log_uniform(rng, 0.012, 0.075),
        "num_leaves": int(rng.choice([16, 24, 32, 48, 64, 96, 128])),
        "max_depth": int(rng.choice([-1, 4, 5, 6, 7, 8, 10])),
        "min_child_samples": int(rng.choice([20, 40, 60, 80, 120, 160, 220, 300])),
        "subsample": float(rng.uniform(0.60, 1.00)),
        "colsample_bytree": float(rng.uniform(0.55, 1.00)),
        "reg_alpha": log_uniform(rng, 1e-4, 5.0),
        "reg_lambda": log_uniform(rng, 1e-3, 50.0),
        "min_split_gain": log_uniform(rng, 1e-5, 1.0),
    }


def sample_policy(rng, search_space):
    if search_space == "focused_l1":
        return {
            "min_output_ratio": float(rng.choice([0.03, 0.05, 0.10, 0.15])),
            "weight_policy": rng.choice(["none", "metric_x2", "actual_sqrt"]),
        }
    return {
        "min_output_ratio": float(rng.choice([0.00, 0.03, 0.05, 0.10, 0.15])),
        "weight_policy": rng.choice(["none", "metric_x2", "actual_sqrt"]),
    }


def filter_scada_years(scada, years):
    out = scada.copy()
    out["kst_dtm"] = pd.to_datetime(out["kst_dtm"])
    return out[out["kst_dtm"].dt.year.isin(years)].reset_index(drop=True)


def filter_weather_years(weather, years):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out[out["forecast_kst_dtm"].dt.year.isin(years)].reset_index(drop=True)


def build_group_table(weather, labels, group):
    labels = labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    table = weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    table = table.dropna(subset=[group]).reset_index(drop=True)
    table["year"] = pd.to_datetime(table["forecast_kst_dtm"]).dt.year
    return table.rename(columns={group: "target"})


def feature_columns(weather):
    return [col for col in weather.columns if col not in TIME_KEY_COLS]


def score_one(y_true, pred, group):
    capacity = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    nmae, ficr = group_nmae_ficr(y_true, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def sample_weight(y, group, weight_policy):
    if weight_policy == "none":
        return None
    capacity = GROUP_CAPACITY_KWH[group]
    y = np.asarray(y, dtype=float)
    if weight_policy == "metric_x2":
        return 1.0 + 2.0 * (y >= capacity * 0.10)
    if weight_policy == "actual_sqrt":
        return 0.5 + np.sqrt(np.clip(y / capacity, 0, 1))
    raise ValueError(f"unknown weight_policy: {weight_policy}")


def prepare_fold_cache(groups, feature_profile=FEATURE_PROFILE_FULL_V2):
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    feature_cache = {}
    for group in groups:
        print(f"build tree features {group}: profile={feature_profile}")
        feature_cache[group] = build_tree_features(ldaps, gfs, group, feature_profile=feature_profile)

    cache = {}
    for group in groups:
        cache[group] = []
        for pred_year in YEARS:
            train_years = [year for year in YEARS if year != pred_year]
            scada = filter_scada_years(scada_by_group[group], train_years)
            if len(scada) == 0:
                continue
            train_weather_base = filter_weather_years(feature_cache[group], train_years)
            val_weather_base = filter_weather_years(feature_cache[group], [pred_year])
            train_weather, val_weather = add_power_curve_feature_oof(
                train_weather_base,
                val_weather_base,
                scada,
                group,
                HUB_HEIGHT_PROXY_COL,
                GROUP_N_TURBINES[group],
            )
            weather = pd.concat([train_weather, val_weather], ignore_index=True)
            table = build_group_table(weather, labels, group)
            train = table[table["year"].isin(train_years)].copy()
            val = table[table["year"] == pred_year].copy()
            if len(train) < 1000 or len(val) < 200:
                continue
            cols = feature_columns(weather)
            cache[group].append(
                {
                    "pred_year": pred_year,
                    "train_years": ",".join(map(str, train_years)),
                    "x_train": train[cols],
                    "y_train": train["target"],
                    "time_train": train["forecast_kst_dtm"],
                    "x_val": val[cols],
                    "y_val": val["target"],
                    "time_val": val["forecast_kst_dtm"],
                }
            )
    return cache


def evaluate_trial(group, folds, params, policy):
    rows = []
    capacity = GROUP_CAPACITY_KWH[group]
    for fold in folds:
        train_mask = fold["y_train"].to_numpy(float) >= capacity * policy["min_output_ratio"]
        x_train = fold["x_train"].loc[train_mask]
        y_train = fold["y_train"].loc[train_mask]
        if len(y_train) < 500:
            continue
        model = LGBMRegressor(**params)
        weights = sample_weight(y_train, group, policy["weight_policy"])
        model.fit(x_train, y_train, sample_weight=weights)
        pred = model.predict(fold["x_val"])
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
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-space", default="broad", choices=["broad", "focused_l1"])
    parser.add_argument("--stem", default="power_lgbm_hyperparams")
    parser.add_argument("--feature-profile", default=FEATURE_PROFILE_FULL_V2, choices=FEATURE_PROFILES)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    cache = prepare_fold_cache(groups, feature_profile=args.feature_profile)
    rng = np.random.default_rng(args.seed)

    trial_rows = []
    fold_rows = []
    for group in groups:
        folds = cache[group]
        print(f"\n=== tune {group}: folds={len(folds)} trials={args.trials} ===")
        for trial in range(args.trials):
            params = sample_params(rng, args.seed * 1000 + trial, args.search_space)
            policy = sample_policy(rng, args.search_space)
            rows = evaluate_trial(group, folds, params, policy)
            if not rows:
                continue
            scores = pd.DataFrame(rows)
            summary = {
                "group": group,
                "trial": trial,
                "mean_score": scores["score"].mean(),
                "mean_nmae": scores["nmae"].mean(),
                "mean_ficr": scores["ficr"].mean(),
                "worst_fold": scores["score"].min(),
                "std_score": scores["score"].std(ddof=0),
                **policy,
                **params,
            }
            trial_rows.append(summary)
            for row in rows:
                fold_rows.append({**{"trial": trial}, **policy, **params, **row})
            print(
                f"{group} trial={trial:03d} score={summary['mean_score']:.5f} "
                f"ficr={summary['mean_ficr']:.5f} worst={summary['worst_fold']:.5f} "
                f"policy={policy}"
            )

    trials = pd.DataFrame(trial_rows).sort_values(["group", "mean_score"], ascending=[True, False])
    folds = pd.DataFrame(fold_rows)
    best = trials.groupby("group", as_index=False).head(1).reset_index(drop=True)

    trials_path = RESULTS_DIR / f"{args.stem}_trials.csv"
    folds_path = RESULTS_DIR / f"{args.stem}_folds.csv"
    best_path = RESULTS_DIR / f"{args.stem}_best.csv"
    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    print("\n=== best ===")
    print(best[["group", "trial", "mean_score", "mean_nmae", "mean_ficr", "worst_fold", "std_score"]].to_string(index=False))
    print(f"saved {trials_path}")
    print(f"saved {folds_path}")
    print(f"saved {best_path}")
    return trials, folds, best


if __name__ == "__main__":
    main()
