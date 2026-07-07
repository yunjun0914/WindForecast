import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor

import _bootstrap  # noqa: F401
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS


RESULTS_DIR = Path("results")
GROUPS = TARGET_COLS
YEARS = [2022, 2023, 2024]


def fit_lgbm_tuned(seed=42):
    return LGBMRegressor(
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
    )


def fit_lgbm_regularized(seed=42):
    return LGBMRegressor(
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=32,
        min_child_samples=150,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=4.0,
    )


MODEL_FACTORIES = {
    "lgbm_tuned": fit_lgbm_tuned,
    "lgbm_regularized": fit_lgbm_regularized,
    "xgb_compact": lambda seed=42: XGBRegressor(
        random_state=seed,
        n_jobs=-1,
        n_estimators=700,
        learning_rate=0.035,
        max_depth=5,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="reg:squarederror",
    ),
    "rf_compact": lambda seed=42: RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.70,
    ),
    "extra_compact": lambda seed=42: ExtraTreesRegressor(
        random_state=seed,
        n_jobs=-1,
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.70,
    ),
}


def parse_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def score_one(y_true, pred, group):
    capacity = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    nmae, ficr = group_nmae_ficr(y_true, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


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


def train_mask_for_policy(table, group, policy):
    if policy == "all":
        return np.ones(len(table), dtype=bool)
    if policy == "metric_valid":
        return table["target"].to_numpy(float) >= GROUP_CAPACITY_KWH[group] * 0.10
    raise ValueError(f"unknown train policy: {policy}")


def model_predict(models, x_train, y_train, x_val):
    preds = {}
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        preds[name] = fitted.predict(x_val)
    if len(preds) > 1:
        preds["model_mean"] = np.mean(np.column_stack(list(preds.values())), axis=1)
    return preds


def add_mean_rows(results):
    means = (
        results.groupby(["pred_year", "train_years", "train_policy", "model"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "fold_mean"
    means["n_train"] = np.nan
    means["n_used"] = np.nan
    return pd.concat([results, means[results.columns]], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="lgbm_tuned,lgbm_regularized")
    parser.add_argument("--train-policies", default="all,metric_valid")
    parser.add_argument("--stem", default="tree_compact_v2_multi_year_models")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = parse_list(args.models)
    train_policies = parse_list(args.train_policies)
    models = {name: MODEL_FACTORIES[name]() for name in model_names}

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
    for group in GROUPS:
        print(f"build all_meteo_compact_v2 {group}")
        feature_cache[group] = build_all_meteo_compact_v2(ldaps, gfs, group)

    rows = []
    pred_rows = []
    for pred_year in YEARS:
        train_years = [year for year in YEARS if year != pred_year]
        train_years_text = ",".join(map(str, train_years))
        print(f"\n=== pred_year={pred_year}, train_years={train_years_text} ===")
        for group in GROUPS:
            scada = filter_scada_years(scada_by_group[group], train_years)
            if len(scada) == 0:
                print(f"{group}: skip no scada")
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
            train_base = table[table["year"].isin(train_years)].copy()
            val = table[table["year"] == pred_year].copy()
            if len(train_base) < 1000 or len(val) < 200:
                print(f"{group}: skip train={len(train_base)}, val={len(val)}")
                continue
            cols = feature_columns(weather)
            for policy in train_policies:
                policy_mask = train_mask_for_policy(train_base, group, policy)
                train = train_base[policy_mask].copy()
                preds = model_predict(models, train[cols], train["target"], val[cols])
                for model_name, pred in preds.items():
                    score, nmae, ficr = score_one(val["target"], pred, group)
                    rows.append(
                        {
                            "pred_year": pred_year,
                            "train_years": train_years_text,
                            "train_policy": policy,
                            "model": model_name,
                            "group": group,
                            "score": score,
                            "nmae": nmae,
                            "ficr": ficr,
                            "n": len(val),
                            "n_train": len(train_base),
                            "n_used": len(train),
                        }
                    )
                    pred_rows.append(
                        pd.DataFrame(
                            {
                                "forecast_kst_dtm": val["forecast_kst_dtm"].to_numpy(),
                                "pred_year": pred_year,
                                "train_years": train_years_text,
                                "train_policy": policy,
                                "model": model_name,
                                "group": group,
                                "actual": val["target"].to_numpy(float),
                                "pred": np.clip(pred, 0, GROUP_CAPACITY_KWH[group]),
                            }
                        )
                    )
                    print(f"{group} {policy} {model_name}: score={score:.4f}, nmae={nmae:.4f}, ficr={ficr:.4f}")

    results = add_mean_rows(pd.DataFrame(rows))
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = (
        results[results["group"] == "fold_mean"]
        .groupby(["train_policy", "model"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_folds=("score", "count"),
        )
        .sort_values("mean_score", ascending=False)
    )

    scores_path = RESULTS_DIR / f"{args.stem}_scores.csv"
    pred_path = RESULTS_DIR / f"{args.stem}_predictions.csv"
    summary_path = RESULTS_DIR / f"{args.stem}_summary.csv"
    results.to_csv(scores_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"saved {scores_path}")
    print(f"saved {pred_path}")
    print(f"saved {summary_path}")
    return results, predictions, summary


if __name__ == "__main__":
    main()
