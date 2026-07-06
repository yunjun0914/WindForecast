import time

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, cross_val_score
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr, make_group_scorer
from utils.preprocessing import build_group_dataset, build_weather_features

TUNED_ARTIFACT_DIR = "models/artifacts/tuned"
RESULTS_PATH = "results/tuning_results.csv"
SCORE_COMPARISON_PATH = "results/tuning_score_comparison.csv"

N_ITER = 15
CV = 3
RANDOM_STATE = 42
# explicit shuffle=True: sklearn's default cv=int uses a non-shuffled KFold,
# which would validate on contiguous time blocks instead of a random split
CV_SPLITTER = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)

# n_jobs=-1 on each estimator (parallel tree building within a single fit) and
# n_jobs=1 on the CV wrappers (sequential fits). Measured on this machine:
# nesting CV-level (-1) + estimator-level (-1) parallelism causes heavy
# process-spawn contention on Windows -- 15 RF fits took 460s that way vs 140s
# with parallelism kept at the estimator level only.
DEFAULT_ESTIMATORS = {
    "random_forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
}

PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5, 1.0],
    },
    "lgbm": {
        "n_estimators": [100, 200, 300, 500],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 5, 10, 15],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "xgb": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
    },
}


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    tuning_rows = []
    comparison_rows = []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        X, y = build_group_dataset(weather, labels, group)
        scorer = make_group_scorer(capacity)

        oof_preds = {}
        for model_name, base_estimator in DEFAULT_ESTIMATORS.items():
            # untuned baseline, scored with the exact same CV protocol for a fair comparison
            default_cv_score = cross_val_score(
                clone(base_estimator), X, y, scoring=scorer, cv=CV_SPLITTER, n_jobs=1
            ).mean()

            t0 = time.time()
            search = RandomizedSearchCV(
                estimator=clone(base_estimator),
                param_distributions=PARAM_DISTRIBUTIONS[model_name],
                n_iter=N_ITER,
                scoring=scorer,
                cv=CV_SPLITTER,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
            )
            search.fit(X, y)
            elapsed = time.time() - t0

            joblib.dump(search.best_estimator_, f"{TUNED_ARTIFACT_DIR}/{group}_{model_name}.pkl")
            tuning_rows.append(
                {
                    "group": group,
                    "model": model_name,
                    "default_cv_score": default_cv_score,
                    "tuned_cv_score": search.best_score_,
                    "best_params": search.best_params_,
                    "seconds": round(elapsed, 1),
                }
            )
            print(
                f"{group}/{model_name}: default={default_cv_score:.4f} tuned={search.best_score_:.4f} "
                f"({elapsed:.0f}s) {search.best_params_}"
            )

            # out-of-fold predictions from the TUNED hyperparameters (same splitter),
            # so we can check whether averaging the 3 tuned models still helps
            oof_estimator = clone(base_estimator).set_params(**search.best_params_)
            oof_preds[model_name] = cross_val_predict(oof_estimator, X, y, cv=CV_SPLITTER, n_jobs=1)

        ensemble_oof = sum(oof_preds.values()) / len(oof_preds)
        for model_name, pred in {**oof_preds, "ensemble": ensemble_oof}.items():
            nmae, ficr = group_nmae_ficr(y, pred, capacity)
            comparison_rows.append(
                {"group": group, "model": model_name, "oof_score": 0.5 * (1 - nmae) + 0.5 * ficr}
            )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    comparison_df = pd.DataFrame(comparison_rows)
    print()
    print(comparison_df.pivot(index="group", columns="model", values="oof_score").round(4))
    comparison_df.to_csv(SCORE_COMPARISON_PATH, index=False, encoding="utf-8-sig")

    return tuning_df, comparison_df


if __name__ == "__main__":
    main()
