import ast
import itertools

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.preprocessing import build_group_dataset, build_weather_features

RANDOM_STATE = 42
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
TUNING_RESULTS_PATH = "results/tuning_results.csv"
RESULTS_PATH = "results/ensemble_comparison.csv"

# RF keeps its default hyperparameters (tuning made it worse); LGBM/XGB use their tuned params
BASE_ESTIMATORS = {
    "random_forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
}
TUNE_MODELS = {"lgbm", "xgb"}


def main():
    tuning_results = pd.read_csv(TUNING_RESULTS_PATH, encoding="utf-8-sig")

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    rows = []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        X, y = build_group_dataset(weather, labels, group)

        oof_preds = {}
        for model_name, base_estimator in BASE_ESTIMATORS.items():
            estimator = clone(base_estimator)
            if model_name in TUNE_MODELS:
                params_str = tuning_results.loc[
                    (tuning_results["group"] == group) & (tuning_results["model"] == model_name), "best_params"
                ].iloc[0]
                estimator.set_params(**ast.literal_eval(params_str))
            oof_preds[model_name] = cross_val_predict(estimator, X, y, cv=CV_SPLITTER, n_jobs=1)

        model_names = list(oof_preds)
        for r in range(1, len(model_names) + 1):
            for combo in itertools.combinations(model_names, r):
                pred = sum(oof_preds[m] for m in combo) / len(combo)
                nmae, ficr = group_nmae_ficr(y, pred, capacity)
                rows.append(
                    {
                        "group": group,
                        "combo": "+".join(combo),
                        "score": 0.5 * (1 - nmae) + 0.5 * ficr,
                    }
                )

    results_df = pd.DataFrame(rows)
    pivot = results_df.pivot(index="combo", columns="group", values="score").round(4)
    print(pivot)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    return results_df


if __name__ == "__main__":
    main()
