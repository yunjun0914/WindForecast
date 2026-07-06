from collections import defaultdict

import pandas as pd
import joblib

from utils.preprocessing import build_weather_features, build_group_dataset, HUB_HEIGHT_PROXY_COL
from utils.metrics import group_nmae_ficr, total_score, GROUP_CAPACITY_KWH
from utils.power_curve import fit_group_power_curve, add_power_curve_feature, GROUP_N_TURBINES
from models import random_forest, lgbm, xgb

MODEL_TRAINERS = {
    "random_forest": random_forest.train,
    "lgbm": lgbm.train,
    "xgb": xgb.train,
}

ARTIFACT_DIR = "models/artifacts"
RESULTS_PATH = "results/nmae_results.csv"
SCORE_PATH = "results/total_score.csv"


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    results = []
    # per model_name: list of (nmae, ficr) across the 3 groups, for the official total_score
    scores_by_model = defaultdict(list)

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        X, y = build_group_dataset(group_weather, labels, group)

        val_preds = {}
        y_val_ref = None
        for model_name, train_fn in MODEL_TRAINERS.items():
            model, X_val, y_val = train_fn(X, y)
            pred_val = model.predict(X_val)
            nmae, ficr = group_nmae_ficr(y_val, pred_val, capacity)
            results.append({"group": group, "model": model_name, "nmae": nmae, "ficr": ficr})
            scores_by_model[model_name].append((nmae, ficr))

            val_preds[model_name] = pred_val
            y_val_ref = y_val
            joblib.dump(model, f"{ARTIFACT_DIR}/{group}_{model_name}.pkl")

        # simple average across the 3 models (train() uses the same random_state,
        # so X_val/y_val are identical across models for a given group)
        ensemble_pred = sum(val_preds.values()) / len(val_preds)
        nmae, ficr = group_nmae_ficr(y_val_ref, ensemble_pred, capacity)
        results.append({"group": group, "model": "ensemble", "nmae": nmae, "ficr": ficr})
        scores_by_model["ensemble"].append((nmae, ficr))

    results_df = pd.DataFrame(results)
    print(results_df.pivot(index="group", columns="model", values=["nmae", "ficr"]).round(4))
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    score_rows = []
    for model_name, group_scores in scores_by_model.items():
        nmaes, ficrs = zip(*group_scores)
        score, one_minus_nmae, ficr = total_score(nmaes, ficrs)
        score_rows.append(
            {"model": model_name, "total_score": score, "one_minus_nmae": one_minus_nmae, "ficr": ficr}
        )
    score_df = pd.DataFrame(score_rows).sort_values("total_score", ascending=False)
    print()
    print(score_df.round(4).to_string(index=False))
    score_df.to_csv(SCORE_PATH, index=False, encoding="utf-8-sig")

    return results_df, score_df


if __name__ == "__main__":
    main()
