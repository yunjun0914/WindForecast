import numpy as np
import pandas as pd

from evaluate_group3_pinn_teacher_transfer import build_teacher_weather_variants, train_single_group
from evaluate_group3_transfer_blend import group2_transfer_tree_candidates
from evaluate_pinn_tree_blend import build_pinn_predictions, build_tree_predictions, score_one
from utils.metrics import GROUP_CAPACITY_KWH

RESULTS_PATH = "results/final_tree_ensemble_scores.csv"
PREDICTIONS_PATH = "results/final_tree_ensemble_predictions.csv"
GROUPS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]


def score_groups(pred_by_group, y_by_group, capacity_by_group):
    rows = []
    scores = []
    for group in GROUPS:
        score, nmae, ficr = score_one(y_by_group[group], pred_by_group[group], capacity_by_group[group])
        rows.append({"group": group, "score": score, "nmae": nmae, "ficr": ficr})
        scores.append(score)
    rows.append({"group": "mean", "score": float(np.mean(scores)), "nmae": np.nan, "ficr": np.nan})
    return rows


def build_group3_mixed_teacher_pinn():
    variants, labels = build_teacher_weather_variants()
    result = train_single_group("unison", variants["mix_30_unison_g3_vestas_g2"], labels, seed=42, verbose=False)
    return {
        "time": result["val_time"],
        "y": result["y"],
        "pred": result["pred"],
    }


def main():
    tree = build_tree_predictions()
    pinn = build_pinn_predictions()
    group3_transfer = group2_transfer_tree_candidates()
    group3_mixed_pinn = build_group3_mixed_teacher_pinn()

    times = {}
    y_by_group = {}
    capacity_by_group = {}
    current_best = {}
    tree_candidates = {}

    for group in ["kpx_group_1", "kpx_group_2"]:
        times[group] = tree[group]["time"].reset_index(drop=True)
        if not times[group].equals(pinn[group]["time"].reset_index(drop=True)):
            raise ValueError(f"time mismatch for {group}")
        capacity = GROUP_CAPACITY_KWH[group]
        y_by_group[group] = tree[group]["y"]
        capacity_by_group[group] = capacity
        current_best[group] = np.clip(0.70 * pinn[group]["pred"] + 0.30 * tree[group]["calibrated"], 0, capacity)
        tree_candidates[group] = {
            "own_tree_raw": tree[group]["raw"],
            "own_tree_calibrated": tree[group]["calibrated"],
        }

    group = "kpx_group_3"
    times[group] = tree[group]["time"].reset_index(drop=True)
    if not times[group].equals(group3_mixed_pinn["time"].reset_index(drop=True)):
        raise ValueError("group3 mixed PINN time mismatch")
    transfer = group3_transfer["tree_group2_transfer_group3_features"]
    if not times[group].equals(transfer["time"].reset_index(drop=True)):
        raise ValueError("group3 transfer tree time mismatch")

    capacity = GROUP_CAPACITY_KWH[group]
    y_by_group[group] = tree[group]["y"]
    capacity_by_group[group] = capacity
    current_best[group] = np.clip(0.70 * group3_mixed_pinn["pred"] + 0.30 * transfer["pred"], 0, capacity)
    tree_candidates[group] = {
        "own_tree_raw": tree[group]["raw"],
        "own_tree_calibrated": tree[group]["calibrated"],
        "group2_tree_transfer_features": transfer["pred"],
        "group2_tree_same_time_proxy": group3_transfer["tree_group2_proxy_same_time"]["pred"],
    }

    rows = []
    for row in score_groups(current_best, y_by_group, capacity_by_group):
        rows.append({"experiment": "current_best", "group": row["group"], "tree_candidate": "none", "base_weight": 1.0, **row})

    weights = np.linspace(0, 1, 21)  # weight on current best; 1.0 means no extra tree.

    # Global blend with each group's own calibrated/raw tree.
    for candidate_name in ["own_tree_raw", "own_tree_calibrated"]:
        for w in weights:
            pred = {}
            for group in GROUPS:
                capacity = capacity_by_group[group]
                pred[group] = np.clip(w * current_best[group] + (1 - w) * tree_candidates[group][candidate_name], 0, capacity)
            for row in score_groups(pred, y_by_group, capacity_by_group):
                rows.append(
                    {
                        "experiment": f"global_blend_{candidate_name}",
                        "group": row["group"],
                        "tree_candidate": candidate_name,
                        "base_weight": w,
                        **row,
                    }
                )

    # Group-wise one-at-a-time blend: does any group still want more tree?
    for group in GROUPS:
        for candidate_name, candidate_pred in tree_candidates[group].items():
            for w in weights:
                pred = dict(current_best)
                capacity = capacity_by_group[group]
                pred[group] = np.clip(w * current_best[group] + (1 - w) * candidate_pred, 0, capacity)
                score, nmae, ficr = score_one(y_by_group[group], pred[group], capacity)
                mean_score = np.mean(
                    [
                        score if g == group else score_one(y_by_group[g], pred[g], capacity_by_group[g])[0]
                        for g in GROUPS
                    ]
                )
                rows.append(
                    {
                        "experiment": f"groupwise_extra_tree_{group}",
                        "group": group,
                        "tree_candidate": candidate_name,
                        "base_weight": w,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                    }
                )
                rows.append(
                    {
                        "experiment": f"groupwise_extra_tree_{group}",
                        "group": "mean",
                        "tree_candidate": candidate_name,
                        "base_weight": w,
                        "score": float(mean_score),
                        "nmae": np.nan,
                        "ficr": np.nan,
                    }
                )

    results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    pred_out = pd.DataFrame({"forecast_kst_dtm": times["kpx_group_1"]})
    for group in GROUPS:
        pred_out[group] = current_best[group]
    pred_out.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    print("=== Best mean rows ===")
    print(results[results["group"] == "mean"].head(25).to_string(index=False))
    print("\n=== Best group rows ===")
    print(results[results["group"] != "mean"].groupby("group").head(8).to_string(index=False))
    return results


if __name__ == "__main__":
    main()
