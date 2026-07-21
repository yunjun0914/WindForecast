from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


KEYS = ["forecast_kst_dtm", "pred_year", "group"]
LOSS_COMBINATIONS = {
    "hybrid": ("pinn_mixed", "tcn_pure"),
    "all_pure": ("pinn_pure", "tcn_pure"),
    "all_mixed": ("pinn_mixed", "tcn_mixed"),
    "reverse": ("pinn_pure", "tcn_mixed"),
}
FIXED_WEIGHTS = np.asarray([0.50, 0.05, 0.45], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pinn-oof",
        type=Path,
        default=Path("results/group_pinn_direct_band_loss_v1_oof.csv"),
    )
    parser.add_argument(
        "--tcn-oof",
        type=Path,
        default=Path("results/group_tcn_direct_band_loss_v1_oof.csv"),
    )
    parser.add_argument(
        "--tree-oof",
        type=Path,
        default=Path("results/group_base_recovery_v1_oof.csv"),
    )
    parser.add_argument("--tree-column", default="tree_common72_l1")
    parser.add_argument("--weight-step", type=float, default=0.025)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_three_branch_loss_ensemble_v1")
    return parser.parse_args()


def read_oof(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in [*KEYS, "actual", *columns] if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[[*KEYS, "actual", *columns]].copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} has duplicate OOF keys")
    return frame


def align_predictions(args: argparse.Namespace) -> pd.DataFrame:
    pinn = read_oof(args.pinn_oof, ["pure_band_ficr", "ficr_nmae"]).rename(
        columns={
            "actual": "actual_pinn",
            "pure_band_ficr": "pinn_pure",
            "ficr_nmae": "pinn_mixed",
        }
    )
    tcn = read_oof(args.tcn_oof, ["pure_band_ficr", "ficr_nmae"]).rename(
        columns={
            "actual": "actual_tcn",
            "pure_band_ficr": "tcn_pure",
            "ficr_nmae": "tcn_mixed",
        }
    )
    tree = read_oof(args.tree_oof, [args.tree_column]).rename(
        columns={"actual": "actual_tree", args.tree_column: "tree"}
    )
    aligned = pinn.merge(tcn, on=KEYS, how="inner", validate="one_to_one").merge(
        tree, on=KEYS, how="inner", validate="one_to_one"
    )
    expected = (len(pinn), len(tcn), len(tree))
    if len(aligned) != min(expected) or len(set(expected)) != 1:
        raise ValueError(f"OOF row counts do not align: sources={expected}, merged={len(aligned)}")
    actuals = aligned[["actual_pinn", "actual_tcn", "actual_tree"]].to_numpy(float)
    if not np.allclose(actuals, actuals[:, :1], rtol=0.0, atol=1e-4):
        raise ValueError("OOF targets differ across PINN, TCN, and TREE")
    return aligned.rename(columns={"actual_pinn": "actual"}).drop(
        columns=["actual_tcn", "actual_tree"]
    )


def weight_grid(step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    if units <= 0 or not np.isclose(units * step, 1.0):
        raise ValueError("weight-step must divide 1.0 exactly")
    return [
        np.asarray([pinn / units, tree / units, (units - pinn - tree) / units])
        for pinn in range(units + 1)
        for tree in range(units - pinn + 1)
    ]


def prediction_matrix(frame: pd.DataFrame, combination: str) -> np.ndarray:
    pinn_column, tcn_column = LOSS_COMBINATIONS[combination]
    return frame[[pinn_column, "tree", tcn_column]].to_numpy(float)


def score_prediction(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    group_scores = []
    group_nmaes = []
    group_ficrs = []
    fold_scores = []
    for group in TARGET_COLS:
        group_mask = frame["group"].eq(group).to_numpy()
        if not bool(group_mask.any()):
            continue
        capacity = GROUP_CAPACITY_KWH[group]
        nmae, ficr = group_nmae_ficr(
            frame.loc[group_mask, "actual"], prediction[group_mask], capacity
        )
        group_nmaes.append(nmae)
        group_ficrs.append(ficr)
        group_scores.append(0.5 * (1.0 - nmae) + 0.5 * ficr)
        for year in sorted(frame.loc[group_mask, "pred_year"].unique()):
            fold_mask = group_mask & frame["pred_year"].eq(year).to_numpy()
            fold_nmae, fold_ficr = group_nmae_ficr(
                frame.loc[fold_mask, "actual"], prediction[fold_mask], capacity
            )
            fold_scores.append(0.5 * (1.0 - fold_nmae) + 0.5 * fold_ficr)
    if not group_scores:
        raise ValueError("No groups available for scoring")
    mean_nmae = float(np.mean(group_nmaes))
    mean_ficr = float(np.mean(group_ficrs))
    return {
        "mean_nmae": mean_nmae,
        "mean_ficr": mean_ficr,
        "mean_score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
        "worst_group_score": float(np.min(group_scores)),
        "std_group_score": float(np.std(group_scores)),
        "worst_fold": float(np.min(fold_scores)),
        "std_fold": float(np.std(fold_scores)),
    }


def evaluate_weights(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    metrics = score_prediction(frame, matrix @ weights)
    return {
        "pinn_weight": float(weights[0]),
        "tree_weight": float(weights[1]),
        "tcn_weight": float(weights[2]),
        **metrics,
    }


def best_grid_row(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["mean_score"],
            row["worst_fold"],
            -row["std_fold"],
        ),
    )


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    aligned = align_predictions(args)
    grid = weight_grid(float(args.weight_step))
    years = sorted(aligned["pred_year"].unique().tolist())
    summary_rows = []
    diagnostic_rows = []
    selected_predictions = aligned[[*KEYS, "actual"]].copy()

    for combination in LOSS_COMBINATIONS:
        matrix = prediction_matrix(aligned, combination)
        fixed = evaluate_weights(aligned, matrix, FIXED_WEIGHTS)
        summary_rows.append(
            {"combination": combination, "selection": "fixed_50_05_45", **fixed}
        )

        full_rows = [evaluate_weights(aligned, matrix, weights) for weights in grid]
        for row in full_rows:
            diagnostic_rows.append(
                {"combination": combination, "selection": "full_oof_grid", **row}
            )
        full_best = best_grid_row(full_rows)
        summary_rows.append(
            {"combination": combination, "selection": "full_oof_best", **full_best}
        )

        crossfit_prediction = np.full(len(aligned), np.nan, dtype=float)
        selected_weights = []
        for heldout_year in years:
            train_mask = aligned["pred_year"].ne(heldout_year).to_numpy()
            validation_mask = ~train_mask
            train = aligned.loc[train_mask].reset_index(drop=True)
            train_matrix = matrix[train_mask]
            candidate_rows = [
                evaluate_weights(train, train_matrix, weights) for weights in grid
            ]
            selected = best_grid_row(candidate_rows)
            weights = np.asarray(
                [
                    selected["pinn_weight"],
                    selected["tree_weight"],
                    selected["tcn_weight"],
                ]
            )
            selected_weights.append(weights)
            crossfit_prediction[validation_mask] = matrix[validation_mask] @ weights
            diagnostic_rows.append(
                {
                    "combination": combination,
                    "selection": "crossfit_selected",
                    "heldout_year": heldout_year,
                    **selected,
                }
            )
        if not np.isfinite(crossfit_prediction).all():
            raise RuntimeError(f"Cross-fit predictions are incomplete for {combination}")
        crossfit_metrics = score_prediction(aligned, crossfit_prediction)
        summary_rows.append(
            {
                "combination": combination,
                "selection": "crossfit_prediction",
                "pinn_weight": np.nan,
                "tree_weight": np.nan,
                "tcn_weight": np.nan,
                **crossfit_metrics,
            }
        )
        selected_predictions[f"crossfit_{combination}"] = crossfit_prediction

        median_weights = np.median(np.stack(selected_weights), axis=0)
        median_weights /= median_weights.sum()
        median = evaluate_weights(aligned, matrix, median_weights)
        summary_rows.append(
            {
                "combination": combination,
                "selection": "crossfit_median_weights",
                **median,
            }
        )
        selected_predictions[f"median_{combination}"] = matrix @ median_weights

    summary = pd.DataFrame(summary_rows).sort_values(
        ["selection", "mean_score"], ascending=[True, False]
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["combination", "selection", "mean_score"], ascending=[True, True, False]
    )
    oof_path = args.results_dir / f"{args.stem}_oof.csv"
    summary_path = args.results_dir / f"{args.stem}_summary.csv"
    diagnostics_path = args.results_dir / f"{args.stem}_weight_diagnostics.csv"
    selected_predictions.to_csv(oof_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    print("=== three-branch ensemble summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved {oof_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
