from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401
from experiments.evaluate_group_base_recovery_oof import (
    CausalWindowDataset,
    FoldData,
    build_site_static,
    clean_matrix,
    fold_metric,
    make_loader,
    make_model,
    predict_tcn,
    prepare_folds,
    score_predictions,
    set_seed,
    target_arrays,
    tcn_prediction_frame,
)
from experiments.evaluate_group_tcn_band_loss_oof import train_direct_epoch
from experiments.evaluate_group_unified_oof import build_group_tables, read_inputs
from utils.metrics import TARGET_COLS
from utils.nwp_augmentation import perturb_nwp_issues
from utils.seq_dataset import SequenceStandardScaler


VARIANTS = ("repeat_control", "coherent_nwp")
KEYS = ["forecast_kst_dtm", "pred_year", "group"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_tcn_nwp_augmentation_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_tcn_nwp_augmentation_v1")
    parser.add_argument("--baseline-oof", type=Path)
    parser.add_argument(
        "--loss-mode", choices=("pure_band_ficr", "ficr_nmae")
    )
    parser.add_argument("--fixed-epoch", type=int)
    parser.add_argument("--baseline-column")
    parser.add_argument("--pinn-oof", type=Path)
    parser.add_argument("--tree-oof", type=Path)
    parser.add_argument("--pinn-column", default="ficr_nmae")
    parser.add_argument("--tree-column", default="tree_common72_l1")
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs=3,
        metavar=("PINN", "TREE", "TCN"),
        default=(13.0 / 39.0, 2.0 / 39.0, 24.0 / 39.0),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-outer-folds", type=int, default=0)
    return parser.parse_args()


def load_configs(path: Path, smoke_test: bool) -> tuple[dict, dict]:
    experiment = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(
        Path(experiment["base_config_path"]).read_text(encoding="utf-8")
    )
    if smoke_test:
        experiment["fixed_epoch"] = 1
        base["tcn"]["batch_size"] = 128
    return experiment, base


def build_folds(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    labels: pd.DataFrame,
    scada_by_group: dict[str, pd.DataFrame],
    max_outer_folds: int,
    label: str,
) -> list[FoldData]:
    print(f"build {label} site inputs", flush=True)
    site_static = build_site_static(ldaps, gfs)
    tables_by_group = {}
    for group in TARGET_COLS:
        print(f"build {label} group inputs: {group}", flush=True)
        tables_by_group[group] = build_group_tables(
            ldaps,
            gfs,
            labels,
            scada_by_group[group],
            group,
            include_residual=False,
        )
    folds = prepare_folds(labels, tables_by_group, site_static, max_outer_folds)
    del site_static, tables_by_group
    gc.collect()
    return folds


def validate_fold_pair(original: FoldData, augmented: FoldData) -> None:
    if original.pred_year != augmented.pred_year:
        raise ValueError("Original and augmented folds use different validation years")
    if original.tcn_features != augmented.tcn_features:
        raise ValueError("Original and augmented TCN feature contracts differ")
    for split in ("train", "validation"):
        left = getattr(original, split)
        right = getattr(augmented, split)
        if not pd.to_datetime(left["forecast_kst_dtm"]).equals(
            pd.to_datetime(right["forecast_kst_dtm"])
        ):
            raise ValueError(f"Original and augmented {split} timestamps differ")
        if not np.allclose(
            left[TARGET_COLS].to_numpy(float),
            right[TARGET_COLS].to_numpy(float),
            equal_nan=True,
        ):
            raise ValueError(f"Augmentation changed {split} targets")


def make_sequence_dataset(
    frame: pd.DataFrame,
    features: np.ndarray,
    base_config: dict,
    keep_observed: bool,
) -> CausalWindowDataset:
    _, target, weight, observed = target_arrays(
        frame, float(base_config["min_output_ratio"])
    )
    return CausalWindowDataset(
        features,
        target,
        weight,
        observed,
        frame["forecast_kst_dtm"],
        int(base_config["tcn"]["window"]),
        keep_observed=keep_observed,
    )


def prepare_variant_data(
    original: FoldData,
    augmented: FoldData,
    base_config: dict,
    variant: str,
):
    original_raw = clean_matrix(original.train, original.tcn_features)
    validation_raw = clean_matrix(original.validation, original.tcn_features)
    if variant == "repeat_control":
        second_raw = original_raw
        second_frame = original.train
    elif variant == "coherent_nwp":
        second_raw = clean_matrix(augmented.train, augmented.tcn_features)
        second_frame = augmented.train
    else:
        raise ValueError(f"Unknown training variant: {variant}")

    scaler = SequenceStandardScaler().fit(
        np.concatenate([original_raw, second_raw], axis=0)[:, None, :]
    )
    original_features = scaler.transform(original_raw[:, None, :])[:, 0, :]
    second_features = scaler.transform(second_raw[:, None, :])[:, 0, :]
    validation_features = scaler.transform(validation_raw[:, None, :])[:, 0, :]
    first_dataset = make_sequence_dataset(
        original.train, original_features, base_config, keep_observed=True
    )
    second_dataset = make_sequence_dataset(
        second_frame, second_features, base_config, keep_observed=True
    )
    validation_dataset = make_sequence_dataset(
        original.validation,
        validation_features,
        base_config,
        keep_observed=False,
    )
    return ConcatDataset([first_dataset, second_dataset]), validation_dataset


def fit_variant(
    original: FoldData,
    augmented: FoldData,
    variant: str,
    experiment: dict,
    base_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    train_dataset, validation_dataset = prepare_variant_data(
        original, augmented, base_config, variant
    )
    config = base_config["tcn"]
    loader = make_loader(
        train_dataset, int(config["batch_size"]), True, seed, device
    )
    model = make_model(len(original.tcn_features), config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_stats = {}
    fixed_epoch = int(experiment["fixed_epoch"])
    for epoch in range(1, fixed_epoch + 1):
        train_stats = train_direct_epoch(
            model,
            loader,
            optimizer,
            experiment["loss_mode"],
            float(experiment["gamma"]),
            float(config["grad_clip"]),
            device,
        )
        print(
            f"  {variant} val={original.pred_year} "
            f"epoch={epoch:03d}/{fixed_epoch:03d}",
            flush=True,
        )
    prediction = predict_tcn(
        model, validation_dataset, int(config["eval_batch_size"]), device
    )
    score, group_rows = fold_metric(original.validation, prediction)
    stats = {
        "variant": variant,
        "pred_year": original.pred_year,
        "fixed_epoch": fixed_epoch,
        "score": score,
        "group_scores": ";".join(
            f"{row['group']}={row['score']:.6f}" for row in group_rows
        ),
        "n_train_samples": len(train_dataset),
        **train_stats,
    }
    del model, optimizer, loader, train_dataset, validation_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats


def read_baseline(
    path: Path, max_outer_folds: int, baseline_column: str
) -> pd.DataFrame:
    baseline = pd.read_csv(path, encoding="utf-8-sig")
    required = [*KEYS, "actual", baseline_column]
    missing = [column for column in required if column not in baseline]
    if missing:
        raise ValueError(f"Baseline OOF columns missing: {missing}")
    baseline["forecast_kst_dtm"] = pd.to_datetime(baseline["forecast_kst_dtm"])
    if max_outer_folds:
        years = sorted(baseline["pred_year"].unique())[:max_outer_folds]
        baseline = baseline.loc[baseline["pred_year"].isin(years)]
    return baseline[required].rename(columns={baseline_column: "baseline"})


def fixed_ensemble_summary(
    tcn_oof: pd.DataFrame,
    pinn_path: Path,
    tree_path: Path,
    pinn_column: str,
    tree_column: str,
    weights: tuple[float, float, float] | list[float],
) -> pd.DataFrame:
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (3,) or not np.isclose(weights_array.sum(), 1.0):
        raise ValueError("PINN, TREE, and TCN ensemble weights must sum to one")

    def read_branch(path: Path, column: str, branch: str) -> pd.DataFrame:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = [*KEYS, "actual", column]
        missing = [name for name in required if name not in frame]
        if missing:
            raise ValueError(f"{path} ensemble columns missing: {missing}")
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
        return frame[required].rename(
            columns={"actual": f"actual_{branch}", column: branch}
        )

    pinn = read_branch(pinn_path, pinn_column, "pinn")
    tree = read_branch(tree_path, tree_column, "tree")
    aligned = tcn_oof.merge(pinn, on=KEYS, how="inner", validate="one_to_one").merge(
        tree, on=KEYS, how="inner", validate="one_to_one"
    )
    if len(aligned) != len(tcn_oof):
        raise ValueError("PINN, TREE, and TCN OOF rows do not align")
    actuals = aligned[["actual", "actual_pinn", "actual_tree"]].to_numpy(float)
    if not np.allclose(actuals, actuals[:, :1], rtol=0.0, atol=1e-4):
        raise ValueError("PINN, TREE, and TCN OOF targets differ")

    for tcn_column in ("baseline", *VARIANTS):
        name = f"ensemble_{tcn_column}"
        aligned[name] = (
            weights_array[0] * aligned["pinn"]
            + weights_array[1] * aligned["tree"]
            + weights_array[2] * aligned[tcn_column]
        )
    columns = [f"ensemble_{name}" for name in ("baseline", *VARIANTS)]
    summary, _ = score_predictions(aligned, columns)
    return summary.assign(
        pinn_weight=weights_array[0],
        tree_weight=weights_array[1],
        tcn_weight=weights_array[2],
    )


def combined_summary(oof: pd.DataFrame) -> pd.DataFrame:
    variants = ["baseline", *VARIANTS]
    overall, diagnostics = score_predictions(oof, variants)
    overall_rows = overall.rename(columns={"mean_score": "score"}).assign(
        row_type="overall", pred_year=np.nan, group="all"
    )
    detail_rows = diagnostics.rename(columns={"n_rows": "rows"})
    return pd.concat([overall_rows, detail_rows], ignore_index=True, sort=False)


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    overall = summary.loc[summary["row_type"].eq("overall")].set_index("variant")
    groups = summary.loc[summary["row_type"].eq("pooled_group")].copy()
    groups["delta"] = groups["score"] - groups["group"].map(
        groups.loc[groups["variant"].eq("baseline")].set_index("group")["score"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = ["baseline", *VARIANTS]
    labels = ["Baseline", "Repeat control", "Coherent NWP"]
    colors = ["#4C78A8", "#A0A0A0", "#F58518"]
    axes[0].bar(labels, overall.loc[order, "score"], color=colors)
    axes[0].set_title("Pooled OOF score")
    axes[0].set_ylim(max(0.0, overall["score"].min() - 0.02), overall["score"].max() + 0.01)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(TARGET_COLS))
    width = 0.34
    for index, variant in enumerate(VARIANTS):
        values = (
            groups.loc[groups["variant"].eq(variant)]
            .set_index("group")
            .reindex(TARGET_COLS)["delta"]
        )
        axes[1].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=labels[index + 1],
            color=colors[index + 1],
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, ["Group 1", "Group 2", "Group 3"])
    axes[1].set_title("Score change from baseline")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment, base_config = load_configs(args.config, args.smoke_test)
    if args.loss_mode is not None:
        experiment["loss_mode"] = args.loss_mode
    if args.fixed_epoch is not None and not args.smoke_test:
        experiment["fixed_epoch"] = args.fixed_epoch
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} fixed_epoch={experiment['fixed_epoch']} "
        f"variants={VARIANTS}",
        flush=True,
    )

    ldaps, gfs, labels, scada_by_group = read_inputs()
    original_folds = build_folds(
        ldaps, gfs, labels, scada_by_group, args.max_outer_folds, "original"
    )
    augmentation = experiment["augmentation"]
    augmented_ldaps, augmented_gfs, parameters = perturb_nwp_issues(
        ldaps,
        gfs,
        seed=int(experiment["seed"]),
        speed_scale_std=float(augmentation["speed_scale_std"]),
        direction_std_deg=float(augmentation["direction_std_deg"]),
        max_speed_scale_delta=float(augmentation["max_speed_scale_delta"]),
        max_direction_deg=float(augmentation["max_direction_deg"]),
    )
    print(
        "augmentation issues="
        f"{len(parameters)} speed_std={parameters['speed_scale'].std(ddof=0):.4f} "
        f"direction_std_deg={parameters['direction_deg'].std(ddof=0):.3f}",
        flush=True,
    )
    del ldaps, gfs, parameters
    gc.collect()
    augmented_folds = build_folds(
        augmented_ldaps,
        augmented_gfs,
        labels,
        scada_by_group,
        args.max_outer_folds,
        "augmented",
    )
    del augmented_ldaps, augmented_gfs, labels, scada_by_group
    gc.collect()

    if len(original_folds) != len(augmented_folds):
        raise ValueError("Original and augmented fold counts differ")
    for original, augmented in zip(original_folds, augmented_folds, strict=True):
        validate_fold_pair(original, augmented)

    baseline_path = (
        args.baseline_oof
        if args.baseline_oof is not None
        else Path(experiment["baseline_oof_path"])
    )
    baseline_column = args.baseline_column or experiment["loss_mode"]
    oof = read_baseline(baseline_path, args.max_outer_folds, baseline_column)
    training_rows = []
    for variant in VARIANTS:
        parts = []
        for fold_index, (original, augmented) in enumerate(
            zip(original_folds, augmented_folds, strict=True)
        ):
            print(f"\n=== {variant} val={original.pred_year} ===", flush=True)
            prediction, stats = fit_variant(
                original,
                augmented,
                variant,
                experiment,
                base_config,
                device,
                int(experiment["seed"]) + 10000 + fold_index * 1000,
            )
            training_rows.append(stats)
            parts.append(
                tcn_prediction_frame(original, prediction).rename(
                    columns={"tcn_multihead": variant}
                )
            )
        variant_oof = pd.concat(parts, ignore_index=True)
        oof = oof.merge(
            variant_oof[KEYS + [variant]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

    summary = combined_summary(oof)
    for row in training_rows:
        print(
            f"{row['variant']} val={row['pred_year']} score={row['score']:.6f}",
            flush=True,
        )
    oof_path = args.results_dir / f"{args.stem}_oof.csv"
    summary_path = args.results_dir / f"{args.stem}_summary.csv"
    figure_path = args.results_dir / f"{args.stem}_comparison.png"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_comparison(summary, figure_path)
    if (args.pinn_oof is None) != (args.tree_oof is None):
        raise ValueError("Both --pinn-oof and --tree-oof are required for ensemble")
    ensemble_path = None
    if args.pinn_oof is not None and args.tree_oof is not None:
        ensemble = fixed_ensemble_summary(
            oof,
            args.pinn_oof,
            args.tree_oof,
            args.pinn_column,
            args.tree_column,
            args.ensemble_weights,
        )
        ensemble_path = args.results_dir / f"{args.stem}_ensemble_summary.csv"
        ensemble.to_csv(ensemble_path, index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF ===", flush=True)
    print(
        summary.loc[summary["row_type"].eq("overall"), [
            "variant",
            "score",
            "mean_nmae",
            "mean_ficr",
        ]].to_string(index=False),
        flush=True,
    )
    print(f"saved {oof_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {figure_path}", flush=True)
    if ensemble_path is not None:
        print(f"saved {ensemble_path}", flush=True)


if __name__ == "__main__":
    main()
