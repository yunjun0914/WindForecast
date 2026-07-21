from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from experiments.evaluate_group_base_recovery_oof import (
    build_fold_data,
    build_site_static,
    make_loader,
    make_model,
    predict_tcn,
    prepare_tcn_data,
    sample_weight,
    tree_params,
)
from experiments.evaluate_group_pinn_band_loss_oof import (
    PINNFold,
    make_pinn,
    make_pinn_loader,
    train_pinn_epoch,
    validation_prediction,
)
from experiments.evaluate_group_tcn_band_loss_oof import train_direct_epoch
from experiments.evaluate_group_unified_oof import (
    build_fold_feature_tables,
    build_group_tables,
    fit_teacher,
    read_inputs,
    set_seed,
)
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


HYBRID_WEIGHTS = {
    "pinn": 0.33333333333333337,
    "tree": 0.05128205128205129,
    "tcn": 0.6153846153846154,
}
ALL_PURE_WEIGHTS = {
    "pinn": 0.34210526315789475,
    "tree": 0.052631578947368425,
    "tcn": 0.6052631578947368,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pinn-config",
        type=Path,
        default=Path("configs/group_pinn_band_loss_v1.json"),
    )
    parser.add_argument(
        "--tcn-config",
        type=Path,
        default=Path("configs/group_tcn_band_loss_v1.json"),
    )
    parser.add_argument(
        "--pinn-diagnostics",
        type=Path,
        default=Path("results/group_pinn_direct_band_loss_v1_epoch_diagnostics.csv"),
    )
    parser.add_argument(
        "--tcn-diagnostics",
        type=Path,
        default=Path("results/group_tcn_direct_band_loss_v1_epoch_diagnostics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--hybrid-name", default="submission_yunjun_group_hybrid_v1.csv"
    )
    parser.add_argument(
        "--all-pure-name", default="submission_yunjun_group_allpure_v1.csv"
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def fixed_epochs(path: Path, groups: bool) -> dict:
    diagnostics = pd.read_csv(path, encoding="utf-8-sig")
    rows = diagnostics.loc[diagnostics["stage"].eq("direct_fixed_epoch")].copy()
    if rows.empty:
        raise ValueError(f"No fixed epochs in {path}")
    output = {}
    for mode in ("pure_band_ficr", "ficr_nmae"):
        selected = rows.loc[rows["variant"].eq(mode)]
        if groups:
            for group in TARGET_COLS:
                values = selected.loc[selected["group"].eq(group), "fixed_epoch"].dropna()
                unique = sorted(values.astype(int).unique().tolist())
                if len(unique) != 1:
                    raise ValueError(f"Expected one epoch for {mode}/{group}: {unique}")
                output[(mode, group)] = unique[0]
        else:
            unique = sorted(selected["fixed_epoch"].dropna().astype(int).unique().tolist())
            if len(unique) != 1:
                raise ValueError(f"Expected one TCN epoch for {mode}: {unique}")
            output[mode] = unique[0]
    return output


def fit_full_tree(
    fold,
    best: pd.DataFrame,
    smoke_test: bool,
) -> dict[str, np.ndarray]:
    predictions = {}
    for group in TARGET_COLS:
        selected = best.loc[best["group"].eq(group)]
        if len(selected) != 1:
            raise ValueError(f"Expected one TREE config for {group}")
        selected = selected.iloc[0]
        target = fold.train[group].to_numpy(float)
        keep = np.isfinite(target) & (
            target
            >= GROUP_CAPACITY_KWH[group] * float(selected["min_output_ratio"])
        )
        feature_cols = fold.group_features[group]["common72"]
        model = LGBMRegressor(
            **tree_params(
                selected,
                "regression_l1",
                40 if smoke_test else 0,
            )
        )
        model.fit(
            fold.train.loc[keep, feature_cols],
            target[keep],
            sample_weight=sample_weight(
                target[keep], group, str(selected["weight_policy"])
            ),
        )
        predictions[group] = np.clip(
            model.predict(fold.validation[feature_cols]),
            0.0,
            GROUP_CAPACITY_KWH[group],
        )
        print(f"TREE {group}: train={int(keep.sum())}", flush=True)
    return predictions


def fit_full_tcn(
    fold,
    experiment: dict,
    base_config: dict,
    epoch: int,
    device: torch.device,
) -> np.ndarray:
    seed = int(experiment["seed"]) + 10000
    set_seed(seed)
    train_dataset, test_dataset = prepare_tcn_data(fold, base_config)
    config = base_config["tcn"]
    loader = make_loader(
        train_dataset, int(config["batch_size"]), True, seed, device
    )
    model = make_model(len(fold.tcn_features), config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    for current in range(1, epoch + 1):
        stats = train_direct_epoch(
            model,
            loader,
            optimizer,
            "pure_band_ficr",
            float(experiment["gamma"]),
            float(config["grad_clip"]),
            device,
        )
        print(
            f"TCN Pure epoch={current:03d}/{epoch:03d} "
            f"loss={stats['train_loss']:.6f}",
            flush=True,
        )
    prediction = predict_tcn(
        model, test_dataset, int(config["eval_batch_size"]), device
    )
    del model, optimizer, loader, train_dataset, test_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction


def fit_full_pinn(
    fold: PINNFold,
    mode: str,
    epoch: int,
    config: dict,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    model = make_pinn(fold.group, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = make_pinn_loader(fold, int(config["batch_size"]), seed, device)
    for current in range(1, epoch + 1):
        stats = train_pinn_epoch(model, loader, optimizer, mode, config, device)
        if current == 1 or current == epoch or current % 10 == 0:
            print(
                f"PINN {mode} {fold.group} epoch={current:03d}/{epoch:03d} "
                f"loss={stats['train_loss']:.6f}",
                flush=True,
            )
    prediction = validation_prediction(model, fold, config, device)
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction


def align_prediction(
    sample: pd.DataFrame,
    times: pd.Series,
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    prediction = pd.DataFrame(
        {"forecast_kst_dtm": pd.to_datetime(times), name: values}
    )
    if prediction["forecast_kst_dtm"].duplicated().any():
        raise ValueError(f"Duplicate test timestamps for {name}")
    aligned = sample[["forecast_kst_dtm"]].merge(
        prediction, on="forecast_kst_dtm", how="left", validate="one_to_one"
    )
    if aligned[name].isna().any():
        raise ValueError(f"Missing test predictions for {name}")
    return aligned[name].to_numpy(float)


def validate_submission(output: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(output.columns) != list(sample.columns):
        raise ValueError(f"Submission columns differ: {output.columns.tolist()}")
    if len(output) != len(sample):
        raise ValueError("Submission row count differs from sample")
    if not output["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError("forecast_id order differs from sample")
    values = output[TARGET_COLS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Submission has non-finite predictions")
    for group in TARGET_COLS:
        if not output[group].between(0.0, GROUP_CAPACITY_KWH[group]).all():
            raise ValueError(f"Submission predictions out of range for {group}")


def build_submission(
    sample: pd.DataFrame,
    pinn: dict[str, np.ndarray],
    tree: dict[str, np.ndarray],
    tcn: dict[str, np.ndarray],
    weights: dict[str, float],
) -> pd.DataFrame:
    output = sample.copy()
    for group in TARGET_COLS:
        output[group] = np.clip(
            weights["pinn"] * pinn[group]
            + weights["tree"] * tree[group]
            + weights["tcn"] * tcn[group],
            0.0,
            GROUP_CAPACITY_KWH[group],
        )
    validate_submission(output, sample)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_experiment = json.loads(args.pinn_config.read_text(encoding="utf-8"))
    pinn_base = json.loads(
        Path(pinn_experiment["base_config_path"]).read_text(encoding="utf-8")
    )
    tcn_experiment = json.loads(args.tcn_config.read_text(encoding="utf-8"))
    tcn_base = json.loads(
        Path(tcn_experiment["base_config_path"]).read_text(encoding="utf-8")
    )
    pinn_epochs = fixed_epochs(args.pinn_diagnostics, groups=True)
    tcn_epochs = fixed_epochs(args.tcn_diagnostics, groups=False)
    if args.smoke_test:
        pinn_base["teacher_trees"] = min(24, int(pinn_base["teacher_trees"]))
        pinn_epochs = {key: 1 for key in pinn_epochs}
        tcn_epochs = {key: 1 for key in tcn_epochs}

    ldaps_train, gfs_train, labels, scada_by_group = read_inputs()
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    sample = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    ldaps = pd.concat([ldaps_train, ldaps_test], ignore_index=True)
    gfs = pd.concat([gfs_train, gfs_test], ignore_index=True)
    train_times = labels.loc[
        labels[TARGET_COLS].notna().any(axis=1), "kst_dtm"
    ].to_numpy()
    test_times = sample["forecast_kst_dtm"].to_numpy()

    print("build shared full train/test features", flush=True)
    site_static = build_site_static(ldaps, gfs)
    tables_by_group = {
        group: build_group_tables(
            ldaps,
            gfs,
            labels,
            scada_by_group[group],
            group,
            include_residual=False,
        )
        for group in TARGET_COLS
    }
    full_fold = build_fold_data(
        tables_by_group,
        site_static,
        labels,
        train_times,
        test_times,
        2025,
    )

    tree_best = pd.read_csv(tcn_base["tree_best_path"], encoding="utf-8-sig")
    tree_raw = fit_full_tree(full_fold, tree_best, args.smoke_test)
    tcn_norm = fit_full_tcn(
        full_fold,
        tcn_experiment,
        tcn_base,
        tcn_epochs["pure_band_ficr"],
        device,
    )
    test_order = full_fold.validation["forecast_kst_dtm"]
    tree_predictions = {
        group: align_prediction(sample, test_order, tree_raw[group], f"tree_{group}")
        for group in TARGET_COLS
    }
    tcn_predictions = {
        group: align_prediction(
            sample,
            test_order,
            tcn_norm[:, group_index] * GROUP_CAPACITY_KWH[group],
            f"tcn_{group}",
        )
        for group_index, group in enumerate(TARGET_COLS)
    }

    pinn_predictions = {mode: {} for mode in ("pure_band_ficr", "ficr_nmae")}
    for group_index, group in enumerate(TARGET_COLS):
        group_train_times = labels.loc[labels[group].notna(), "kst_dtm"].to_numpy()
        train_base, test_base, feature_cols = build_fold_feature_tables(
            tables_by_group[group], group, group_train_times, test_times
        )
        label_table = labels[["kst_dtm", group]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", group: "target"}
        )
        train = train_base.merge(
            label_table, on="forecast_kst_dtm", how="inner", validate="one_to_one"
        ).dropna(subset=["target"])
        teacher_seed = int(pinn_experiment["seed"]) + group_index * 10000 + 20250
        teacher_train, teacher_test = fit_teacher(
            train,
            test_base,
            tables_by_group[group].teacher_targets,
            feature_cols,
            int(pinn_base["teacher_trees"]),
            teacher_seed,
        )
        fold = PINNFold(
            group=group,
            pred_year=2025,
            train=train,
            validation=test_base.assign(target=np.nan),
            teacher_train=teacher_train,
            teacher_validation=teacher_test,
        )
        for mode in ("pure_band_ficr", "ficr_nmae"):
            prediction = fit_full_pinn(
                fold,
                mode,
                pinn_epochs[(mode, group)],
                pinn_base["pinn"],
                device,
                int(pinn_experiment["seed"]) + 10000 + group_index * 1000,
            )
            pinn_predictions[mode][group] = align_prediction(
                sample,
                test_base["forecast_kst_dtm"],
                prediction,
                f"pinn_{mode}_{group}",
            )
        del fold, teacher_train, teacher_test
        gc.collect()

    hybrid = build_submission(
        sample,
        pinn_predictions["ficr_nmae"],
        tree_predictions,
        tcn_predictions,
        HYBRID_WEIGHTS,
    )
    all_pure = build_submission(
        sample,
        pinn_predictions["pure_band_ficr"],
        tree_predictions,
        tcn_predictions,
        ALL_PURE_WEIGHTS,
    )
    paths = [
        (args.output_dir / args.hybrid_name, hybrid),
        (args.output_dir / args.all_pure_name, all_pure),
    ]
    for path, output in paths:
        output.to_csv(path, index=False, encoding="utf-8-sig")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"saved {path}: rows={len(output)} sha256={digest}", flush=True)
        print(output[TARGET_COLS].agg(["min", "max", "mean"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
