from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from experiments.evaluate_group_base_recovery_oof import score_predictions
from experiments.evaluate_group_unified_oof import (
    YEARS,
    available_target_times,
    build_fold_tables,
    build_group_tables,
    calendar_arrays,
    fit_teacher,
    predict_pinn,
    read_inputs,
    set_seed,
)
from models.group_unified import GroupPhysicsPINN, normalized_metric_loss
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.pinn_data import C_MAX_BY_MANUFACTURER
from utils.pinn_physics import MANUFACTURER_AREA
from utils.power_curve import GROUP_MANUFACTURER, GROUP_N_TURBINES


LOSS_MODES = ("pure_band_ficr", "ficr_nmae")


@dataclass
class PINNFold:
    group: str
    pred_year: int
    train: pd.DataFrame
    validation: pd.DataFrame
    teacher_train: np.ndarray
    teacher_validation: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_pinn_band_loss_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_pinn_direct_band_loss_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-outer-folds", type=int, default=0)
    return parser.parse_args()


def load_configs(path: Path, smoke_test: bool) -> tuple[dict, dict]:
    experiment = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(Path(experiment["base_config_path"]).read_text(encoding="utf-8"))
    if smoke_test:
        experiment["max_epochs"] = 2
        experiment["patience"] = 2
        base["teacher_trees"] = min(24, int(base["teacher_trees"]))
    return experiment, base


def make_pinn(group: str, config: dict, device: torch.device) -> GroupPhysicsPINN:
    manufacturer = GROUP_MANUFACTURER[group]
    capacity = GROUP_CAPACITY_KWH[group]
    return GroupPhysicsPINN(
        n_turbines=GROUP_N_TURBINES[group],
        rotor_area_m2=MANUFACTURER_AREA[manufacturer],
        rated_power_w=capacity * 1000.0,
        c_max=C_MAX_BY_MANUFACTURER[manufacturer],
        hidden_size=int(config["hidden_size"]),
        residual_amplitude=float(config["residual_amplitude"]),
    ).to(device)


def make_pinn_loader(
    fold: PINNFold,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    capacity = GROUP_CAPACITY_KWH[fold.group]
    target = (fold.train["target"].to_numpy(np.float32) / capacity).astype(np.float32)
    rho = fold.train["phys_gfs_air_density"].to_numpy(np.float32)
    doy, lead = calendar_arrays(fold.train)
    keep = np.isfinite(target) & (target >= 0.10)
    dataset = TensorDataset(
        torch.from_numpy(fold.teacher_train[keep]),
        torch.from_numpy(rho[keep]),
        torch.from_numpy(doy[keep]),
        torch.from_numpy(lead[keep]),
        torch.from_numpy(target[keep]),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )


def data_loss_weights(mode: str) -> tuple[float, float]:
    if mode == "pure_band_ficr":
        return 0.0, 1.0
    if mode == "ficr_nmae":
        return 0.5, 0.5
    raise ValueError(f"Unknown PINN loss mode: {mode}")


def train_pinn_epoch(
    model: GroupPhysicsPINN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mode: str,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    nmae_weight, ficr_weight = data_loss_weights(mode)
    losses = []
    nmaes = []
    ficrs = []
    model.train()
    for teacher, rho, doy, lead, target in loader:
        teacher = teacher.to(device, non_blocking=True)
        rho = rho.to(device, non_blocking=True)
        doy = doy.to(device, non_blocking=True)
        lead = lead.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(teacher, rho, doy, lead)
        data_loss, nmae, ficr = normalized_metric_loss(
            prediction,
            target,
            gamma=float(config["gamma"]),
            nmae_weight=nmae_weight,
            ficr_weight=ficr_weight,
        )
        regularization = model.physics_regularization(device)
        loss = (
            data_loss
            + float(config["boundary_weight"]) * regularization["boundary"]
            + float(config["smoothness_weight"]) * regularization["smoothness"]
            + float(config["flatness_weight"]) * regularization["flatness"]
            + float(config["residual_l2_weight"]) * regularization["residual_l2"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        nmaes.append(float(nmae.cpu()))
        ficrs.append(float(ficr.cpu()))
    if not losses:
        raise RuntimeError("PINN epoch had no observed targets")
    return {
        "train_loss": float(np.mean(losses)),
        "train_soft_nmae": float(np.mean(nmaes)),
        "train_soft_ficr": float(np.mean(ficrs)),
    }


def validation_prediction(
    model: GroupPhysicsPINN,
    fold: PINNFold,
    config: dict,
    device: torch.device,
) -> np.ndarray:
    rho = fold.validation["phys_gfs_air_density"].to_numpy(np.float32)
    doy, lead = calendar_arrays(fold.validation)
    normalized = predict_pinn(
        model,
        fold.teacher_validation,
        rho,
        doy,
        lead,
        device,
        int(config["batch_size"]) * 4,
    )
    return normalized * GROUP_CAPACITY_KWH[fold.group]


def fold_metric(fold: PINNFold, prediction: np.ndarray) -> dict[str, float]:
    actual = fold.validation["target"].to_numpy(float)
    nmae, ficr = group_nmae_ficr(
        actual, prediction, GROUP_CAPACITY_KWH[fold.group]
    )
    return {
        "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
        "nmae": nmae,
        "ficr": ficr,
    }


def discover_epoch(
    fold: PINNFold,
    mode: str,
    experiment: dict,
    pinn_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict]]:
    set_seed(seed)
    model = make_pinn(fold.group, pinn_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pinn_config["learning_rate"]),
        weight_decay=float(pinn_config["weight_decay"]),
    )
    loader = make_pinn_loader(
        fold, int(pinn_config["batch_size"]), seed, device
    )
    best_score = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, int(experiment["max_epochs"]) + 1):
        train_stats = train_pinn_epoch(
            model, loader, optimizer, mode, pinn_config, device
        )
        metrics = fold_metric(
            fold, validation_prediction(model, fold, pinn_config, device)
        )
        history.append(
            {
                "stage": "direct_epoch_discovery",
                "variant": mode,
                "group": fold.group,
                "pred_year": fold.pred_year,
                "epoch": epoch,
                **metrics,
                **train_stats,
            }
        )
        if metrics["score"] > best_score + float(experiment["min_delta"]):
            best_score = metrics["score"]
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"  direct {mode} {fold.group} val={fold.pred_year} epoch={epoch:03d} "
            f"score={metrics['score']:.6f} best={best_score:.6f}@{best_epoch}",
            flush=True,
        )
        if bad_epochs >= int(experiment["patience"]):
            break
    if best_epoch <= 0:
        raise RuntimeError(
            f"No PINN epoch selected for {mode}/{fold.group}/{fold.pred_year}"
        )
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history


def fit_fixed_epoch(
    fold: PINNFold,
    mode: str,
    fixed_epoch: int,
    pinn_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    model = make_pinn(fold.group, pinn_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pinn_config["learning_rate"]),
        weight_decay=float(pinn_config["weight_decay"]),
    )
    loader = make_pinn_loader(
        fold, int(pinn_config["batch_size"]), seed, device
    )
    train_stats = {}
    for epoch in range(1, fixed_epoch + 1):
        train_stats = train_pinn_epoch(
            model, loader, optimizer, mode, pinn_config, device
        )
        print(
            f"  fixed {mode} {fold.group} val={fold.pred_year} "
            f"epoch={epoch:03d}/{fixed_epoch:03d}",
            flush=True,
        )
    prediction = validation_prediction(model, fold, pinn_config, device)
    metrics = fold_metric(fold, prediction)
    stats = {
        "stage": "direct_fixed_epoch",
        "variant": mode,
        "group": fold.group,
        "pred_year": fold.pred_year,
        "fixed_epoch": fixed_epoch,
        **metrics,
        **train_stats,
    }
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats


def build_folds(
    experiment: dict,
    base_config: dict,
    max_outer_folds: int,
) -> list[PINNFold]:
    ldaps, gfs, labels, scada_by_group = read_inputs()
    folds = []
    for group_index, group in enumerate(TARGET_COLS):
        print(f"\n=== build PINN group data: {group} ===", flush=True)
        tables = build_group_tables(
            ldaps,
            gfs,
            labels,
            scada_by_group[group],
            group,
            include_residual=False,
        )
        available = available_target_times(tables)
        for pred_year in YEARS:
            if max_outer_folds and len(folds) >= max_outer_folds:
                break
            outer_train = available.loc[available["year"].ne(pred_year)]
            outer_validation = available.loc[available["year"].eq(pred_year)]
            if len(outer_train) < 500 or len(outer_validation) < 200:
                continue
            train, validation, feature_cols = build_fold_tables(
                tables,
                group,
                outer_train["forecast_kst_dtm"].to_numpy(),
                outer_validation["forecast_kst_dtm"].to_numpy(),
            )
            teacher_seed = (
                int(experiment["seed"]) + group_index * 10000 + pred_year * 10
            )
            teacher_train, teacher_validation = fit_teacher(
                train,
                validation,
                tables.teacher_targets,
                feature_cols,
                int(base_config["teacher_trees"]),
                teacher_seed,
            )
            folds.append(
                PINNFold(
                    group=group,
                    pred_year=pred_year,
                    train=train[
                        [
                            "forecast_kst_dtm",
                            "data_available_kst_dtm",
                            "phys_gfs_air_density",
                            "target",
                        ]
                    ].copy(),
                    validation=validation[
                        [
                            "forecast_kst_dtm",
                            "data_available_kst_dtm",
                            "phys_gfs_air_density",
                            "target",
                        ]
                    ].copy(),
                    teacher_train=teacher_train,
                    teacher_validation=teacher_validation,
                )
            )
            print(
                f"cached teacher {group} val={pred_year}: "
                f"train={len(train)} validation={len(validation)}",
                flush=True,
            )
        del tables
        gc.collect()
        if max_outer_folds and len(folds) >= max_outer_folds:
            break
    del ldaps, gfs, labels, scada_by_group
    gc.collect()
    return folds


def main() -> None:
    args = parse_args()
    experiment, base_config = load_configs(args.config, args.smoke_test)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} smoke={args.smoke_test} training=direct", flush=True)
    folds = build_folds(experiment, base_config, args.max_outer_folds)
    pinn_config = base_config["pinn"]

    epoch_rows = []
    discovered = {
        mode: {group: [] for group in TARGET_COLS} for mode in LOSS_MODES
    }
    for mode in LOSS_MODES:
        for fold_index, fold in enumerate(folds):
            print(
                f"\n=== discovery {mode} {fold.group} val={fold.pred_year} ===",
                flush=True,
            )
            best_epoch, history = discover_epoch(
                fold,
                mode,
                experiment,
                pinn_config,
                device,
                int(experiment["seed"]) + fold_index * 1000,
            )
            discovered[mode][fold.group].append(best_epoch)
            epoch_rows.extend(history)

    fixed_epochs = {
        mode: {
            group: int(np.median(values))
            for group, values in groups.items()
            if values
        }
        for mode, groups in discovered.items()
    }
    print(f"\nfixed direct epochs={fixed_epochs}", flush=True)

    oof_parts = []
    for fold_index, fold in enumerate(folds):
        output = pd.DataFrame(
            {
                "forecast_kst_dtm": pd.to_datetime(
                    fold.validation["forecast_kst_dtm"]
                ),
                "pred_year": fold.pred_year,
                "group": fold.group,
                "actual": fold.validation["target"].to_numpy(float),
            }
        )
        for mode in LOSS_MODES:
            fixed_epoch = fixed_epochs[mode][fold.group]
            print(
                f"\n=== fixed {mode} {fold.group} val={fold.pred_year} ===",
                flush=True,
            )
            prediction, stats = fit_fixed_epoch(
                fold,
                mode,
                fixed_epoch,
                pinn_config,
                device,
                int(experiment["seed"]) + 10000 + fold_index * 1000,
            )
            output[mode] = prediction
            epoch_rows.append(stats)
        oof_parts.append(output)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["group", "forecast_kst_dtm"]
    )
    summary, score_diagnostics = score_predictions(oof, list(LOSS_MODES))
    diagnostics = pd.concat(
        [pd.DataFrame(epoch_rows).assign(row_type="training"), score_diagnostics],
        ignore_index=True,
    )
    oof_path = args.results_dir / f"{args.stem}_oof.csv"
    summary_path = args.results_dir / f"{args.stem}_summary.csv"
    diagnostics_path = args.results_dir / f"{args.stem}_epoch_diagnostics.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved {oof_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
