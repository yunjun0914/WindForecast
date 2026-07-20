from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from torch.utils.data import DataLoader, Dataset

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401
from experiments.evaluate_group_unified_oof import (
    BASE_EXTRA_FEATURES,
    GroupTables,
    build_fold_feature_tables,
    build_group_tables,
    read_inputs,
    select_times,
)
from models.group_unified import MultiHeadTCNPowerRegressor
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.preprocessing import TIME_KEY_COLS
from utils.seq_dataset import SequenceStandardScaler
from utils.tree_feature_profiles import (
    FEATURE_PROFILE_FULL_V2,
    GROUP_FAMILY_QUOTA65_V1_FEATURES,
    build_tree_features,
)
from utils.weather_time_features import add_weather_time_features


YEARS = [2022, 2023, 2024]
TREE_VARIANTS = {
    "tree_quota65_l1": ("quota65", "regression_l1"),
    "tree_common72_l1": ("common72", "regression_l1"),
    "tree_common72_mse": ("common72", "regression"),
}
TREE_PARAM_COLUMNS = [
    "random_state",
    "n_jobs",
    "verbose",
    "objective",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_split_gain",
    "subsample_freq",
]
SITE_QUOTA_FEATURES = list(
    dict.fromkeys(
        feature
        for group in TARGET_COLS
        for feature in GROUP_FAMILY_QUOTA65_V1_FEATURES[group]
    )
)
SITE_FEATURES = [*SITE_QUOTA_FEATURES, "phys_gfs_air_density"]
GROUP_TCN_FEATURES = [
    *[feature for feature in BASE_EXTRA_FEATURES if feature.startswith("wake_")],
    "power_curve_est",
]


@dataclass
class FoldData:
    pred_year: int
    train: pd.DataFrame
    validation: pd.DataFrame
    tcn_features: list[str]
    group_features: dict[str, dict[str, list[str]]]


class CausalWindowDataset(Dataset):
    """Build causal windows lazily and never cross a calendar-year boundary."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        observed: np.ndarray,
        times: pd.Series | np.ndarray,
        window: int,
        keep_observed: bool,
    ) -> None:
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.targets = np.ascontiguousarray(targets, dtype=np.float32)
        self.weights = np.ascontiguousarray(weights, dtype=np.float32)
        self.observed = np.ascontiguousarray(observed, dtype=np.float32)
        self.window = int(window)
        years = pd.DatetimeIndex(pd.to_datetime(times)).year.to_numpy()
        if len(years) != len(self.features):
            raise ValueError("Feature rows and timestamps have different lengths")
        starts = np.zeros(len(years), dtype=np.int64)
        current_start = 0
        for index in range(len(years)):
            if index == 0 or years[index] != years[index - 1]:
                current_start = index
            starts[index] = current_start
        self.year_starts = starts
        if keep_observed:
            self.indices = np.flatnonzero(self.observed.astype(bool).any(axis=1))
        else:
            self.indices = np.arange(len(self.features), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row_index = int(self.indices[item])
        start = max(int(self.year_starts[row_index]), row_index - self.window + 1)
        chunk = self.features[start : row_index + 1]
        if len(chunk) < self.window:
            padding = np.repeat(chunk[:1], self.window - len(chunk), axis=0)
            chunk = np.concatenate([padding, chunk], axis=0)
        return (
            chunk,
            self.targets[row_index],
            self.weights[row_index],
            self.observed[row_index],
            row_index,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_base_recovery_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_base_recovery_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-outer-folds", type=int, default=0)
    return parser.parse_args()


def load_config(path: Path, smoke_test: bool) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if smoke_test:
        config["tcn"]["max_epochs"] = 2
        config["tcn"]["patience"] = 2
        config["tcn"]["batch_size"] = 128
        config["tree_smoke_estimators"] = 40
    return config


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefix_for_group(group: str) -> str:
    return f"g{TARGET_COLS.index(group) + 1}__"


def build_site_static(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    site_full = add_weather_time_features(
        build_tree_features(ldaps, gfs, None, feature_profile=FEATURE_PROFILE_FULL_V2)
    )
    missing = [feature for feature in SITE_FEATURES if feature not in site_full]
    if missing:
        raise ValueError(f"Site features missing: {missing}")
    output = site_full[TIME_KEY_COLS + SITE_FEATURES].copy()
    for column in TIME_KEY_COLS:
        output[column] = pd.to_datetime(output[column])
    return output


def build_fold_data(
    tables_by_group: dict[str, GroupTables],
    site_static: pd.DataFrame,
    labels: pd.DataFrame,
    train_times: np.ndarray,
    validation_times: np.ndarray,
    pred_year: int,
) -> FoldData:
    train_parts = [select_times(site_static, train_times)]
    validation_parts = [select_times(site_static, validation_times)]
    group_features: dict[str, dict[str, list[str]]] = {}
    tcn_features = list(SITE_FEATURES)

    for group in TARGET_COLS:
        train, validation, feature_cols = build_fold_feature_tables(
            tables_by_group[group], group, train_times, validation_times
        )
        if len(feature_cols) != len(GROUP_FAMILY_QUOTA65_V1_FEATURES[group]) + len(
            BASE_EXTRA_FEATURES
        ):
            raise ValueError(f"Unexpected common feature count for {group}")
        prefix = prefix_for_group(group)
        renamed = {name: f"{prefix}{name}" for name in feature_cols}
        quota_cols = [
            renamed[name] for name in GROUP_FAMILY_QUOTA65_V1_FEATURES[group]
        ] + [renamed["power_curve_est"]]
        common_cols = [renamed[name] for name in feature_cols]
        group_features[group] = {
            "quota65": quota_cols,
            "common72": common_cols,
        }
        tcn_features.extend(renamed[name] for name in GROUP_TCN_FEATURES)
        train_parts.append(train[TIME_KEY_COLS + feature_cols].rename(columns=renamed))
        validation_parts.append(
            validation[TIME_KEY_COLS + feature_cols].rename(columns=renamed)
        )

    def merge_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.merge(
                part, on=TIME_KEY_COLS, how="inner", validate="one_to_one"
            )
        label_table = labels[["kst_dtm", *TARGET_COLS]].copy()
        label_table["kst_dtm"] = pd.to_datetime(label_table["kst_dtm"])
        merged = merged.merge(
            label_table,
            left_on="forecast_kst_dtm",
            right_on="kst_dtm",
            how="left",
            validate="one_to_one",
        ).drop(columns="kst_dtm")
        return merged.sort_values("forecast_kst_dtm").reset_index(drop=True)

    train = merge_parts(train_parts)
    validation = merge_parts(validation_parts)
    expected_tcn_features = len(SITE_FEATURES) + len(TARGET_COLS) * len(
        GROUP_TCN_FEATURES
    )
    if len(tcn_features) != expected_tcn_features:
        raise ValueError(
            f"Expected {expected_tcn_features} deduplicated TCN features, "
            f"received {len(tcn_features)}"
        )
    return FoldData(
        pred_year=pred_year,
        train=train,
        validation=validation,
        tcn_features=tcn_features,
        group_features=group_features,
    )


def prepare_folds(
    labels: pd.DataFrame,
    tables_by_group: dict[str, GroupTables],
    site_static: pd.DataFrame,
    max_outer_folds: int,
) -> list[FoldData]:
    master = tables_by_group[TARGET_COLS[0]].base_static[
        ["forecast_kst_dtm"]
    ].copy()
    master["forecast_kst_dtm"] = pd.to_datetime(master["forecast_kst_dtm"])
    master["year"] = master["forecast_kst_dtm"].dt.year
    master = master.loc[master["year"].isin(YEARS)].drop_duplicates(
        "forecast_kst_dtm"
    )
    folds = []
    for pred_year in YEARS:
        if max_outer_folds and len(folds) >= max_outer_folds:
            break
        train_times = master.loc[
            master["year"].ne(pred_year), "forecast_kst_dtm"
        ].to_numpy()
        validation_times = master.loc[
            master["year"].eq(pred_year), "forecast_kst_dtm"
        ].to_numpy()
        print(
            f"build fold val={pred_year}: train={len(train_times)} "
            f"validation={len(validation_times)}",
            flush=True,
        )
        folds.append(
            build_fold_data(
                tables_by_group,
                site_static,
                labels,
                train_times,
                validation_times,
                pred_year,
            )
        )
    return folds


def clean_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return (
        frame[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def target_arrays(
    frame: pd.DataFrame, min_output_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = frame[TARGET_COLS].to_numpy(np.float32)
    capacities = np.asarray(
        [GROUP_CAPACITY_KWH[group] for group in TARGET_COLS], dtype=np.float32
    )
    normalized = raw / capacities.reshape(1, -1)
    observed = np.isfinite(normalized) & (normalized >= min_output_ratio)
    normalized = np.nan_to_num(
        np.clip(normalized, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0
    ).astype(np.float32)
    weights = (0.5 + np.sqrt(normalized)).astype(np.float32)
    return raw, normalized, weights, observed.astype(np.float32)


def make_model(n_features: int, config: dict, device: torch.device):
    return MultiHeadTCNPowerRegressor(
        input_size=n_features,
        n_groups=len(TARGET_COLS),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
    ).to(device)


def prepare_tcn_data(
    fold: FoldData, config: dict
) -> tuple[CausalWindowDataset, CausalWindowDataset]:
    train_raw = clean_matrix(fold.train, fold.tcn_features)
    validation_raw = clean_matrix(fold.validation, fold.tcn_features)
    scaler = SequenceStandardScaler()
    train_features = scaler.fit_transform(train_raw[:, None, :])[:, 0, :]
    validation_features = scaler.transform(validation_raw[:, None, :])[:, 0, :]
    _, train_target, train_weight, train_observed = target_arrays(
        fold.train, float(config["min_output_ratio"])
    )
    _, validation_target, validation_weight, validation_observed = target_arrays(
        fold.validation, float(config["min_output_ratio"])
    )
    train_dataset = CausalWindowDataset(
        train_features,
        train_target,
        train_weight,
        train_observed,
        fold.train["forecast_kst_dtm"],
        int(config["tcn"]["window"]),
        keep_observed=True,
    )
    validation_dataset = CausalWindowDataset(
        validation_features,
        validation_target,
        validation_weight,
        validation_observed,
        fold.validation["forecast_kst_dtm"],
        int(config["tcn"]["window"]),
        keep_observed=False,
    )
    return train_dataset, validation_dataset


def make_loader(
    dataset: CausalWindowDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def train_tcn_epoch(
    model: MultiHeadTCNPowerRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for features, target, weight, observed, _ in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        observed = observed.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        group_losses = []
        for group_index in range(len(TARGET_COLS)):
            weighted_mask = weight[:, group_index] * observed[:, group_index]
            if float(weighted_mask.sum().detach().cpu()) <= 0.0:
                continue
            group_losses.append(
                (
                    torch.abs(prediction[:, group_index] - target[:, group_index])
                    * weighted_mask
                ).sum()
                / weighted_mask.sum().clamp_min(1.0)
            )
        if not group_losses:
            continue
        loss = torch.stack(group_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("TCN epoch had no observed targets")
    return float(np.mean(losses))


def predict_tcn(
    model: MultiHeadTCNPowerRegressor,
    dataset: CausalWindowDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(dataset, batch_size, False, 0, device)
    prediction = np.empty((len(dataset.features), len(TARGET_COLS)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for features, _, _, _, row_index in loader:
            values = model(features.to(device, non_blocking=True))
            prediction[row_index.numpy()] = values.cpu().numpy()
    return np.clip(prediction, 0.0, 1.0)


def fold_metric(
    frame: pd.DataFrame, prediction_norm: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for group_index, group in enumerate(TARGET_COLS):
        actual = frame[group].to_numpy(float)
        valid = np.isfinite(actual)
        if not bool(valid.any()):
            continue
        capacity = GROUP_CAPACITY_KWH[group]
        prediction = prediction_norm[:, group_index] * capacity
        nmae, ficr = group_nmae_ficr(actual[valid], prediction[valid], capacity)
        rows.append(
            {
                "group": group,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
            }
        )
    if not rows:
        raise ValueError("Validation fold has no observable groups")
    mean_nmae = float(np.mean([row["nmae"] for row in rows]))
    mean_ficr = float(np.mean([row["ficr"] for row in rows]))
    return 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr, rows


def discover_epoch(
    fold: FoldData,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict]]:
    set_seed(seed)
    train_dataset, validation_dataset = prepare_tcn_data(fold, config)
    tcn_config = config["tcn"]
    loader = make_loader(
        train_dataset, int(tcn_config["batch_size"]), True, seed, device
    )
    model = make_model(len(fold.tcn_features), tcn_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcn_config["learning_rate"]),
        weight_decay=float(tcn_config["weight_decay"]),
    )
    best_score = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, int(tcn_config["max_epochs"]) + 1):
        train_loss = train_tcn_epoch(
            model, loader, optimizer, float(tcn_config["grad_clip"]), device
        )
        prediction = predict_tcn(
            model,
            validation_dataset,
            int(tcn_config["eval_batch_size"]),
            device,
        )
        score, group_rows = fold_metric(fold.validation, prediction)
        history.append(
            {
                "stage": "epoch_discovery",
                "pred_year": fold.pred_year,
                "epoch": epoch,
                "train_loss": train_loss,
                "score": score,
                "group_scores": ";".join(
                    f"{row['group']}={row['score']:.6f}" for row in group_rows
                ),
            }
        )
        if score > best_score + float(tcn_config["min_delta"]):
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"  discovery val={fold.pred_year} epoch={epoch:03d} "
            f"loss={train_loss:.5f} score={score:.6f} "
            f"best={best_score:.6f}@{best_epoch}",
            flush=True,
        )
        if bad_epochs >= int(tcn_config["patience"]):
            break
    if best_epoch <= 0:
        raise RuntimeError(f"No epoch selected for val={fold.pred_year}")
    del model, optimizer, loader, train_dataset, validation_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history


def fit_fixed_tcn(
    fold: FoldData,
    fixed_epoch: int,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    train_dataset, validation_dataset = prepare_tcn_data(fold, config)
    tcn_config = config["tcn"]
    loader = make_loader(
        train_dataset, int(tcn_config["batch_size"]), True, seed, device
    )
    model = make_model(len(fold.tcn_features), tcn_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcn_config["learning_rate"]),
        weight_decay=float(tcn_config["weight_decay"]),
    )
    train_loss = np.nan
    for epoch in range(1, fixed_epoch + 1):
        train_loss = train_tcn_epoch(
            model, loader, optimizer, float(tcn_config["grad_clip"]), device
        )
        print(
            f"  fixed val={fold.pred_year} epoch={epoch:03d}/{fixed_epoch:03d} "
            f"loss={train_loss:.5f}",
            flush=True,
        )
    prediction = predict_tcn(
        model,
        validation_dataset,
        int(tcn_config["eval_batch_size"]),
        device,
    )
    score, group_rows = fold_metric(fold.validation, prediction)
    stats = {
        "stage": "fixed_epoch",
        "pred_year": fold.pred_year,
        "fixed_epoch": fixed_epoch,
        "train_loss": train_loss,
        "score": score,
        "group_scores": ";".join(
            f"{row['group']}={row['score']:.6f}" for row in group_rows
        ),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "n_train_rows": len(train_dataset),
        "n_validation_rows": len(validation_dataset),
    }
    del model, optimizer, loader, train_dataset, validation_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats


def tree_params(row: pd.Series, objective: str, smoke_estimators: int = 0) -> dict:
    params = {name: row[name] for name in TREE_PARAM_COLUMNS}
    integer_columns = {
        "random_state",
        "n_jobs",
        "verbose",
        "n_estimators",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample_freq",
    }
    for name in integer_columns:
        params[name] = int(params[name])
    for name in set(TREE_PARAM_COLUMNS) - integer_columns - {"objective"}:
        params[name] = float(params[name])
    params["objective"] = objective
    if smoke_estimators:
        params["n_estimators"] = min(params["n_estimators"], smoke_estimators)
    return params


def sample_weight(values: np.ndarray, group: str, policy: str):
    if policy == "none":
        return None
    ratio = np.clip(values / GROUP_CAPACITY_KWH[group], 0.0, 1.0)
    if policy == "actual_sqrt":
        return 0.5 + np.sqrt(ratio)
    if policy == "metric_x2":
        return 1.0 + 2.0 * (ratio >= 0.10)
    raise ValueError(f"Unknown TREE weight policy: {policy}")


def fit_tree_fold(
    fold: FoldData, best: pd.DataFrame, config: dict
) -> pd.DataFrame:
    output_parts = []
    for group in TARGET_COLS:
        selected = best.loc[best["group"].eq(group)]
        if len(selected) != 1:
            raise ValueError(f"Expected one TREE config row for {group}")
        selected = selected.iloc[0]
        target_train = fold.train[group].to_numpy(float)
        target_validation = fold.validation[group].to_numpy(float)
        train_keep = np.isfinite(target_train) & (
            target_train
            >= GROUP_CAPACITY_KWH[group] * float(selected["min_output_ratio"])
        )
        validation_keep = np.isfinite(target_validation)
        if int(train_keep.sum()) < 500 or not bool(validation_keep.any()):
            continue
        part = pd.DataFrame(
            {
                "forecast_kst_dtm": pd.to_datetime(
                    fold.validation.loc[validation_keep, "forecast_kst_dtm"]
                ),
                "pred_year": fold.pred_year,
                "group": group,
                "actual": target_validation[validation_keep],
            }
        )
        for variant, (feature_set, objective) in TREE_VARIANTS.items():
            feature_cols = fold.group_features[group][feature_set]
            model = LGBMRegressor(
                **tree_params(
                    selected,
                    objective,
                    int(config.get("tree_smoke_estimators", 0)),
                )
            )
            model.fit(
                fold.train.loc[train_keep, feature_cols],
                target_train[train_keep],
                sample_weight=sample_weight(
                    target_train[train_keep], group, str(selected["weight_policy"])
                ),
            )
            part[variant] = np.clip(
                model.predict(fold.validation.loc[validation_keep, feature_cols]),
                0.0,
                GROUP_CAPACITY_KWH[group],
            )
        output_parts.append(part)
    if not output_parts:
        raise RuntimeError(f"No TREE predictions for val={fold.pred_year}")
    return pd.concat(output_parts, ignore_index=True)


def tcn_prediction_frame(fold: FoldData, prediction: np.ndarray) -> pd.DataFrame:
    parts = []
    for group_index, group in enumerate(TARGET_COLS):
        actual = fold.validation[group].to_numpy(float)
        keep = np.isfinite(actual)
        if not bool(keep.any()):
            continue
        parts.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(
                        fold.validation.loc[keep, "forecast_kst_dtm"]
                    ),
                    "pred_year": fold.pred_year,
                    "group": group,
                    "actual": actual[keep],
                    "tcn_multihead": prediction[keep, group_index]
                    * GROUP_CAPACITY_KWH[group],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def add_blends(oof: pd.DataFrame, shares: list[float]) -> pd.DataFrame:
    output = oof.copy()
    for tree_variant in TREE_VARIANTS:
        tree_name = tree_variant.removeprefix("tree_")
        for share in shares:
            tcn_percent = int(round(share * 100))
            name = f"blend_tcn{tcn_percent}_{tree_name}{100 - tcn_percent}"
            output[name] = (
                share * output["tcn_multihead"]
                + (1.0 - share) * output[tree_variant]
            )
    return output


def score_predictions(
    oof: pd.DataFrame, prediction_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    fold_rows = []
    for variant in prediction_cols:
        for group, part in oof.groupby("group", sort=False):
            nmae, ficr = group_nmae_ficr(
                part["actual"], part[variant], GROUP_CAPACITY_KWH[group]
            )
            group_rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n_rows": len(part),
                }
            )
        for (pred_year, group), part in oof.groupby(
            ["pred_year", "group"], sort=False
        ):
            nmae, ficr = group_nmae_ficr(
                part["actual"], part[variant], GROUP_CAPACITY_KWH[group]
            )
            fold_rows.append(
                {
                    "variant": variant,
                    "pred_year": pred_year,
                    "group": group,
                    "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n_rows": len(part),
                }
            )
    groups = pd.DataFrame(group_rows)
    folds = pd.DataFrame(fold_rows)
    summary = (
        groups.groupby("variant", as_index=False)
        .agg(
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_group_score=("score", "min"),
            std_group_score=("score", lambda values: values.std(ddof=0)),
            n_groups=("group", "nunique"),
        )
    )
    summary["mean_score"] = (
        0.5 * (1.0 - summary["mean_nmae"]) + 0.5 * summary["mean_ficr"]
    )
    fold_stability = (
        folds.groupby("variant", as_index=False)
        .agg(
            worst_fold=("score", "min"),
            std_fold=("score", lambda values: values.std(ddof=0)),
        )
    )
    summary = summary.merge(fold_stability, on="variant", how="left")
    return summary.sort_values("mean_score", ascending=False), pd.concat(
        [
            folds.assign(row_type="fold"),
            groups.assign(pred_year=np.nan, row_type="pooled_group"),
        ],
        ignore_index=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.smoke_test)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} smoke={args.smoke_test}", flush=True)
    ldaps, gfs, labels, scada_by_group = read_inputs()
    print("build deduplicated site inputs", flush=True)
    site_static = build_site_static(ldaps, gfs)
    tables_by_group = {}
    for group in TARGET_COLS:
        print(f"build group inputs: {group}", flush=True)
        tables_by_group[group] = build_group_tables(
            ldaps,
            gfs,
            labels,
            scada_by_group[group],
            group,
            include_residual=False,
        )
    folds = prepare_folds(
        labels, tables_by_group, site_static, args.max_outer_folds
    )
    print(
        f"TCN input features={len(folds[0].tcn_features)} "
        f"(site={len(SITE_FEATURES)}, group_specific="
        f"{len(TARGET_COLS) * len(GROUP_TCN_FEATURES)})",
        flush=True,
    )
    del ldaps, gfs, scada_by_group, tables_by_group, site_static
    gc.collect()

    epoch_rows = []
    best_epochs = []
    for fold_index, fold in enumerate(folds):
        print(f"\n=== TCN epoch discovery: val={fold.pred_year} ===", flush=True)
        best_epoch, history = discover_epoch(
            fold,
            config,
            device,
            int(config["seed"]) + fold_index * 1000,
        )
        best_epochs.append(best_epoch)
        epoch_rows.extend(history)
        epoch_rows.append(
            {
                "stage": "epoch_selected",
                "pred_year": fold.pred_year,
                "best_epoch": best_epoch,
            }
        )
    fixed_epoch = int(np.median(best_epochs))
    print(
        f"\nTCN fixed epoch = median({best_epochs}) = {fixed_epoch}", flush=True
    )

    tree_best = pd.read_csv(config["tree_best_path"], encoding="utf-8-sig")
    oof_parts = []
    for fold_index, fold in enumerate(folds):
        print(f"\n=== fixed OOF: val={fold.pred_year} ===", flush=True)
        tcn_prediction, stats = fit_fixed_tcn(
            fold,
            fixed_epoch,
            config,
            device,
            int(config["seed"]) + 10000 + fold_index * 1000,
        )
        epoch_rows.append(stats)
        tcn_frame = tcn_prediction_frame(fold, tcn_prediction)
        tree_frame = fit_tree_fold(fold, tree_best, config)
        combined = tcn_frame.merge(
            tree_frame,
            on=["forecast_kst_dtm", "pred_year", "group", "actual"],
            how="inner",
            validate="one_to_one",
        )
        oof_parts.append(combined)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["group", "forecast_kst_dtm"]
    )
    oof = add_blends(oof, [float(value) for value in config["tcn_shares"]])
    prediction_cols = [
        column
        for column in oof.columns
        if column not in {"forecast_kst_dtm", "pred_year", "group", "actual"}
    ]
    summary, score_diagnostics = score_predictions(oof, prediction_cols)
    epoch_diagnostics = pd.concat(
        [
            pd.DataFrame(epoch_rows).assign(row_type="training"),
            score_diagnostics,
        ],
        ignore_index=True,
    )

    oof_path = args.results_dir / f"{args.stem}_oof.csv"
    summary_path = args.results_dir / f"{args.stem}_summary.csv"
    diagnostics_path = args.results_dir / f"{args.stem}_epoch_diagnostics.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    epoch_diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved {oof_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
