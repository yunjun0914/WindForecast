from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_scada import clean_power_10m, turbine_capacity_kwh
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.scada_operating_envelope import (
    OperatingEnvelope,
    fit_operating_envelope,
    soft_operating_gate,
)


YEARS = [2022, 2023, 2024]
KEYS = ["forecast_kst_dtm", "group", "pred_year"]
MODEL_GROUPS = {
    "vestas": ["kpx_group_1", "kpx_group_2"],
    "unison": ["kpx_group_3"],
}
SCADA_FILE_NAMES = {
    "vestas": "scada_vestas_train.csv",
    "unison": "scada_unison_train.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCADA-derived model-specific cut-in/out gate OOF diagnostic."
    )
    parser.add_argument(
        "--pinn-group-oof",
        type=Path,
        default=Path("results/per_turbine_pinn_share50_pure_band_oof_v1_predictions.csv"),
    )
    parser.add_argument("--pinn-variant", default="baseline_retrained_floor20")
    parser.add_argument(
        "--baseline-ensemble-oof",
        type=Path,
        default=None,
        help="Optional already-blended OOF. When set, TREE/TCN files are unnecessary.",
    )
    parser.add_argument("--baseline-ensemble-variant", default="baseline_best")
    parser.add_argument(
        "--pinn-turbine-oof",
        type=Path,
        default=Path(
            "results/per_turbine_pinn_share50_pure_band_oof_v1_baseline_turbine_predictions.csv"
        ),
    )
    parser.add_argument(
        "--tree-oof",
        type=Path,
        default=Path("results/tree_scoreboost_group_year_tune_v1_predictions.csv"),
    )
    parser.add_argument("--tree-variant", default="scoreboost_group_year_tuned")
    parser.add_argument(
        "--tcn-oof",
        type=Path,
        default=Path("results/group_pure_band_tcn_oof_v1_predictions.csv"),
    )
    parser.add_argument("--tcn-variant", default="pure_band")
    parser.add_argument(
        "--teacher-cache-root", type=Path, default=Path("cache/per_turbine_teacher_v1")
    )
    parser.add_argument(
        "--teacher-cache-tag", default="optimal_grid_replace_local16_v1"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--active-power-ratio", type=float, default=0.01)
    parser.add_argument("--crossing-probability", type=float, default=0.50)
    parser.add_argument("--bin-width", type=float, default=0.50)
    parser.add_argument("--min-bin-count", type=int, default=30)
    parser.add_argument("--isolated-zero-peer-threshold", type=float, default=3.0)
    parser.add_argument("--tau-grid", default="0.5,1.0,1.5")
    parser.add_argument("--primary-tau", type=float, default=1.0)
    parser.add_argument("--pinn-floor", type=float, default=0.20)
    parser.add_argument("--pinn-weight", type=float, default=0.50)
    parser.add_argument("--tree-weight", type=float, default=0.05)
    parser.add_argument("--tcn-weight", type=float, default=0.45)
    parser.add_argument("--final-floor", type=float, default=0.10)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="pinn_scada_operating_gate_oof_v1")
    return parser.parse_args()


def _tau_values(value: str, primary_tau: float) -> list[float]:
    values = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if any(tau <= 0 for tau in values) or primary_tau <= 0:
        raise ValueError("All gate temperatures must be positive")
    if not any(abs(tau - primary_tau) <= 1e-12 for tau in values):
        values.append(float(primary_tau))
        values.sort()
    return values


def _tau_label(tau: float) -> str:
    return f"{tau:g}".replace(".", "p")


def _load_branch(path: Path, variant: str, pred_name: str) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    actual_col = "official_target" if "official_target" in table.columns else "actual"
    required = [*KEYS, "variant", "pred", actual_col]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    work = table.loc[table["variant"].eq(variant), required].copy()
    if work.empty:
        available = sorted(map(str, table["variant"].dropna().unique()))
        raise ValueError(f"{path} has no variant={variant}; available={available}")
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    if work.duplicated(KEYS).any():
        raise ValueError(f"{path} variant={variant} has duplicate keys")
    return work[[*KEYS, actual_col, "pred"]].rename(
        columns={actual_col: f"actual_{pred_name}", "pred": pred_name}
    )


def load_components(args: argparse.Namespace) -> pd.DataFrame:
    if args.baseline_ensemble_oof is not None:
        baseline = _load_branch(
            args.baseline_ensemble_oof,
            args.baseline_ensemble_variant,
            "baseline_pred",
        ).rename(columns={"actual_baseline_pred": "official_target"})
        table = baseline[[*KEYS, "official_target", "baseline_pred"]].copy()
    else:
        parts = [
            _load_branch(args.pinn_group_oof, args.pinn_variant, "pinn_saved"),
            _load_branch(args.tree_oof, args.tree_variant, "tree_pred"),
            _load_branch(args.tcn_oof, args.tcn_variant, "tcn_pred"),
        ]
        table = parts[0]
        for part in parts[1:]:
            table = table.merge(part, on=KEYS, how="inner", validate="one_to_one")
        if len(table) != len(parts[0]):
            raise ValueError("Baseline branches do not have identical OOF key coverage")

        actual_cols = ["actual_pinn_saved", "actual_tree_pred", "actual_tcn_pred"]
        reference = table[actual_cols[0]].to_numpy(float)
        for column in actual_cols[1:]:
            delta = float(np.nanmax(np.abs(reference - table[column].to_numpy(float))))
            if delta > 0.02 * max(GROUP_CAPACITY_KWH.values()):
                raise ValueError(f"Baseline target mismatch {column}: max_delta={delta}")
            if delta > 1e-2:
                print(f"warning: {column} target max_delta={delta:.6f}", flush=True)
        table["official_target"] = reference

    labels_path = args.labels or (args.data_root / "train_labels.csv")
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    label_long = labels.melt(
        id_vars="kst_dtm",
        value_vars=TARGET_COLS,
        var_name="group",
        value_name="label_target",
    ).rename(columns={"kst_dtm": "forecast_kst_dtm"})
    checked = table.merge(
        label_long, on=["forecast_kst_dtm", "group"], how="left", validate="many_to_one"
    )
    comparable = checked["label_target"].notna() & checked["official_target"].notna()
    label_delta = float(
        np.max(
            np.abs(
                checked.loc[comparable, "label_target"].to_numpy(float)
                - checked.loc[comparable, "official_target"].to_numpy(float)
            )
        )
    )
    if label_delta > 1e-3:
        raise ValueError(f"PINN OOF target does not match train_labels: {label_delta}")
    component_cols = [
        column
        for column in ["pinn_saved", "tree_pred", "tcn_pred", "baseline_pred"]
        if column in table.columns
    ]
    return table[[*KEYS, "official_target", *component_cols]]


def _model_scada_samples(
    scada: pd.DataFrame,
    model: str,
    train_years: list[int],
    peer_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    source = scada.copy()
    source["kst_dtm"] = pd.to_datetime(source["kst_dtm"])
    source = source.loc[source["kst_dtm"].dt.year.isin(train_years)].reset_index(drop=True)
    wind_parts: list[np.ndarray] = []
    power_parts: list[np.ndarray] = []
    masked_isolated = 0
    candidate_rows = 0
    for group in MODEL_GROUPS[model]:
        turbines = GROUP_TURBINE_PREFIXES[group]
        wind_table = pd.DataFrame(
            {
                turbine: pd.to_numeric(source[f"{turbine}_ws"], errors="coerce")
                for turbine in turbines
            }
        )
        for turbine in turbines:
            peers = wind_table.drop(columns=turbine).median(axis=1, skipna=True)
            wind = wind_table[turbine]
            isolated_zero = wind.le(0.1) & peers.ge(peer_threshold)
            power = clean_power_10m(source[f"{turbine}_power_kw10m"], group)
            rated_10m = turbine_capacity_kwh(group) / 6.0
            valid = wind.between(0.0, 35.0) & power.notna() & ~isolated_zero
            wind_parts.append(wind.loc[valid].to_numpy(float))
            power_parts.append((power.loc[valid] / rated_10m).to_numpy(float))
            masked_isolated += int(isolated_zero.sum())
            candidate_rows += len(source)
    return (
        np.concatenate(wind_parts),
        np.concatenate(power_parts),
        {
            "candidate_rows": int(candidate_rows),
            "masked_isolated_zero": int(masked_isolated),
        },
    )


def fit_fold_envelopes(
    args: argparse.Namespace,
    pred_years: list[int],
) -> tuple[dict[tuple[str, int], OperatingEnvelope], pd.DataFrame, pd.DataFrame]:
    scada_by_model = {
        model: pd.read_csv(args.data_root / file_name, encoding="utf-8-sig")
        for model, file_name in SCADA_FILE_NAMES.items()
    }
    envelopes: dict[tuple[str, int], OperatingEnvelope] = {}
    threshold_rows = []
    curve_rows = []
    for pred_year in pred_years:
        train_years = [year for year in YEARS if year != pred_year]
        for model, scada in scada_by_model.items():
            wind, power_ratio, sample_stats = _model_scada_samples(
                scada,
                model,
                train_years,
                args.isolated_zero_peer_threshold,
            )
            envelope = fit_operating_envelope(
                wind,
                power_ratio,
                bin_width=args.bin_width,
                min_bin_count=args.min_bin_count,
                active_power_ratio=args.active_power_ratio,
                crossing_probability=args.crossing_probability,
            )
            envelopes[(model, pred_year)] = envelope
            threshold_rows.append(
                {
                    "model": model,
                    "pred_year": pred_year,
                    "train_years": ",".join(map(str, train_years)),
                    "cut_in_speed": envelope.cut_in_speed,
                    "cut_out_speed": envelope.cut_out_speed,
                    "cut_out_detected": envelope.cut_out_detected,
                    "n_observations": envelope.n_observations,
                    "n_high_wind": envelope.n_high_wind,
                    "active_power_ratio": envelope.active_power_ratio,
                    "crossing_probability": envelope.crossing_probability,
                    **sample_stats,
                }
            )
            for index, center in enumerate(envelope.bin_centers):
                curve_rows.append(
                    {
                        "model": model,
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "wind_bin": center,
                        "bin_count": envelope.bin_counts[index],
                        "active_probability": envelope.active_probability[index],
                        "lower_probability": envelope.lower_probability[index],
                        "upper_probability": envelope.upper_probability[index],
                    }
                )
            cut_out_text = (
                f"{envelope.cut_out_speed:.3f}"
                if envelope.cut_out_detected
                else "undetected"
            )
            print(
                f"{model} pred_year={pred_year} train={train_years} "
                f"cut_in={envelope.cut_in_speed:.3f} cut_out={cut_out_text} "
                f"n={envelope.n_observations} high={envelope.n_high_wind}",
                flush=True,
            )
    return envelopes, pd.DataFrame(threshold_rows), pd.DataFrame(curve_rows)


def load_turbine_predictions_and_wind(
    args: argparse.Namespace,
    envelopes: dict[tuple[str, int], OperatingEnvelope],
) -> pd.DataFrame:
    turbines = pd.read_csv(args.pinn_turbine_oof, encoding="utf-8-sig")
    required = ["forecast_kst_dtm", "turbine_id", "pred", "group", "pred_year"]
    missing = [column for column in required if column not in turbines.columns]
    if missing:
        raise ValueError(f"{args.pinn_turbine_oof} missing columns: {missing}")
    turbines["forecast_kst_dtm"] = pd.to_datetime(turbines["forecast_kst_dtm"])
    if turbines.duplicated(["forecast_kst_dtm", "turbine_id", "pred_year"]).any():
        raise ValueError("Turbine OOF predictions have duplicate keys")

    wind_parts = []
    for (group, pred_year), _ in turbines.groupby(["group", "pred_year"], sort=False):
        cache_path = (
            args.teacher_cache_root
            / f"{group}_pred{int(pred_year)}_{args.teacher_cache_tag}.pkl"
        )
        teacher = pd.read_pickle(cache_path)
        teacher["forecast_kst_dtm"] = pd.to_datetime(teacher["forecast_kst_dtm"])
        if "split" in teacher.columns:
            teacher = teacher.loc[teacher["split"].eq("validation")]
        teacher = teacher.loc[
            teacher["forecast_kst_dtm"].dt.year.eq(int(pred_year)),
            ["forecast_kst_dtm", "turbine_id", "teacher_ws_cubic"],
        ].copy()
        if teacher.duplicated(["forecast_kst_dtm", "turbine_id"]).any():
            raise ValueError(f"Teacher cache has duplicate keys: {cache_path}")
        teacher["group"] = group
        teacher["pred_year"] = int(pred_year)
        wind_parts.append(teacher)
    wind = pd.concat(wind_parts, ignore_index=True)
    merged = turbines.merge(
        wind,
        on=["forecast_kst_dtm", "turbine_id", "group", "pred_year"],
        how="left",
        validate="one_to_one",
    )
    if merged["teacher_ws_cubic"].isna().any():
        raise ValueError(
            f"Missing teacher wind for {int(merged['teacher_ws_cubic'].isna().sum())} turbine rows"
        )
    group_to_model = {
        group: model for model, groups in MODEL_GROUPS.items() for group in groups
    }
    merged["model"] = merged["group"].map(group_to_model)
    merged["cut_in_speed"] = [
        envelopes[(model, int(pred_year))].cut_in_speed
        for model, pred_year in zip(merged["model"], merged["pred_year"])
    ]
    merged["cut_out_speed"] = [
        envelopes[(model, int(pred_year))].cut_out_speed
        for model, pred_year in zip(merged["model"], merged["pred_year"])
    ]
    merged["turbine_capacity"] = merged["group"].map(turbine_capacity_kwh).astype(float)
    return merged


def _aggregate_turbine_prediction(
    turbines: pd.DataFrame,
    values: np.ndarray,
) -> pd.DataFrame:
    work = turbines[["forecast_kst_dtm", "group", "pred_year", "turbine_id"]].copy()
    work["pred"] = np.asarray(values, dtype=float)
    expected = {group: len(turbines_) for group, turbines_ in GROUP_TURBINE_PREFIXES.items()}
    grouped = (
        work.groupby(KEYS, as_index=False)
        .agg(pred=("pred", "sum"), n_turbines=("turbine_id", "nunique"))
    )
    expected_count = grouped["group"].map(expected)
    if not grouped["n_turbines"].eq(expected_count).all():
        raise ValueError("Incomplete turbine aggregation")
    capacity = grouped["group"].map(GROUP_CAPACITY_KWH).astype(float)
    grouped["pred"] = np.clip(grouped["pred"], 0.0, capacity)
    return grouped[[*KEYS, "pred"]]


def _attach_gate(
    turbines: pd.DataFrame,
    tau: float,
    include_cut_in: bool,
    include_cut_out: bool,
) -> np.ndarray:
    gates = np.empty(len(turbines), dtype=float)
    for (_, cut_in, cut_out), indices in turbines.groupby(
        ["model", "cut_in_speed", "cut_out_speed"], dropna=False
    ).groups.items():
        index = np.asarray(list(indices), dtype=int)
        gates[index] = soft_operating_gate(
            turbines.loc[index, "teacher_ws_cubic"].to_numpy(float),
            float(cut_in),
            float(cut_out),
            tau_in=tau,
            tau_out=tau,
            include_cut_in=include_cut_in,
            include_cut_out=include_cut_out,
        )
    return gates


def _group_variant(
    components: pd.DataFrame,
    group_pred: pd.DataFrame,
    variant: str,
    *,
    ensemble: bool,
    args: argparse.Namespace,
) -> pd.DataFrame:
    merged = components.merge(
        group_pred.rename(columns={"pred": "candidate_pinn"}),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if merged["candidate_pinn"].isna().any():
        raise ValueError(f"Missing gated PINN predictions for {variant}")
    capacity = merged["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
    if ensemble:
        if "baseline_pred" in merged.columns:
            pred = merged["baseline_pred"].to_numpy(float) + args.pinn_weight * (
                merged["candidate_pinn"].to_numpy(float)
                - merged["pinn_reference"].to_numpy(float)
            )
        else:
            pred = (
                args.pinn_weight * merged["candidate_pinn"].to_numpy(float)
                + args.tree_weight * merged["tree_pred"].to_numpy(float)
                + args.tcn_weight * merged["tcn_pred"].to_numpy(float)
            )
        pred = np.clip(pred, args.final_floor * capacity, capacity)
    else:
        pred = np.clip(merged["candidate_pinn"].to_numpy(float), 0.0, capacity)
    return pd.DataFrame(
        {
            "forecast_kst_dtm": merged["forecast_kst_dtm"],
            "group": merged["group"],
            "pred_year": merged["pred_year"],
            "official_target": merged["official_target"],
            "pred": pred,
            "variant": variant,
        }
    )


def _fold_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, group, pred_year), part in predictions.groupby(
        ["variant", "group", "pred_year"], sort=False
    ):
        nmae, ficr = group_nmae_ficr(
            part["official_target"], part["pred"], GROUP_CAPACITY_KWH[group]
        )
        rows.append(
            {
                "variant": variant,
                "group": group,
                "pred_year": int(pred_year),
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": len(part),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    weights = args.pinn_weight + args.tree_weight + args.tcn_weight
    if abs(weights - 1.0) > 1e-12:
        raise ValueError(f"Ensemble weights must sum to 1, got {weights}")
    if not 0.0 <= args.pinn_floor < 1.0 or not 0.0 <= args.final_floor < 1.0:
        raise ValueError("Floors must be in [0, 1)")
    tau_values = _tau_values(args.tau_grid, args.primary_tau)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    components = load_components(args)
    pred_years = sorted(map(int, components["pred_year"].unique()))
    envelopes, thresholds, curves = fit_fold_envelopes(args, pred_years)
    turbines = load_turbine_predictions_and_wind(args, envelopes)

    raw = turbines["pred"].to_numpy(float)
    turbine_capacity = turbines["turbine_capacity"].to_numpy(float)
    floor_value = args.pinn_floor * turbine_capacity
    baseline_turbine = np.clip(raw, floor_value, turbine_capacity)
    baseline_group = _aggregate_turbine_prediction(turbines, baseline_turbine)
    if "pinn_saved" in components.columns:
        reconstructed = baseline_group.merge(
            components[[*KEYS, "pinn_saved"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        reconstruction_delta = float(
            np.max(np.abs(reconstructed["pred"] - reconstructed["pinn_saved"]))
        )
        if reconstruction_delta > 0.1:
            raise ValueError(
                "Could not reconstruct saved PINN floor20 prediction: "
                f"max_delta={reconstruction_delta}"
            )
        print(f"PINN floor reconstruction max_delta={reconstruction_delta:.6f}", flush=True)
    components = components.merge(
        baseline_group.rename(columns={"pred": "pinn_reference"}),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    if "baseline_pred" in components.columns:
        other_contribution = (
            components["baseline_pred"]
            - args.pinn_weight * components["pinn_reference"]
        )
        if float(other_contribution.min()) < -0.1:
            raise ValueError(
                "The supplied baseline ensemble is inconsistent with the reconstructed PINN"
            )

    prediction_parts = [
        _group_variant(
            components,
            baseline_group,
            "pinn_baseline_floor20",
            ensemble=False,
            args=args,
        ),
        _group_variant(
            components,
            baseline_group,
            "ensemble_baseline_50_5_45",
            ensemble=True,
            args=args,
        ),
    ]
    diagnostic_rows = []
    primary_variants = {"pinn_baseline_floor20", "ensemble_baseline_50_5_45"}
    for tau in tau_values:
        label = _tau_label(tau)
        both_gate = _attach_gate(turbines, tau, True, True)
        after_floor = baseline_turbine * both_gate
        before_floor = np.clip(raw * both_gate, floor_value, turbine_capacity)
        candidates = [
            ("both_after_floor", after_floor),
            ("both_before_floor", before_floor),
            (
                "cutin_after_floor",
                baseline_turbine * _attach_gate(turbines, tau, True, False),
            ),
            (
                "cutout_after_floor",
                baseline_turbine * _attach_gate(turbines, tau, False, True),
            ),
        ]
        for gate_name, turbine_pred in candidates:
            group_pred = _aggregate_turbine_prediction(turbines, turbine_pred)
            pinn_variant = f"pinn_{gate_name}_tau{label}"
            ensemble_variant = f"ensemble_{gate_name}_tau{label}"
            prediction_parts.append(
                _group_variant(
                    components, group_pred, pinn_variant, ensemble=False, args=args
                )
            )
            prediction_parts.append(
                _group_variant(
                    components, group_pred, ensemble_variant, ensemble=True, args=args
                )
            )
            if abs(tau - args.primary_tau) <= 1e-12 and gate_name == "both_after_floor":
                primary_variants.update({pinn_variant, ensemble_variant})

        for (group, pred_year), indices in turbines.groupby(["group", "pred_year"]).groups.items():
            index = np.asarray(list(indices), dtype=int)
            group_gate = both_gate[index]
            wind = turbines.loc[index, "teacher_ws_cubic"].to_numpy(float)
            cut_out = turbines.loc[index, "cut_out_speed"].iloc[0]
            diagnostic_rows.append(
                {
                    "group": group,
                    "pred_year": int(pred_year),
                    "tau": tau,
                    "n_turbine_rows": len(index),
                    "wind_min": float(np.min(wind)),
                    "wind_p01": float(np.quantile(wind, 0.01)),
                    "wind_p99": float(np.quantile(wind, 0.99)),
                    "wind_max": float(np.max(wind)),
                    "cut_out_speed": cut_out,
                    "n_forecast_at_or_above_cutout": int(
                        np.sum(wind >= cut_out) if np.isfinite(cut_out) else 0
                    ),
                    "fraction_gate_below_099": float(np.mean(group_gate < 0.99)),
                    "fraction_gate_below_090": float(np.mean(group_gate < 0.90)),
                    "mean_gate": float(np.mean(group_gate)),
                }
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    summary, pooled_groups = pooled_oof_summary(predictions)
    folds = _fold_scores(predictions)
    fold_diag = folds.groupby("variant", as_index=False).agg(
        worst_group_year=("score", "min"),
        std_group_year=("score", lambda values: values.std(ddof=0)),
        n_group_years=("score", "count"),
    )
    summary = summary.merge(fold_diag, on="variant", how="left")
    baseline_score = float(
        summary.loc[
            summary["variant"].eq("ensemble_baseline_50_5_45"), "mean_score"
        ].iloc[0]
    )
    summary["delta_vs_baseline_ensemble"] = summary["mean_score"] - baseline_score
    summary = summary.sort_values("mean_score", ascending=False).reset_index(drop=True)

    prefix = args.results_dir / args.stem
    thresholds.to_csv(f"{prefix}_thresholds.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(f"{prefix}_curve_bins.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostic_rows).to_csv(
        f"{prefix}_gate_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(f"{prefix}_fold_scores.csv", index=False, encoding="utf-8-sig")
    pooled_groups.to_csv(
        f"{prefix}_pooled_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    predictions.loc[predictions["variant"].isin(primary_variants)].to_csv(
        f"{prefix}_primary_predictions.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== SCADA operating gate pooled OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
