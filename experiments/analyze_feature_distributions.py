from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import PowerTransformer

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.preprocessing import (
    FARM_CENTROID,
    add_gfs_derived_features,
    add_ldaps_derived_features,
    nearest_gfs_grid_id,
)


TIME_COLUMNS = {"forecast_kst_dtm", "data_available_kst_dtm", "kst_dtm"}
GRID_COLUMNS = {"grid_id", "latitude", "longitude"}
TRANSFORM_NAMES = ("raw", "signed_sqrt", "signed_log1p", "yeo_johnson")
TRANSFORM_COLORS = {
    "raw": "#777777",
    "signed_sqrt": "#277da1",
    "signed_log1p": "#f3722c",
    "yeo_johnson": "#43aa8b",
}
FAMILY_COLORS = {
    "LDAPS": "#e76f51",
    "GFS": "#457b9d",
    "SCADA Vestas": "#8e7dbe",
    "SCADA Unison": "#52b788",
    "Target": "#222222",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ldaps", type=Path, default=Path("data/train/ldaps_train.csv")
    )
    parser.add_argument(
        "--gfs", type=Path, default=Path("data/train/gfs_train.csv")
    )
    parser.add_argument(
        "--scada-vestas",
        type=Path,
        default=Path("data/train/scada_vestas_train.csv"),
    )
    parser.add_argument(
        "--scada-unison",
        type=Path,
        default=Path("data/train/scada_unison_train.csv"),
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/train/train_labels.csv")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/feature_distribution_transform_summary.csv"),
    )
    parser.add_argument(
        "--overview-output",
        type=Path,
        default=Path("results/feature_distribution_transform_overview.png"),
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=Path("results/feature_distribution_transform_examples.png"),
    )
    return parser.parse_args()


def numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def aggregate_weather(path: Path, family: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    value_cols = [
        col
        for col in frame.columns
        if col not in TIME_COLUMNS | GRID_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if family == "LDAPS":
        hourly = frame.groupby("forecast_kst_dtm", as_index=False)[value_cols].mean()
        hourly = add_ldaps_derived_features(hourly)
    else:
        grid_id = nearest_gfs_grid_id(
            frame, FARM_CENTROID[0], FARM_CENTROID[1]
        )
        hourly = frame.loc[frame["grid_id"].eq(grid_id), ["forecast_kst_dtm", *value_cols]].copy()
        hourly = add_gfs_derived_features(hourly)
    return hourly


def aggregate_scada(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["kst_dtm"]).dt.floor("h")
    value_cols = [col for col in frame.columns if col not in TIME_COLUMNS]
    direction_cols = [col for col in value_cols if col.endswith("_wd")]
    regular_cols = [col for col in value_cols if col not in direction_cols]
    hourly = frame.groupby("forecast_kst_dtm", as_index=False)[regular_cols].mean()

    for col in direction_cols:
        radians = np.deg2rad(pd.to_numeric(frame[col], errors="coerce") % 360.0)
        sin_mean = pd.Series(np.sin(radians)).groupby(frame["forecast_kst_dtm"]).mean()
        cos_mean = pd.Series(np.cos(radians)).groupby(frame["forecast_kst_dtm"]).mean()
        angles = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0
        hourly[col] = hourly["forecast_kst_dtm"].map(angles)
    return hourly


def collect_features(args: argparse.Namespace) -> dict[str, tuple[str, np.ndarray]]:
    features: dict[str, tuple[str, np.ndarray]] = {}
    sources = (
        ("LDAPS", aggregate_weather(args.ldaps, "LDAPS")),
        ("GFS", aggregate_weather(args.gfs, "GFS")),
        ("SCADA Vestas", aggregate_scada(args.scada_vestas)),
        ("SCADA Unison", aggregate_scada(args.scada_unison)),
    )
    for family, frame in sources:
        for col in frame.columns:
            if col in TIME_COLUMNS:
                continue
            key = f"{family.lower().replace(' ', '_')}::{col}"
            features[key] = (family, numeric_values(frame[col]))

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    for target in TARGET_COLS:
        ratio = pd.to_numeric(labels[target], errors="coerce") / GROUP_CAPACITY_KWH[target]
        features[f"target::{target}_capacity_ratio"] = (
            "Target",
            numeric_values(ratio),
        )
    return features


def signed_sqrt(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.sqrt(np.abs(values))


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def transform_variants(values: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    variants = {
        "raw": values,
        "signed_sqrt": signed_sqrt(values),
        "signed_log1p": signed_log1p(values),
    }
    power = PowerTransformer(method="yeo-johnson", standardize=False)
    try:
        transformed = power.fit_transform(values.reshape(-1, 1)).ravel()
        if np.isfinite(transformed).all():
            variants["yeo_johnson"] = transformed
            power_lambda = float(power.lambdas_[0])
        else:
            variants["yeo_johnson"] = values
            power_lambda = float("nan")
    except (ValueError, FloatingPointError, OverflowError):
        variants["yeo_johnson"] = values
        power_lambda = float("nan")
    return variants, power_lambda


def normality_metrics(values: np.ndarray) -> dict[str, float]:
    std = float(np.std(values))
    if len(values) < 20 or std <= 1e-12:
        return {"skewness": np.nan, "excess_kurtosis": np.nan, "qq_rmse": np.nan}
    standardized = (values - float(np.mean(values))) / std
    probabilities = np.linspace(0.01, 0.99, 99)
    empirical = np.quantile(standardized, probabilities)
    theoretical = norm.ppf(probabilities)
    return {
        "skewness": float(pd.Series(standardized).skew()),
        "excess_kurtosis": float(pd.Series(standardized).kurt()),
        "qq_rmse": float(np.sqrt(np.mean((empirical - theoretical) ** 2))),
    }


def feature_kind(name: str, values: np.ndarray) -> str:
    if name.endswith("_wd"):
        return "circular"
    unique_count = int(len(np.unique(values)))
    if unique_count <= 1:
        return "constant"
    if unique_count <= 20:
        return "discrete"
    zero_fraction = float(np.mean(np.isclose(values, 0.0)))
    if zero_fraction >= 0.20:
        return "zero_inflated"
    return "continuous"


def analyze_feature(name: str, family: str, values: np.ndarray) -> dict[str, object]:
    kind = feature_kind(name, values)
    zero_fraction = float(np.mean(np.isclose(values, 0.0))) if len(values) else np.nan
    row: dict[str, object] = {
        "feature": name,
        "family": family,
        "kind": kind,
        "n": int(len(values)),
        "unique": int(len(np.unique(values))),
        "zero_fraction": zero_fraction,
        "minimum": float(np.min(values)),
        "p50": float(np.median(values)),
        "maximum": float(np.max(values)),
    }

    if kind == "circular":
        row.update(
            {
                "analysis_scope": "circular",
                "recommended_transform": "sin_cos",
                "yeo_johnson_lambda": np.nan,
                "best_qq_rmse": np.nan,
                "qq_improvement": np.nan,
            }
        )
        return row
    if kind in {"constant", "discrete"}:
        row.update(
            {
                "analysis_scope": kind,
                "recommended_transform": "raw_or_categorical",
                "yeo_johnson_lambda": np.nan,
                "best_qq_rmse": np.nan,
                "qq_improvement": np.nan,
            }
        )
        return row

    analysis_values = values[~np.isclose(values, 0.0)] if kind == "zero_inflated" else values
    variants, power_lambda = transform_variants(analysis_values)
    metrics = {name: normality_metrics(array) for name, array in variants.items()}
    for transform_name in TRANSFORM_NAMES:
        for metric_name, metric_value in metrics[transform_name].items():
            row[f"{transform_name}_{metric_name}"] = metric_value
    finite_scores = {
        transform_name: values_["qq_rmse"]
        for transform_name, values_ in metrics.items()
        if np.isfinite(values_["qq_rmse"])
    }
    best_transform = min(finite_scores, key=finite_scores.get)
    prefix = "zero_indicator+" if kind == "zero_inflated" else ""
    raw_score = metrics["raw"]["qq_rmse"]
    best_score = metrics[best_transform]["qq_rmse"]
    row.update(
        {
            "analysis_scope": "positive_only" if kind == "zero_inflated" else "all",
            "recommended_transform": f"{prefix}{best_transform}",
            "yeo_johnson_lambda": power_lambda,
            "best_qq_rmse": best_score,
            "qq_improvement": raw_score - best_score,
        }
    )
    return row


def short_label(value: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(value.replace("::", " | "), width=width))


def draw_overview(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    recommendation_counts = (
        summary.groupby(["family", "recommended_transform"])
        .size()
        .unstack(fill_value=0)
    )
    recommendation_counts.plot(kind="bar", stacked=True, ax=axes[0, 0], colormap="tab20")
    axes[0, 0].set_title("Recommended representation counts")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Number of features")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(fontsize=8, frameon=False)

    eligible = summary.dropna(subset=["raw_qq_rmse", "best_qq_rmse"])
    for family, part in eligible.groupby("family"):
        axes[0, 1].scatter(
            part["raw_qq_rmse"],
            part["best_qq_rmse"],
            s=30,
            alpha=0.7,
            color=FAMILY_COLORS[family],
            label=family,
        )
    limit = float(max(eligible["raw_qq_rmse"].max(), eligible["best_qq_rmse"].max()))
    axes[0, 1].plot([0, limit], [0, limit], "--", color="#555555", linewidth=1.0)
    axes[0, 1].set_title("Normal Q-Q error before and after transformation")
    axes[0, 1].set_xlabel("Raw Q-Q RMSE")
    axes[0, 1].set_ylabel("Best transformed Q-Q RMSE")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    top = eligible.nlargest(15, "qq_improvement").sort_values("qq_improvement")
    axes[1, 0].barh(
        [short_label(value, 28) for value in top["feature"]],
        top["qq_improvement"],
        color=[FAMILY_COLORS[value] for value in top["family"]],
    )
    axes[1, 0].set_title("Largest reductions in normal Q-Q error")
    axes[1, 0].set_xlabel("Raw minus transformed Q-Q RMSE")
    axes[1, 0].tick_params(axis="y", labelsize=8)
    axes[1, 0].grid(axis="x", alpha=0.2)

    target = summary.loc[summary["family"].eq("Target")]
    x = np.arange(len(target))
    width = 0.18
    for index, transform_name in enumerate(TRANSFORM_NAMES):
        axes[1, 1].bar(
            x + (index - 1.5) * width,
            target[f"{transform_name}_qq_rmse"],
            width,
            label=transform_name,
            color=TRANSFORM_COLORS[transform_name],
        )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        [value.replace("target::", "") for value in target["feature"]], rotation=12
    )
    axes[1, 1].set_title("Target normal Q-Q error")
    axes[1, 1].set_ylabel("Q-Q RMSE (lower is closer to normal)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle("Distribution audit across weather, SCADA, and targets", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def recommended_base_transform(recommendation: str) -> str:
    return recommendation.replace("zero_indicator+", "")


def draw_examples(
    features: dict[str, tuple[str, np.ndarray]], summary: pd.DataFrame, output: Path
) -> None:
    targets = summary.loc[summary["family"].eq("Target"), "feature"].tolist()
    candidates = summary.loc[
        summary["family"].ne("Target")
        & summary["kind"].isin(["continuous", "zero_inflated"])
    ].nlargest(9, "qq_improvement")
    selected = targets + candidates["feature"].tolist()
    bins = np.arange(-4.0, 5.05, 0.1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    normal_density = norm.pdf(centers)
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), sharex=True, sharey=True)

    indexed = summary.set_index("feature")
    for ax, name in zip(axes.flat, selected):
        values = features[name][1]
        row = indexed.loc[name]
        if row["kind"] == "zero_inflated":
            values = values[~np.isclose(values, 0.0)]
        variants, _ = transform_variants(values)
        transform_name = recommended_base_transform(str(row["recommended_transform"]))
        for label in ("raw", transform_name):
            transformed = variants[label]
            z_values = (transformed - np.mean(transformed)) / np.std(transformed)
            density, _ = np.histogram(z_values, bins=bins, density=True)
            ax.stairs(
                density,
                bins,
                color=TRANSFORM_COLORS[label],
                linewidth=1.8,
                label=label,
            )
        ax.plot(centers, normal_density, "--", color="#111111", linewidth=1.1, label="normal")
        ax.set_title(short_label(name, 38), fontsize=9)
        ax.set_xlim(-3.5, 4.5)
        ax.set_ylim(0.0, 0.85)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.flat[len(selected) :]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Raw distributions and their recommended transforms", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Zero-inflated features show their positive component; targets are normalized by group capacity.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    features = collect_features(args)
    rows = [
        analyze_feature(name, family, values)
        for name, (family, values) in features.items()
        if len(values) > 0
    ]
    summary = pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    draw_overview(summary, args.overview_output)
    draw_examples(features, summary, args.examples_output)

    print(f"features analyzed: {len(summary)}")
    print("\n=== recommendation counts ===")
    print(summary.groupby(["family", "recommended_transform"]).size().to_string())
    print("\n=== targets ===")
    print(
        summary.loc[
            summary["family"].eq("Target"),
            [
                "feature",
                "kind",
                "zero_fraction",
                "recommended_transform",
                "raw_qq_rmse",
                "best_qq_rmse",
                "qq_improvement",
            ],
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"saved summary: {args.summary_output}")
    print(f"saved overview: {args.overview_output}")
    print(f"saved examples: {args.examples_output}")


if __name__ == "__main__":
    main()
