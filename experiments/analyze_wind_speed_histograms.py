from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import (
    FARM_CENTROID,
    add_gfs_derived_features,
    add_ldaps_derived_features,
    aggregate_gfs_grids,
    aggregate_ldaps_grids,
    nearest_gfs_grid_id,
)


GROUPS = tuple(GROUP_TURBINE_PREFIXES)
SOURCE_LABELS = {
    "scada_ws": "SCADA",
    "ldaps_ws10": "LDAPS 10m",
    "ldaps_ws50_max": "LDAPS 50m max proxy",
    "gfs_ws10": "GFS 10m",
    "gfs_ws80": "GFS 80m",
    "gfs_ws100": "GFS 100m",
}
SOURCE_COLORS = {
    "scada_ws": "#202124",
    "ldaps_ws10": "#e76f51",
    "ldaps_ws50_max": "#d1495b",
    "gfs_ws10": "#2a9d8f",
    "gfs_ws80": "#457b9d",
    "gfs_ws100": "#6d597a",
}
TRANSFORMS = {
    "raw": lambda values: values,
    "sqrt": np.sqrt,
    "log1p": np.log1p,
}
TRANSFORM_COLORS = {
    "raw": "#777777",
    "sqrt": "#277da1",
    "log1p": "#f3722c",
}
LOW_LEVEL_SOURCES = ("scada_ws", "ldaps_ws10", "gfs_ws10")
HUB_LEVEL_SOURCES = (
    "scada_ws",
    "ldaps_ws50_max",
    "gfs_ws80",
    "gfs_ws100",
)


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
        "--figure-output",
        type=Path,
        default=Path("results/wind_speed_distribution_histograms.png"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/wind_speed_distribution_summary.csv"),
    )
    parser.add_argument(
        "--transform-figure-output",
        type=Path,
        default=Path("results/wind_speed_distribution_transforms.png"),
    )
    parser.add_argument(
        "--transform-summary-output",
        type=Path,
        default=Path("results/wind_speed_distribution_transform_summary.csv"),
    )
    return parser.parse_args()


def hourly_group_scada(scada: pd.DataFrame, group: str) -> pd.DataFrame:
    frame = scada.copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["kst_dtm"]).dt.floor("h")
    ws_cols = [f"{prefix}_ws" for prefix in GROUP_TURBINE_PREFIXES[group]]
    hourly_by_turbine = frame.groupby("forecast_kst_dtm", as_index=False)[ws_cols].mean()
    hourly_by_turbine["scada_ws"] = hourly_by_turbine[ws_cols].median(
        axis=1, skipna=True
    )
    return hourly_by_turbine[["forecast_kst_dtm", "scada_ws"]]


def build_weather(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    ldaps_hourly = add_ldaps_derived_features(aggregate_ldaps_grids(ldaps))
    gfs_hourly = add_gfs_derived_features(
        aggregate_gfs_grids(gfs, FARM_CENTROID[0], FARM_CENTROID[1])
    )

    ldaps_hourly = ldaps_hourly.rename(
        columns={
            "ws10_speed": "ldaps_ws10",
            "ws50_max_speed": "ldaps_ws50_max",
        }
    )
    gfs_hourly = gfs_hourly.rename(
        columns={
            "ws10_speed": "gfs_ws10",
            "ws80_speed": "gfs_ws80",
            "ws100_speed": "gfs_ws100",
        }
    )
    keep_ldaps = ["forecast_kst_dtm", "ldaps_ws10", "ldaps_ws50_max"]
    keep_gfs = ["forecast_kst_dtm", "gfs_ws10", "gfs_ws80", "gfs_ws100"]
    weather = ldaps_hourly[keep_ldaps].merge(
        gfs_hourly[keep_gfs], on="forecast_kst_dtm", how="inner"
    )
    weather["forecast_kst_dtm"] = pd.to_datetime(weather["forecast_kst_dtm"])
    return weather


def clean_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values >= 0.0) & (values <= 50.0)]


def summarize(group: str, source: str, values: np.ndarray) -> dict[str, float | int | str]:
    return {
        "group": group,
        "source": SOURCE_LABELS[source],
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "share_lt_5": float(np.mean(values < 5.0)),
        "share_5_to_10": float(np.mean((values >= 5.0) & (values < 10.0))),
        "share_ge_10": float(np.mean(values >= 10.0)),
    }


def histogram_percent(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    return counts / max(counts.sum(), 1) * 100.0


def transformed_statistics(
    group: str, source: str, transform_name: str, values: np.ndarray
) -> dict[str, float | int | str]:
    transformed = TRANSFORMS[transform_name](values)
    series = pd.Series(transformed)
    return {
        "group": group,
        "source": SOURCE_LABELS[source],
        "transform": transform_name,
        "n": int(len(transformed)),
        "mean": float(np.mean(transformed)),
        "std": float(np.std(transformed)),
        "skewness": float(series.skew()),
        "excess_kurtosis": float(series.kurt()),
    }


def standardized(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    sources: tuple[str, ...],
    bins: np.ndarray,
    title: str,
) -> float:
    panel_max = 0.0
    for source in sources:
        values = clean_values(frame[source])
        percentages = histogram_percent(values, bins)
        panel_max = max(panel_max, float(percentages.max()))
        ax.stairs(
            percentages,
            bins,
            label=SOURCE_LABELS[source],
            color=SOURCE_COLORS[source],
            linewidth=2.0,
        )
    ax.axvline(5.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.axvline(10.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, 25.0)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    return panel_max


def draw_transform_figure(
    frames: dict[str, pd.DataFrame], output: Path
) -> None:
    sources = ("scada_ws", "ldaps_ws50_max", "gfs_ws100")
    bins = np.arange(-4.0, 5.05, 0.1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    normal_density = np.exp(-(centers**2) / 2.0) / np.sqrt(2.0 * np.pi)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)

    for row, group in enumerate(GROUPS):
        for col, source in enumerate(sources):
            ax = axes[row, col]
            values = clean_values(frames[group][source])
            for transform_name, transform in TRANSFORMS.items():
                z_values = standardized(transform(values))
                density, _ = np.histogram(z_values, bins=bins, density=True)
                ax.stairs(
                    density,
                    bins,
                    color=TRANSFORM_COLORS[transform_name],
                    label=transform_name,
                    linewidth=1.8,
                )
            ax.plot(
                centers,
                normal_density,
                color="#111111",
                linestyle="--",
                linewidth=1.3,
                label="standard normal",
            )
            ax.set_xlim(-3.5, 4.5)
            ax.set_ylim(0.0, 0.72)
            ax.set_title(f"{group} | {SOURCE_LABELS[source]}", fontsize=11)
            ax.grid(axis="y", alpha=0.2)
            if col == 0:
                ax.set_ylabel("Density")
            if row == len(GROUPS) - 1:
                ax.set_xlabel("Standardized transformed wind speed")

    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Wind-speed power transforms compared with a normal distribution",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Each transformed distribution is standardized independently. Lower absolute skewness is more symmetric.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ldaps = pd.read_csv(args.ldaps, encoding="utf-8-sig")
    gfs = pd.read_csv(args.gfs, encoding="utf-8-sig")
    scada_vestas = pd.read_csv(args.scada_vestas, encoding="utf-8-sig")
    scada_unison = pd.read_csv(args.scada_unison, encoding="utf-8-sig")

    weather = build_weather(ldaps, gfs)
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    for group in GROUPS:
        scada_hourly = hourly_group_scada(scada_by_group[group], group)
        frame = weather.merge(scada_hourly, on="forecast_kst_dtm", how="inner")
        frames[group] = frame
        for source in SOURCE_LABELS:
            summary_rows.append(summarize(group, source, clean_values(frame[source])))

    bins = np.arange(0.0, 30.5, 0.5)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True)
    panel_max = 0.0
    for row, group in enumerate(GROUPS):
        panel_max = max(
            panel_max,
            draw_panel(
                axes[row, 0],
                frames[group],
                LOW_LEVEL_SOURCES,
                bins,
                f"{group}: low-level forecast wind",
            ),
            draw_panel(
                axes[row, 1],
                frames[group],
                HUB_LEVEL_SOURCES,
                bins,
                f"{group}: elevated forecast wind",
            ),
        )
        axes[row, 0].set_ylabel("Share per 0.5 m/s bin (%)")

    for ax in axes.flat:
        ax.set_ylim(0.0, panel_max * 1.12)
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[-1, 0].set_xlabel("Wind speed (m/s)")
    axes[-1, 1].set_xlabel("Wind speed (m/s)")
    fig.suptitle(
        "Hourly wind-speed distributions: SCADA vs LDAPS/GFS",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Dashed lines mark 5 and 10 m/s. LDAPS uses the 16-grid mean; GFS uses the nearest grid.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))

    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    transform_rows = []
    for group, frame in frames.items():
        for source in SOURCE_LABELS:
            values = clean_values(frame[source])
            for transform_name in TRANSFORMS:
                transform_rows.append(
                    transformed_statistics(group, source, transform_name, values)
                )
    transform_summary = pd.DataFrame(transform_rows)
    transform_summary["abs_skewness"] = transform_summary["skewness"].abs()
    transform_summary["best_abs_skew"] = transform_summary.groupby(
        ["group", "source"]
    )["abs_skewness"].transform("min").eq(transform_summary["abs_skewness"])
    transform_summary.to_csv(
        args.transform_summary_output, index=False, encoding="utf-8-sig"
    )
    draw_transform_figure(frames, args.transform_figure_output)

    gfs_grid_id = nearest_gfs_grid_id(gfs, FARM_CENTROID[0], FARM_CENTROID[1])
    print(f"nearest GFS grid: {gfs_grid_id}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved figure: {args.figure_output}")
    print(f"saved summary: {args.summary_output}")
    print("\n=== lowest absolute skewness ===")
    print(
        transform_summary.loc[transform_summary["best_abs_skew"]]
        .sort_values(["group", "source"])
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"saved transform figure: {args.transform_figure_output}")
    print(f"saved transform summary: {args.transform_summary_output}")


if __name__ == "__main__":
    main()
