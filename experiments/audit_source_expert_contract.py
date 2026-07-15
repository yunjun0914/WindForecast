from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # type: ignore[no-redef]  # noqa: F401

from utils.source_expert_dataset import (
    GFS_CORE_SPEC,
    LDAPS_CORE_SPEC,
    GEFSIssueTensor,
    SourceIssueTensor,
    apply_gefs_publication_fallback,
    build_gefs_mean_core_tensor,
    build_grid_source_core_tensor,
    gefs_publication_audit,
    load_gefs_core_frames,
    select_gefs_issues,
    source_required_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit minimal raw-grid contracts for LDAPS/GFS/GEFS source experts."
    )
    parser.add_argument("--ldaps-train", default="data/train/ldaps_train.csv")
    parser.add_argument("--ldaps-test", default="data/test/ldaps_test.csv")
    parser.add_argument("--gfs-train", default="data/train/gfs_train.csv")
    parser.add_argument("--gfs-test", default="data/test/gfs_test.csv")
    parser.add_argument("--gefs-root", default="data/external/gefs")
    parser.add_argument(
        "--output-dir", default="windforecast_runs/source_experts_v1"
    )
    return parser.parse_args()


def read_source_csv(path: str, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", usecols=list(columns))


def contract_row(tensor: SourceIssueTensor, split: str) -> dict[str, object]:
    spatial_points = int(tensor.spatial_mask.sum())
    valid_cells = int(
        len(tensor.issue_times)
        * len(tensor.leads)
        * len(tensor.channel_names)
        * spatial_points
    )
    missing_cells = int(tensor.missing_mask[..., tensor.spatial_mask].sum())
    return {
        "source": tensor.source,
        "split": split,
        "issues": len(tensor.issue_times),
        "hours_per_issue": len(tensor.leads),
        "channels": len(tensor.channel_names),
        "height": tensor.values.shape[-2],
        "width": tensor.values.shape[-1],
        "spatial_points": spatial_points,
        "valid_cells": valid_cells,
        "original_missing_cells": missing_cells,
        "original_missing_fraction": missing_cells / max(valid_cells, 1),
        "fallback_issues": int(
            0 if tensor.fallback_flags is None else tensor.fallback_flags.sum()
        ),
        "first_issue": pd.Timestamp(tensor.issue_times.min()),
        "last_issue": pd.Timestamp(tensor.issue_times.max()),
        "first_forecast": pd.Timestamp(tensor.forecast_times.min()),
        "last_forecast": pd.Timestamp(tensor.forecast_times.max()),
        "finite_after_imputation": bool(np.isfinite(tensor.values).all()),
    }


def channel_rows(tensor: SourceIssueTensor, split: str) -> list[dict[str, object]]:
    spatial = tensor.spatial_mask.astype(bool)
    rows = []
    for channel_index, channel in enumerate(tensor.channel_names):
        values = tensor.values[:, :, channel_index][..., spatial].reshape(-1)
        missing = tensor.missing_mask[:, :, channel_index][..., spatial].reshape(-1)
        observed = values[~missing]
        rows.append(
            {
                "source": tensor.source,
                "split": split,
                "channel": channel,
                "cells": len(values),
                "original_missing_cells": int(missing.sum()),
                "original_missing_fraction": float(missing.mean()),
                "observed_min": float(observed.min()) if len(observed) else np.nan,
                "observed_max": float(observed.max()) if len(observed) else np.nan,
                "observed_mean": float(observed.mean()) if len(observed) else np.nan,
                "observed_std": float(observed.std()) if len(observed) else np.nan,
            }
        )
    return rows


def missing_rows(tensor: SourceIssueTensor, split: str) -> list[dict[str, object]]:
    spatial = tensor.spatial_mask.astype(bool)
    missing_by_issue = tensor.missing_mask[..., spatial].sum(axis=(1, 2, 3))
    rows = []
    for index in np.flatnonzero(
        (missing_by_issue > 0)
        | (
            np.zeros_like(missing_by_issue, dtype=bool)
            if tensor.fallback_flags is None
            else tensor.fallback_flags
        )
    ):
        rows.append(
            {
                "source": tensor.source,
                "split": split,
                "issue_time": pd.Timestamp(tensor.issue_times[index]),
                "original_missing_cells": int(missing_by_issue[index]),
                "publication_fallback": bool(
                    False
                    if tensor.fallback_flags is None
                    else tensor.fallback_flags[index]
                ),
            }
        )
    return rows


def issue_rows(tensor: SourceIssueTensor, split: str) -> list[dict[str, object]]:
    return [
        {
            "source": tensor.source,
            "split": split,
            "issue_time": pd.Timestamp(tensor.issue_times[index]),
            "first_forecast_year": int(tensor.years[index, 0]),
            "last_forecast_year": int(tensor.years[index, -1]),
            "crosses_forecast_year": bool(
                tensor.years[index, 0] != tensor.years[index, -1]
            ),
            "first_forecast": pd.Timestamp(tensor.forecast_times[index, 0]),
            "last_forecast": pd.Timestamp(tensor.forecast_times[index, -1]),
            "hours": len(tensor.leads),
            "publication_fallback": bool(
                False
                if tensor.fallback_flags is None
                else tensor.fallback_flags[index]
            ),
        }
        for index in range(len(tensor.issue_times))
    ]


def tensor_schema(tensor: SourceIssueTensor) -> dict[str, object]:
    return {
        "source": tensor.source,
        "shape": list(tensor.values.shape),
        "channels": list(tensor.channel_names),
        "spatial_mask": tensor.spatial_mask.astype(int).tolist(),
        "latitudes": (
            None if tensor.latitudes is None else tensor.latitudes.astype(float).tolist()
        ),
        "longitudes": (
            None if tensor.longitudes is None else tensor.longitudes.astype(float).tolist()
        ),
    }


def assert_same_schema(train: SourceIssueTensor, test: SourceIssueTensor) -> None:
    if train.channel_names != test.channel_names:
        raise ValueError(f"{train.source}: train/test channels differ")
    if not np.array_equal(train.spatial_mask, test.spatial_mask):
        raise ValueError(f"{train.source}: train/test spatial mask differs")
    if train.values.shape[2:] != test.values.shape[2:]:
        raise ValueError(f"{train.source}: train/test tensor schema differs")
    if train.latitudes is not None and not np.allclose(train.latitudes, test.latitudes):
        raise ValueError(f"{train.source}: train/test latitude crop differs")
    if train.longitudes is not None and not np.allclose(
        train.longitudes, test.longitudes
    ):
        raise ValueError(f"{train.source}: train/test longitude crop differs")


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading minimal LDAPS/GFS columns...", flush=True)
    ldaps_train = build_grid_source_core_tensor(
        read_source_csv(args.ldaps_train, source_required_columns(LDAPS_CORE_SPEC)),
        LDAPS_CORE_SPEC,
    )
    ldaps_test = build_grid_source_core_tensor(
        read_source_csv(args.ldaps_test, source_required_columns(LDAPS_CORE_SPEC)),
        LDAPS_CORE_SPEC,
    )
    gfs_train = build_grid_source_core_tensor(
        read_source_csv(args.gfs_train, source_required_columns(GFS_CORE_SPEC)),
        GFS_CORE_SPEC,
    )
    gfs_test = build_grid_source_core_tensor(
        read_source_csv(args.gfs_test, source_required_columns(GFS_CORE_SPEC)),
        GFS_CORE_SPEC,
    )
    assert_same_schema(ldaps_train, ldaps_test)
    assert_same_schema(gfs_train, gfs_test)
    if not np.array_equal(ldaps_train.issue_times, gfs_train.issue_times):
        raise ValueError("LDAPS/GFS train issue times differ")
    if not np.array_equal(ldaps_test.issue_times, gfs_test.issue_times):
        raise ValueError("LDAPS/GFS test issue times differ")

    print("Loading GEFS mean core parquet...", flush=True)
    pressure, gust = load_gefs_core_frames(args.gefs_root)
    gefs_all = build_gefs_mean_core_tensor(pressure, gust)
    publication_mean = gefs_publication_audit(args.gefs_root, kind="geavg")
    publication_spread = gefs_publication_audit(args.gefs_root, kind="gespr")
    gefs_all = apply_gefs_publication_fallback(gefs_all, publication_mean)
    gefs_train: GEFSIssueTensor = select_gefs_issues(
        gefs_all, ldaps_train.issue_times
    )
    gefs_test: GEFSIssueTensor = select_gefs_issues(gefs_all, ldaps_test.issue_times)
    assert_same_schema(gefs_train.pressure, gefs_test.pressure)
    assert_same_schema(gefs_train.gust, gefs_test.gust)

    tensors = [
        (ldaps_train, "train"),
        (ldaps_test, "test"),
        (gfs_train, "train"),
        (gfs_test, "test"),
        (gefs_train.pressure, "train"),
        (gefs_test.pressure, "test"),
        (gefs_train.gust, "train"),
        (gefs_test.gust, "test"),
    ]
    contracts = pd.DataFrame([contract_row(tensor, split) for tensor, split in tensors])
    channels = pd.DataFrame(
        [row for tensor, split in tensors for row in channel_rows(tensor, split)]
    )
    missing = pd.DataFrame(
        [row for tensor, split in tensors for row in missing_rows(tensor, split)]
    )
    issues = pd.DataFrame(
        [row for tensor, split in tensors for row in issue_rows(tensor, split)]
    )

    contracts.to_csv(
        output_dir / "source_contract_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    channels.to_csv(
        output_dir / "source_channel_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing.to_csv(
        output_dir / "source_missing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    issues.to_csv(
        output_dir / "source_issue_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    publication = pd.concat([publication_mean, publication_spread], ignore_index=True)
    publication.to_csv(
        output_dir / "gefs_publication_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "git_head": git_head(),
        "phase": "source_contract_audit_only",
        "training_run": False,
        "submission_created": False,
        "ldaps": tensor_schema(ldaps_train),
        "gfs": tensor_schema(gfs_train),
        "gefs_pressure": tensor_schema(gefs_train.pressure),
        "gefs_gust": tensor_schema(gefs_train.gust),
        "gefs_publication": {
            "mean_unsafe_issues": int((~publication_mean["safe"]).sum()),
            "spread_unsafe_issues": int((~publication_spread["safe"]).sum()),
            "minimum_mean_margin_hours": float(
                publication_mean.loc[publication_mean["safe"], "publication_margin_hours"].min()
            ),
        },
    }
    with (output_dir / "source_expert_contract_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("\n=== Source contract summary ===", flush=True)
    print(contracts.to_string(index=False), flush=True)
    print(
        f"\nGEFS unsafe issues: mean={manifest['gefs_publication']['mean_unsafe_issues']} "
        f"spread={manifest['gefs_publication']['spread_unsafe_issues']}",
        flush=True,
    )
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
