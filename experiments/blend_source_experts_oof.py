from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # type: ignore[no-redef]  # noqa: F401

from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)


SOURCE_FILES = {
    "ldaps_core": "ldaps_core_oof_predictions.csv",
    "gfs_core": "gfs_core_oof_predictions.csv",
    "gefs_mean_core": "gefs_mean_core_oof_predictions.csv",
}
SOURCE_ORDER = tuple(SOURCE_FILES)
KEY_COLUMNS = ("forecast_kst_dtm", "pred_year", "lead", "group")
NESTED_VARIANT = "source_convex_nested"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-year-out convex blend of source expert OOF predictions."
    )
    parser.add_argument("--input-dir", default="results/source_experts_v1")
    parser.add_argument("--output-dir", default="results/source_experts_v1")
    parser.add_argument("--ldaps-file")
    parser.add_argument("--gfs-file")
    parser.add_argument("--gefs-file")
    parser.add_argument("--ldaps-variant", default="ldaps_core")
    parser.add_argument("--gfs-variant", default="gfs_core")
    parser.add_argument("--gefs-variant", default="gefs_mean_core")
    parser.add_argument("--nested-variant", default=NESTED_VARIANT)
    parser.add_argument("--output-prefix", default="source_expert_convex_nested")
    return parser.parse_args()


def git_head() -> str | None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_source_inputs(input_dir: Path) -> dict[str, tuple[Path, str]]:
    return {
        source: (input_dir / filename, source)
        for source, filename in SOURCE_FILES.items()
    }


def load_aligned_predictions(
    source_inputs: dict[str, tuple[Path, str]],
) -> pd.DataFrame:
    if tuple(source_inputs) != SOURCE_ORDER:
        raise ValueError(f"Source input order must be {SOURCE_ORDER}")
    aligned = None
    for source, (path, expected_variant) in source_inputs.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = [*KEY_COLUMNS, "variant", "official_target", "pred"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{source}: missing OOF columns {missing}")
        if set(frame["variant"].unique()) != {expected_variant}:
            raise ValueError(
                f"{source}: expected variant {expected_variant}, "
                f"got {sorted(frame['variant'].unique())}"
            )
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"{source}: duplicate OOF keys")
        selected = frame[[*KEY_COLUMNS, "official_target", "pred"]].copy()
        selected["forecast_kst_dtm"] = pd.to_datetime(selected["forecast_kst_dtm"])
        selected = selected.rename(
            columns={"official_target": f"actual_{source}", "pred": f"pred_{source}"}
        )
        if aligned is None:
            aligned = selected
        else:
            aligned = aligned.merge(
                selected,
                on=list(KEY_COLUMNS),
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            if not aligned["_merge"].eq("both").all():
                counts = aligned["_merge"].value_counts().to_dict()
                raise ValueError(f"{source}: OOF keys are not aligned: {counts}")
            aligned = aligned.drop(columns="_merge")
    if aligned is None:
        raise ValueError("No source predictions loaded")
    actual_columns = [f"actual_{source}" for source in SOURCE_ORDER]
    reference = aligned[actual_columns[0]].to_numpy(float)
    for column in actual_columns[1:]:
        if not np.allclose(reference, aligned[column].to_numpy(float), equal_nan=True):
            raise ValueError(f"Official targets differ across source OOF files: {column}")
    prediction_columns = [f"pred_{source}" for source in SOURCE_ORDER]
    if not np.isfinite(aligned[prediction_columns].to_numpy(float)).all():
        raise ValueError("Source OOF predictions contain non-finite values")
    aligned["official_target"] = reference
    return aligned.drop(columns=actual_columns).sort_values(list(KEY_COLUMNS)).reset_index(
        drop=True
    )


def score_prediction(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> tuple[float, float, float, list[dict[str, float | int | str]]]:
    if len(frame) != len(prediction):
        raise ValueError("Prediction length differs from score frame")
    group_rows = []
    for group in TARGET_COLS:
        mask = frame["group"].eq(group).to_numpy()
        if not mask.any():
            continue
        actual = frame.loc[mask, "official_target"].to_numpy(float)
        forecast = np.asarray(prediction, dtype=float)[mask]
        finite = np.isfinite(actual) & np.isfinite(forecast)
        scored = finite & (actual >= GROUP_CAPACITY_KWH[group] * 0.10)
        if not scored.any():
            continue
        nmae, ficr = group_nmae_ficr(
            actual[scored], forecast[scored], GROUP_CAPACITY_KWH[group]
        )
        group_rows.append(
            {
                "group": group,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": int(scored.sum()),
            }
        )
    if not group_rows:
        raise ValueError("No groups available for blend scoring")
    mean_nmae = float(np.mean([float(row["nmae"]) for row in group_rows]))
    mean_ficr = float(np.mean([float(row["ficr"]) for row in group_rows]))
    score = 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr
    return score, mean_nmae, mean_ficr, group_rows


def simplex_candidates(total_units: int) -> list[tuple[int, int, int]]:
    candidates = []
    for ldaps_units in range(total_units, -1, -1):
        for gfs_units in range(total_units - ldaps_units, -1, -1):
            gefs_units = total_units - ldaps_units - gfs_units
            candidates.append((ldaps_units, gfs_units, gefs_units))
    return candidates


def local_candidates(
    coarse_units: tuple[int, int, int],
    coarse_total: int = 40,
    fine_total: int = 200,
) -> list[tuple[int, int, int]]:
    if sum(coarse_units) != coarse_total or fine_total % coarse_total != 0:
        raise ValueError("Invalid coarse/fine simplex units")
    scale = fine_total // coarse_total
    center = tuple(value * scale for value in coarse_units)
    radius = scale
    return [
        candidate
        for candidate in simplex_candidates(fine_total)
        if all(
            abs(candidate[index] - center[index]) <= radius
            for index in range(len(center))
        )
    ]


def search_candidates(
    frame: pd.DataFrame,
    candidates: list[tuple[int, int, int]],
    total_units: int,
) -> tuple[np.ndarray, float, int]:
    matrix = frame[[f"pred_{source}" for source in SOURCE_ORDER]].to_numpy(float)
    best_weights = None
    best_score = -np.inf
    for candidate in candidates:
        weights = np.asarray(candidate, dtype=float) / float(total_units)
        score, _, _, _ = score_prediction(frame, matrix @ weights)
        if score > best_score + 1e-12:
            best_score = score
            best_weights = weights
    if best_weights is None:
        raise RuntimeError("Convex weight search selected no candidate")
    return best_weights, best_score, len(candidates)


def select_nested_weights(
    frame: pd.DataFrame,
    held_out_year: int,
) -> dict[str, object]:
    train_years = sorted(
        int(year) for year in frame["pred_year"].unique() if int(year) != held_out_year
    )
    train = frame.loc[frame["pred_year"].isin(train_years)].copy()
    validation = frame.loc[frame["pred_year"].eq(held_out_year)].copy()
    if train.empty or validation.empty or held_out_year in train_years:
        raise ValueError(f"Invalid meta fold for held-out year {held_out_year}")

    coarse, coarse_score, coarse_count = search_candidates(
        train, simplex_candidates(40), total_units=40
    )
    coarse_units = tuple(int(round(weight * 40)) for weight in coarse)
    fine_candidates = local_candidates(coarse_units)
    weights, train_score, fine_count = search_candidates(
        train, fine_candidates, total_units=200
    )
    matrix = validation[
        [f"pred_{source}" for source in SOURCE_ORDER]
    ].to_numpy(float)
    validation_prediction = matrix @ weights
    validation_score, validation_nmae, validation_ficr, group_rows = score_prediction(
        validation, validation_prediction
    )
    return {
        "held_out_year": held_out_year,
        "train_years": train_years,
        "weights": weights,
        "coarse_score": coarse_score,
        "train_score": train_score,
        "validation_score": validation_score,
        "validation_nmae": validation_nmae,
        "validation_ficr": validation_ficr,
        "coarse_candidates": coarse_count,
        "fine_candidates": fine_count,
        "validation_index": validation.index.to_numpy(),
        "validation_prediction": validation_prediction,
        "group_rows": group_rows,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_inputs = default_source_inputs(input_dir)
    source_inputs = {
        "ldaps_core": (
            Path(args.ldaps_file) if args.ldaps_file else default_inputs["ldaps_core"][0],
            args.ldaps_variant,
        ),
        "gfs_core": (
            Path(args.gfs_file) if args.gfs_file else default_inputs["gfs_core"][0],
            args.gfs_variant,
        ),
        "gefs_mean_core": (
            Path(args.gefs_file)
            if args.gefs_file
            else default_inputs["gefs_mean_core"][0],
            args.gefs_variant,
        ),
    }
    aligned = load_aligned_predictions(source_inputs)
    years = sorted(int(year) for year in aligned["pred_year"].unique())
    if years != [2022, 2023, 2024]:
        raise ValueError(f"Expected outer years 2022..2024, got {years}")

    nested_prediction = np.full(len(aligned), np.nan, dtype=float)
    weight_rows = []
    fold_rows = []
    for held_out_year in years:
        result = select_nested_weights(aligned, held_out_year)
        indices = result["validation_index"]
        nested_prediction[indices] = result["validation_prediction"]
        weights = result["weights"]
        weight_rows.append(
            {
                "held_out_year": held_out_year,
                "train_years": ",".join(map(str, result["train_years"])),
                "ldaps_weight": weights[0],
                "gfs_weight": weights[1],
                "gefs_weight": weights[2],
                "coarse_train_score": result["coarse_score"],
                "refined_train_score": result["train_score"],
                "held_out_score": result["validation_score"],
                "held_out_nmae": result["validation_nmae"],
                "held_out_ficr": result["validation_ficr"],
                "coarse_candidates": result["coarse_candidates"],
                "fine_candidates": result["fine_candidates"],
            }
        )
        for row in result["group_rows"]:
            fold_rows.append(
                {
                    "variant": args.nested_variant,
                    "pred_year": held_out_year,
                    **row,
                }
            )
        print(
            f"held_out={held_out_year} train={result['train_years']} "
            f"weights={weights.round(3).tolist()} "
            f"train_score={result['train_score']:.6f} "
            f"held_out_score={result['validation_score']:.6f}",
            flush=True,
        )
    if not np.isfinite(nested_prediction).all():
        raise RuntimeError("Nested blend did not predict every OOF row")

    nested = aligned[[*KEY_COLUMNS, "official_target"]].copy()
    nested["variant"] = args.nested_variant
    nested["pred"] = nested_prediction
    nested = nested[
        [
            "forecast_kst_dtm",
            "pred_year",
            "lead",
            "variant",
            "group",
            "official_target",
            "pred",
        ]
    ]
    nested_summary, nested_group_scores = pooled_oof_summary(nested)

    source_long = []
    for source in SOURCE_ORDER:
        part = aligned[[*KEY_COLUMNS, "official_target"]].copy()
        part["variant"] = source_inputs[source][1]
        part["pred"] = aligned[f"pred_{source}"]
        source_long.append(part)
    comparison, _ = pooled_oof_summary(
        pd.concat([*source_long, nested], ignore_index=True)
    )
    comparison = comparison.sort_values("mean_score", ascending=False).reset_index(
        drop=True
    )

    outputs = {
        f"{args.output_prefix}_predictions.csv": nested,
        f"{args.output_prefix}_fold_weights.csv": pd.DataFrame(weight_rows),
        f"{args.output_prefix}_fold_scores.csv": pd.DataFrame(fold_rows),
        f"{args.output_prefix}_summary.csv": nested_summary,
        f"{args.output_prefix}_group_scores.csv": nested_group_scores,
        f"{args.output_prefix}_comparison.csv": comparison,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    manifest = {
        "git_head": git_head(),
        "method": "leave-one-year-out convex source blend",
        "sources": {
            source: {
                "variant": expected_variant,
                "path": str(path),
                "sha256": sha256(path),
            }
            for source, (path, expected_variant) in source_inputs.items()
        },
        "constraints": {"nonnegative": True, "sum_to_one": True, "intercept": False},
        "weight_scope": "one common source weight vector per held-out year",
        "search": {
            "coarse_step": 0.025,
            "local_refinement_step": 0.005,
            "local_radius_per_weight": 0.025,
            "objective": "pooled official hard Score on the other two OOF years",
            "tie_break": "candidate order favors more LDAPS, then more GFS",
        },
        "folds": weight_rows,
        "test_prediction_created": False,
        "submission_created": False,
        "outputs": list(outputs),
    }
    with (output_dir / f"{args.output_prefix}_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("\n=== Meta-year held-out source blend ===", flush=True)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
