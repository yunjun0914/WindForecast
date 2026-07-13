from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TERRAIN_VARIANTS = ["none", "static4", "directional2"]
STATIC4_COLUMNS = [
    "terrain_dem_ground_m",
    "terrain_rotor_absolute_altitude_m",
    "terrain_slope_deg",
    "terrain_tpi_300m",
]
DIRECTIONAL2_COLUMNS = [
    "terrain_upwind_max_angle_deg",
    "terrain_upwind_mean_delta_m",
]


def terrain_feature_columns(variant: str) -> list[str]:
    if variant == "none":
        return []
    if variant == "static4":
        return STATIC4_COLUMNS.copy()
    if variant == "directional2":
        return DIRECTIONAL2_COLUMNS.copy()
    raise ValueError(f"Unknown terrain variant: {variant}")


def _load_static4(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Terrain turbine audit cache is missing: {path}")
    source = pd.read_csv(path, encoding="utf-8-sig")
    required = [
        "turbine_id",
        "dem_ground_m",
        "rotor_absolute_altitude_m",
        "slope_deg",
        "tpi_300m",
    ]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Static terrain cache missing columns: {missing}")
    out = source[required].drop_duplicates("turbine_id").rename(
        columns={
            "dem_ground_m": STATIC4_COLUMNS[0],
            "rotor_absolute_altitude_m": STATIC4_COLUMNS[1],
            "slope_deg": STATIC4_COLUMNS[2],
            "tpi_300m": STATIC4_COLUMNS[3],
        }
    )
    return out


def _load_directional2(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Terrain sector cache is missing: {path}")
    source = pd.read_csv(path, encoding="utf-8-sig")
    required = [
        "turbine_id",
        "sector",
        "upwind_max_terrain_angle_deg",
        "upwind_mean_ground_delta_m",
    ]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Directional terrain cache missing columns: {missing}")
    out = source[required].drop_duplicates(["turbine_id", "sector"]).rename(
        columns={
            "upwind_max_terrain_angle_deg": DIRECTIONAL2_COLUMNS[0],
            "upwind_mean_ground_delta_m": DIRECTIONAL2_COLUMNS[1],
        }
    )
    return out


def add_per_turbine_terrain_features(
    features: pd.DataFrame,
    variant: str,
    turbine_cache: Path = Path(
        "results/external_terrain_audit_v1_turbines.csv"
    ),
    sector_cache: Path = Path(
        "results/terrain_directional_residual_audit_v1_sector_features.csv"
    ),
) -> pd.DataFrame:
    if variant == "none":
        return features.copy()
    before = len(features)
    if variant == "static4":
        out = features.merge(
            _load_static4(turbine_cache),
            on="turbine_id",
            how="left",
            validate="many_to_one",
        )
    elif variant == "directional2":
        required = ["optgrid_wd_sin", "optgrid_wd_cos"]
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise ValueError(
                "Directional terrain requires optimal-grid u/v direction features: "
                f"{missing}"
            )
        out = features.copy()
        # Optimal-grid stores sin=v/speed and cos=u/speed. Meteorological upwind
        # bearing is atan2(-u, -v), clockwise from north.
        wind_from_deg = (
            np.degrees(
                np.arctan2(-out["optgrid_wd_cos"], -out["optgrid_wd_sin"])
            )
            + 360.0
        ) % 360.0
        out["_terrain_sector"] = np.floor(
            ((wind_from_deg + 15.0) % 360.0) / 30.0
        ).astype(int)
        sectors = _load_directional2(sector_cache).rename(
            columns={"sector": "_terrain_sector"}
        )
        out = out.merge(
            sectors,
            on=["turbine_id", "_terrain_sector"],
            how="left",
            validate="many_to_one",
        ).drop(columns="_terrain_sector")
    else:
        raise ValueError(f"Unknown terrain variant: {variant}")
    if len(out) != before:
        raise ValueError(f"Terrain merge changed row count: {before} -> {len(out)}")
    columns = terrain_feature_columns(variant)
    if out[columns].isna().any().any():
        missing_rows = int(out[columns].isna().any(axis=1).sum())
        raise ValueError(f"Terrain feature coverage incomplete: rows={missing_rows}")
    return out


def append_gnn_turbine_terrain_features(
    features: np.ndarray,
    variant: str,
    turbine_ids: list[str],
    n_weather: int,
    train_time_mask: np.ndarray,
    teacher_available: np.ndarray,
    teacher_wd_sin: np.ndarray,
    teacher_wd_cos: np.ndarray,
    turbine_cache: Path = Path(
        "results/external_terrain_audit_v1_turbines.csv"
    ),
    sector_cache: Path = Path(
        "results/terrain_directional_residual_audit_v1_sector_features.csv"
    ),
) -> np.ndarray:
    columns = terrain_feature_columns(variant)
    if not columns:
        return features
    block = np.zeros(
        (features.shape[0], features.shape[1], len(columns)), dtype=np.float32
    )
    if variant == "static4":
        static = _load_static4(turbine_cache).set_index("turbine_id").reindex(
            turbine_ids
        )
        if static[columns].isna().any().any():
            raise ValueError("Incomplete GNN static terrain coverage")
        values = static[columns].to_numpy(float)
        values = (values - values.mean(axis=0)) / np.maximum(
            values.std(axis=0), 1e-6
        )
        block[:, n_weather:, :] = values[None, :, :]
    elif variant == "directional2":
        sectors = _load_directional2(sector_cache)
        terrain_values = np.zeros(
            (features.shape[0], len(turbine_ids), len(columns)), dtype=float
        )
        wind_from_deg = (
            np.degrees(np.arctan2(teacher_wd_sin, teacher_wd_cos)) + 360.0
        ) % 360.0
        wind_from_deg = np.nan_to_num(wind_from_deg, nan=0.0)
        sector_index = np.floor(
            ((wind_from_deg + 15.0) % 360.0) / 30.0
        ).astype(int)
        for turbine_index, turbine_id in enumerate(turbine_ids):
            lookup = (
                sectors.loc[sectors["turbine_id"].eq(turbine_id)]
                .sort_values("sector")
                .set_index("sector")
                .reindex(range(12))
            )
            if lookup[columns].isna().any().any():
                raise ValueError(f"Incomplete GNN terrain sectors for {turbine_id}")
            terrain_values[:, turbine_index, :] = lookup[columns].to_numpy(float)[
                sector_index[:, turbine_index]
            ]
        fit_mask = train_time_mask[:, None] & teacher_available
        for column_index in range(len(columns)):
            fit_values = terrain_values[:, :, column_index][fit_mask]
            mean = float(np.mean(fit_values))
            std = max(float(np.std(fit_values)), 1e-6)
            terrain_values[:, :, column_index] = (
                terrain_values[:, :, column_index] - mean
            ) / std
        terrain_values[~teacher_available] = 0.0
        block[:, n_weather:, :] = terrain_values.astype(np.float32)
    else:
        raise ValueError(f"Unknown terrain variant: {variant}")
    return np.concatenate([features, block], axis=-1)
