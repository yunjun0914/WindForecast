from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_KEY_COLS = ("forecast_kst_dtm", "data_available_kst_dtm")
TARGET_COLS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
EXPECTED_LEADS = np.arange(12, 36, dtype=np.int16)
GEFS_FHOURS = np.arange(21, 49, 3, dtype=np.int16)
FARM_LATITUDE = 37.2819
FARM_LONGITUDE = 128.96237
TURBINE_HUB_HEIGHT_M = 117.0
LDAPS_BLH_FLOOR_M = 20.0
LDAPS_BLH_COLUMN = "etc_0_blh"
LDAPS_HUB_OVER_BLH_CHANNEL = "ldaps_hub_over_blh"
LDAPS_SURFACE_PRESSURE_COLUMN = "surface_0_sp"
LDAPS_PRESSURE_TENDENCY_CHANNEL = "ldaps_pressure_tendency_3h"
LDAPS_PRESSURE_TENDENCY_HOURS = 3
LDAPS_MSLP_COLUMN = "meanSea_0_prmsl"


@dataclass(frozen=True)
class VectorChannels:
    name: str
    u: str
    v: str


@dataclass(frozen=True)
class GridSourceSpec:
    name: str
    layout: dict[int, tuple[int, int]]
    vectors: tuple[VectorChannels, ...]
    scalar_channels: tuple[str, ...] = ()

    @property
    def raw_channels(self) -> tuple[str, ...]:
        columns: list[str] = []
        for vector in self.vectors:
            columns.extend([vector.u, vector.v])
        columns.extend(self.scalar_channels)
        return tuple(columns)

    @property
    def output_channels(self) -> tuple[str, ...]:
        return (*self.raw_channels, *(f"{vector.name}_speed" for vector in self.vectors))


LDAPS_CORE_SPEC = GridSourceSpec(
    name="ldaps_core",
    layout={
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        4: (1, 0),
        5: (1, 1),
        6: (1, 2),
        7: (1, 3),
        8: (1, 4),
        9: (2, 0),
        10: (2, 1),
        11: (2, 2),
        12: (2, 3),
        13: (2, 4),
        14: (3, 1),
        15: (3, 2),
        16: (3, 3),
    },
    vectors=(
        VectorChannels(
            "wind_50m_max",
            "heightAboveGround_50_50MUmax",
            "heightAboveGround_50_50MVmax",
        ),
        VectorChannels(
            "wind_50m_min",
            "heightAboveGround_50_50MUmin",
            "heightAboveGround_50_50MVmin",
        ),
        VectorChannels(
            "wind_10m",
            "heightAboveGround_10_10u",
            "heightAboveGround_10_10v",
        ),
    ),
)


LDAPS_5M_CORE_SPEC = GridSourceSpec(
    name="ldaps_5m_core",
    layout=LDAPS_CORE_SPEC.layout,
    vectors=(
        *LDAPS_CORE_SPEC.vectors,
        VectorChannels(
            "wind_5m",
            "heightAboveGround_5_XBLWS",
            "heightAboveGround_5_YBLWS",
        ),
    ),
)


LDAPS_BLH_RATIO_SPEC = GridSourceSpec(
    name="ldaps_blh_ratio_core",
    layout=LDAPS_CORE_SPEC.layout,
    vectors=LDAPS_CORE_SPEC.vectors,
    scalar_channels=(LDAPS_HUB_OVER_BLH_CHANNEL,),
)


LDAPS_SURFACE_PRESSURE_SPEC = GridSourceSpec(
    name="ldaps_core_sp",
    layout=LDAPS_CORE_SPEC.layout,
    vectors=LDAPS_CORE_SPEC.vectors,
    scalar_channels=(LDAPS_SURFACE_PRESSURE_COLUMN,),
)


LDAPS_PRESSURE_TENDENCY_SPEC = GridSourceSpec(
    name="ldaps_pressure_tendency_core",
    layout=LDAPS_CORE_SPEC.layout,
    vectors=LDAPS_CORE_SPEC.vectors,
    scalar_channels=(LDAPS_PRESSURE_TENDENCY_CHANNEL,),
)


LDAPS_MSLP_SPEC = GridSourceSpec(
    name="ldaps_mslp_core",
    layout=LDAPS_CORE_SPEC.layout,
    vectors=LDAPS_CORE_SPEC.vectors,
    scalar_channels=(LDAPS_MSLP_COLUMN,),
)


LDAPS_THERMO_PBL_CHANNELS = (
    "heightAboveGround_2_t",
    "heightAboveGround_2_dpt",
    "heightAboveGround_2_r",
    "heightAboveGround_2_q",
    "etc_0_blh",
)
LDAPS_THERMO_PBL_SPEC = GridSourceSpec(
    name="ldaps_thermo_pbl_core",
    layout=LDAPS_CORE_SPEC.layout,
    vectors=LDAPS_CORE_SPEC.vectors,
    scalar_channels=LDAPS_THERMO_PBL_CHANNELS,
)


LDAPS_SURFACE_REGIME_CHANNELS = (
    "surface_0_NDNSW",
    "surface_0_NDNLW",
    "heightAboveGround_2_SWDIR",
    "heightAboveGround_2_SWDIF",
    "etc_0_hcc",
    "etc_0_mcc",
    "etc_0_lcc",
    "etc_0_VLCDC",
    "surface_0_avg_lsprate",
    "surface_0_lssrate",
    "surface_0_ncpcp",
    "surface_0_snol",
    "surface_0_SNOM",
    "surface_0_h",
)
LDAPS_SURFACE_REGIME_SPEC = GridSourceSpec(
    name="ldaps_surface_regime_core",
    layout=LDAPS_CORE_SPEC.layout,
    vectors=LDAPS_CORE_SPEC.vectors,
    scalar_channels=LDAPS_SURFACE_REGIME_CHANNELS,
)


LDAPS_DERIVED_FAMILY_CHANNELS = {
    "envelope": (
        "ldaps_50m_mid_u",
        "ldaps_50m_mid_v",
        "ldaps_50m_mid_speed",
        "ldaps_50m_envelope_u",
        "ldaps_50m_envelope_v",
        "ldaps_50m_envelope_norm",
        "ldaps_50m_relative_envelope",
    ),
    "vertical_profile": (
        "ldaps_50m_mid_minus_10m_speed",
        "ldaps_50m_mid_10m_log_shear",
        "ldaps_50m_mid_10m_alignment",
        "ldaps_10m_minus_5m_speed",
        "ldaps_10m_5m_log_shear",
        "ldaps_10m_5m_alignment",
    ),
    "density_power": (
        "ldaps_virtual_temperature",
        "ldaps_air_density",
        "ldaps_density_equivalent_ws10",
        "ldaps_density_equivalent_ws50mid",
        "ldaps_power_density_ws10",
        "ldaps_power_density_ws50mid",
    ),
    "pbl_stability": (
        "ldaps_hub_over_blh_family",
        "ldaps_blh_tendency_3h_family",
        "ldaps_envelope_x_hub_over_blh",
        "ldaps_shear_x_hub_over_blh",
        "ldaps_ventilation_proxy",
    ),
    "spatial_flow": (
        "ldaps_spatial_ws10_mean",
        "ldaps_spatial_ws10_std",
        "ldaps_spatial_ws10_p90_p10",
        "ldaps_spatial_ws10_coherence",
        "ldaps_spatial_ws10_gradient_x",
        "ldaps_spatial_ws10_gradient_y",
        "ldaps_spatial_ws50mid_mean",
        "ldaps_spatial_ws50mid_std",
        "ldaps_spatial_ws50mid_p90_p10",
    ),
    "terrain_interaction": (
        "ldaps_terrain_relative_height",
        "ldaps_terrain_slope_x",
        "ldaps_terrain_slope_y",
        "ldaps_terrain_slope_magnitude",
        "ldaps_terrain_wind_alignment",
        "ldaps_terrain_wind_exposure",
    ),
    "temporal_trajectory": (
        "ldaps_ws10_ramp_1h",
        "ldaps_ws10_vector_ramp_1h",
        "ldaps_ws10_direction_turn_1h",
        "ldaps_ws50mid_ramp_1h",
        "ldaps_envelope_ramp_1h",
        "ldaps_ws10_roll3_mean",
        "ldaps_ws10_roll3_std",
        "ldaps_ws50mid_roll3_mean",
        "ldaps_ws10_issue_range",
        "ldaps_ws10_hours_to_peak",
    ),
    "thermodynamic_regime": (
        "ldaps_dewpoint_depression",
        "ldaps_relative_humidity_deficit",
        "ldaps_mslp_surface_gap",
        "ldaps_surface_pressure_anomaly",
        "ldaps_surface_pressure_gradient_x",
        "ldaps_surface_pressure_gradient_y",
        "ldaps_temperature_gradient_magnitude",
        "ldaps_specific_humidity_anomaly",
    ),
    "weather_regime": (
        "ldaps_shortwave_total",
        "ldaps_shortwave_direct_fraction",
        "ldaps_net_radiation_proxy",
        "ldaps_cloud_layer_mean",
        "ldaps_low_cloud_dominance",
        "ldaps_precip_log_activity",
        "ldaps_precip_occurrence",
        "ldaps_snow_log_activity",
        "ldaps_snow_occurrence",
    ),
}
LDAPS_DERIVED_FAMILY_CHANNELS["density_spatial"] = (
    *LDAPS_DERIVED_FAMILY_CHANNELS["density_power"],
    *LDAPS_DERIVED_FAMILY_CHANNELS["spatial_flow"],
)
LDAPS_DERIVED_SOURCE_FAMILIES = {
    f"ldaps_{family}_derived_core": family
    for family in LDAPS_DERIVED_FAMILY_CHANNELS
}
LDAPS_DERIVED_RAW_CHANNELS = {
    "envelope": (),
    "vertical_profile": (
        "heightAboveGround_5_XBLWS",
        "heightAboveGround_5_YBLWS",
    ),
    "density_power": (
        "heightAboveGround_2_t",
        "heightAboveGround_2_q",
        LDAPS_SURFACE_PRESSURE_COLUMN,
    ),
    "pbl_stability": (LDAPS_BLH_COLUMN,),
    "spatial_flow": (),
    "terrain_interaction": ("surface_0_h",),
    "temporal_trajectory": (),
    "thermodynamic_regime": (
        "heightAboveGround_2_t",
        "heightAboveGround_2_dpt",
        "heightAboveGround_2_r",
        "heightAboveGround_2_q",
        LDAPS_SURFACE_PRESSURE_COLUMN,
        LDAPS_MSLP_COLUMN,
    ),
    "weather_regime": tuple(
        channel for channel in LDAPS_SURFACE_REGIME_CHANNELS if channel != "surface_0_h"
    ),
    "density_spatial": (
        "heightAboveGround_2_t",
        "heightAboveGround_2_q",
        LDAPS_SURFACE_PRESSURE_COLUMN,
    ),
}


GFS_CORE_SPEC = GridSourceSpec(
    name="gfs_core",
    layout={grid_id: ((grid_id - 1) // 3, (grid_id - 1) % 3) for grid_id in range(1, 10)},
    vectors=(
        VectorChannels("wind_80m", "heightAboveGround_80_u", "heightAboveGround_80_v"),
        VectorChannels(
            "wind_100m",
            "heightAboveGround_100_100u",
            "heightAboveGround_100_100v",
        ),
    ),
    scalar_channels=("surface_0_gust",),
)


GFS_10M_CORE_SPEC = GridSourceSpec(
    name="gfs_10m_core",
    layout=GFS_CORE_SPEC.layout,
    vectors=(
        *GFS_CORE_SPEC.vectors,
        VectorChannels(
            "wind_10m",
            "heightAboveGround_10_10u",
            "heightAboveGround_10_10v",
        ),
    ),
    scalar_channels=GFS_CORE_SPEC.scalar_channels,
)


GFS_VERTICAL_WIND_EXTRA_VECTORS = (
    VectorChannels(
        "wind_pbl",
        "planetaryBoundaryLayer_0_u",
        "planetaryBoundaryLayer_0_v",
    ),
    VectorChannels(
        "wind_850",
        "isobaricInhPa_850_u",
        "isobaricInhPa_850_v",
    ),
    VectorChannels(
        "wind_700",
        "isobaricInhPa_700_u",
        "isobaricInhPa_700_v",
    ),
    VectorChannels(
        "wind_500",
        "isobaricInhPa_500_u",
        "isobaricInhPa_500_v",
    ),
)
GFS_VERTICAL_WIND_EXTRA_SCALARS = ("planetaryBoundaryLayer_0_VRATE",)
GFS_VERTICAL_WIND_SPEC = GridSourceSpec(
    name="gfs_vertical_wind_core",
    layout=GFS_CORE_SPEC.layout,
    vectors=(*GFS_10M_CORE_SPEC.vectors, *GFS_VERTICAL_WIND_EXTRA_VECTORS),
    scalar_channels=(
        *GFS_10M_CORE_SPEC.scalar_channels,
        *GFS_VERTICAL_WIND_EXTRA_SCALARS,
    ),
)


GFS_THERMO_SYNOPTIC_CHANNELS = (
    "heightAboveGround_2_2t",
    "heightAboveGround_2_2d",
    "heightAboveGround_2_2r",
    "heightAboveGround_2_2sh",
    "surface_0_sp",
    "meanSea_0_prmsl",
    "isobaricInhPa_850_t",
    "isobaricInhPa_850_r",
    "isobaricInhPa_700_t",
    "isobaricInhPa_500_t",
    "isobaricInhPa_500_gh",
)
GFS_THERMO_SYNOPTIC_SPEC = GridSourceSpec(
    name="gfs_thermo_synoptic_core",
    layout=GFS_CORE_SPEC.layout,
    vectors=GFS_10M_CORE_SPEC.vectors,
    scalar_channels=(
        *GFS_10M_CORE_SPEC.scalar_channels,
        *GFS_THERMO_SYNOPTIC_CHANNELS,
    ),
)


GFS_VERTICAL_THERMO_SPEC = GridSourceSpec(
    name="gfs_vertical_thermo_core",
    layout=GFS_CORE_SPEC.layout,
    vectors=GFS_VERTICAL_WIND_SPEC.vectors,
    scalar_channels=(
        *GFS_10M_CORE_SPEC.scalar_channels,
        *GFS_VERTICAL_WIND_EXTRA_SCALARS,
        *GFS_THERMO_SYNOPTIC_CHANNELS,
    ),
)


GFS_SURFACE_REGIME_CHANNELS = (
    "surface_0_dswrf",
    "surface_0_dlwrf",
    "surface_0_prate",
    "surface_0_tp",
    "lowCloudLayer_0_lcc",
    "middleCloudLayer_0_mcc",
    "highCloudLayer_0_hcc",
    "atmosphere_0_tcc",
)
GFS_SURFACE_REGIME_SPEC = GridSourceSpec(
    name="gfs_surface_regime_core",
    layout=GFS_CORE_SPEC.layout,
    vectors=GFS_10M_CORE_SPEC.vectors,
    scalar_channels=(
        *GFS_10M_CORE_SPEC.scalar_channels,
        *GFS_SURFACE_REGIME_CHANNELS,
    ),
)


GFS_SURFACE_PRESSURE_SPEC = GridSourceSpec(
    name="gfs_core_sp",
    layout=GFS_CORE_SPEC.layout,
    vectors=GFS_CORE_SPEC.vectors,
    scalar_channels=(*GFS_CORE_SPEC.scalar_channels, "surface_0_sp"),
)


GEFS_PRESSURE_MEAN_VECTORS = (
    VectorChannels("wind_10m", "u10m_mean", "v10m_mean"),
    VectorChannels("wind_925", "u925_mean", "v925_mean"),
    VectorChannels("wind_850", "u850_mean", "v850_mean"),
)
GEFS_PRESSURE_700_VECTOR = VectorChannels("wind_700", "u700_mean", "v700_mean")
GEFS_PRESSURE_SPREAD_CHANNELS = (
    "u10m_sprd",
    "v10m_sprd",
    "u925_sprd",
    "v925_sprd",
    "u850_sprd",
    "v850_sprd",
)
GEFS_NEAR_SPREAD_VECTORS = (
    VectorChannels("ensemble_spread_10m", "u10m_sprd", "v10m_sprd"),
    VectorChannels("ensemble_spread_925", "u925_sprd", "v925_sprd"),
)
GEFS_UPPER_SPREAD_VECTORS = (
    VectorChannels("ensemble_spread_850", "u850_sprd", "v850_sprd"),
    VectorChannels("ensemble_spread_700", "u700_sprd", "v700_sprd"),
)
GEFS_GUST_MEAN_CHANNELS = ("gust_mean",)
GEFS_GUST_SPREAD_CHANNELS = ("gust_sprd",)


@dataclass(frozen=True)
class SourceIssueTensor:
    source: str
    values: np.ndarray
    missing_mask: np.ndarray
    spatial_mask: np.ndarray
    channel_names: tuple[str, ...]
    forecast_times: np.ndarray
    issue_times: np.ndarray
    years: np.ndarray
    leads: np.ndarray
    time_features: np.ndarray
    targets: np.ndarray | None = None
    fallback_flags: np.ndarray | None = None
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None

    def validate(self) -> None:
        if self.values.ndim != 5:
            raise ValueError(f"{self.source}: values must be [issue,time,channel,height,width]")
        if self.missing_mask.shape != self.values.shape:
            raise ValueError(f"{self.source}: missing mask shape differs from values")
        issues, steps, channels, height, width = self.values.shape
        if steps != len(self.leads):
            raise ValueError(f"{self.source}: lead count differs from tensor time axis")
        if channels != len(self.channel_names):
            raise ValueError(f"{self.source}: channel names differ from tensor channel axis")
        if self.spatial_mask.shape != (height, width):
            raise ValueError(f"{self.source}: spatial mask shape differs from tensor grid")
        if self.forecast_times.shape != (issues, steps):
            raise ValueError(f"{self.source}: forecast time shape differs from tensor")
        if self.issue_times.shape != (issues,):
            raise ValueError(f"{self.source}: issue time shape differs from tensor")
        if self.years.shape != (issues, steps):
            raise ValueError(f"{self.source}: year shape differs from tensor")
        if self.time_features.shape[:2] != (issues, steps):
            raise ValueError(f"{self.source}: time features differ from tensor")
        if self.targets is not None and self.targets.shape != (issues, steps, len(TARGET_COLS)):
            raise ValueError(f"{self.source}: target shape differs from tensor")
        if self.fallback_flags is not None and self.fallback_flags.shape != (issues,):
            raise ValueError(f"{self.source}: fallback flags differ from tensor")
        if not np.isfinite(self.values).all():
            raise ValueError(f"{self.source}: tensor contains non-finite values after imputation")
        if not np.array_equal(self.leads.astype(np.int16), EXPECTED_LEADS):
            raise ValueError(f"{self.source}: expected leads 12..35")


@dataclass(frozen=True)
class GEFSIssueTensor:
    pressure: SourceIssueTensor
    gust: SourceIssueTensor

    def validate(self) -> None:
        self.pressure.validate()
        self.gust.validate()
        if not np.array_equal(self.pressure.issue_times, self.gust.issue_times):
            raise ValueError("GEFS pressure and gust issue times differ")
        if not np.array_equal(self.pressure.forecast_times, self.gust.forecast_times):
            raise ValueError("GEFS pressure and gust forecast times differ")
        if not np.array_equal(self.pressure.fallback_flags, self.gust.fallback_flags):
            raise ValueError("GEFS pressure and gust fallback flags differ")


@dataclass(frozen=True)
class SourceChannelScaler:
    channel_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray


def source_required_columns(spec: GridSourceSpec) -> tuple[str, ...]:
    return (*TIME_KEY_COLS, "grid_id", *spec.raw_channels)


def ldaps_blh_ratio_required_columns() -> tuple[str, ...]:
    return (*source_required_columns(LDAPS_CORE_SPEC), LDAPS_BLH_COLUMN)


def ldaps_pressure_tendency_required_columns() -> tuple[str, ...]:
    return source_required_columns(LDAPS_SURFACE_PRESSURE_SPEC)


def ldaps_derived_family_required_columns(family: str) -> tuple[str, ...]:
    if family not in LDAPS_DERIVED_RAW_CHANNELS:
        raise ValueError(f"Unknown LDAPS derived family: {family}")
    raw_channels = tuple(
        dict.fromkeys(
            (*LDAPS_CORE_SPEC.raw_channels, *LDAPS_DERIVED_RAW_CHANNELS[family])
        )
    )
    return (*TIME_KEY_COLS, "grid_id", *raw_channels)


def _time_features(forecast_times: np.ndarray, issue_times: np.ndarray) -> np.ndarray:
    flat_forecast = pd.to_datetime(forecast_times.reshape(-1))
    repeated_issue = np.repeat(pd.to_datetime(issue_times), forecast_times.shape[1])
    lead = (flat_forecast - repeated_issue).total_seconds().to_numpy(float) / 3600.0
    day = flat_forecast.dayofyear.to_numpy(float)
    hour = flat_forecast.hour.to_numpy(float)
    features = np.column_stack(
        [
            (lead - EXPECTED_LEADS.mean()) / EXPECTED_LEADS.std(),
            np.sin(2.0 * np.pi * day / 365.0),
            np.cos(2.0 * np.pi * day / 365.0),
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
        ]
    ).astype(np.float32)
    return features.reshape(*forecast_times.shape, features.shape[-1])


def _target_array(
    labels: pd.DataFrame | None,
    forecast_times: np.ndarray,
) -> np.ndarray | None:
    if labels is None:
        return None
    missing = [column for column in ("kst_dtm", *TARGET_COLS) if column not in labels.columns]
    if missing:
        raise ValueError(f"labels missing columns: {missing}")
    table = labels[["kst_dtm", *TARGET_COLS]].copy()
    table["kst_dtm"] = pd.to_datetime(table["kst_dtm"])
    if table["kst_dtm"].duplicated().any():
        raise ValueError("labels contain duplicate timestamps")
    indexed = table.set_index("kst_dtm")
    aligned = indexed.reindex(pd.to_datetime(forecast_times.reshape(-1)))
    return aligned[list(TARGET_COLS)].to_numpy(np.float32).reshape(
        *forecast_times.shape, len(TARGET_COLS)
    )


def _interpolate_issue_time(values: np.ndarray) -> np.ndarray:
    if values.ndim < 2:
        raise ValueError("issue interpolation expects issue and time axes")
    issues, steps = values.shape[:2]
    flattened = values.transpose(0, *range(2, values.ndim), 1).reshape(-1, steps)
    filled = (
        pd.DataFrame(flattened)
        .interpolate(axis=1, method="linear", limit_direction="both")
        .to_numpy(dtype=np.float32)
    )
    if np.isnan(filled).any():
        missing_vectors = int(np.isnan(filled).any(axis=1).sum())
        raise ValueError(f"{missing_vectors} issue/grid/channel vectors are entirely missing")
    restored = filled.reshape(issues, *values.shape[2:], steps).transpose(
        0, values.ndim - 1, *range(1, values.ndim - 1)
    )
    return restored.astype(np.float32)


def _build_spatial_mask(layout: dict[int, tuple[int, int]]) -> np.ndarray:
    height = max(row for row, _ in layout.values()) + 1
    width = max(column for _, column in layout.values()) + 1
    mask = np.zeros((height, width), dtype=bool)
    for row, column in layout.values():
        mask[row, column] = True
    return mask


def build_grid_source_core_tensor(
    frame: pd.DataFrame,
    spec: GridSourceSpec,
    labels: pd.DataFrame | None = None,
) -> SourceIssueTensor:
    required = source_required_columns(spec)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{spec.name} missing columns: {missing_columns}")

    work = frame[list(required)].copy()
    for column in TIME_KEY_COLS:
        work[column] = pd.to_datetime(work[column])
    work["grid_id"] = pd.to_numeric(work["grid_id"], errors="raise").astype(int)
    observed_grids = set(work["grid_id"].unique())
    expected_grids = set(spec.layout)
    if observed_grids != expected_grids:
        raise ValueError(
            f"{spec.name} grid ids differ: missing={sorted(expected_grids - observed_grids)} "
            f"extra={sorted(observed_grids - expected_grids)}"
        )
    if work.duplicated([*TIME_KEY_COLS, "grid_id"]).any():
        raise ValueError(f"{spec.name} contains duplicate time/grid rows")

    lead = (
        (work["forecast_kst_dtm"] - work["data_available_kst_dtm"])
        .dt.total_seconds()
        .div(3600.0)
    )
    if not np.allclose(lead, np.round(lead)):
        raise ValueError(f"{spec.name} contains non-integer lead hours")
    work["_lead"] = np.round(lead).astype(np.int16)
    unexpected_leads = sorted(set(work["_lead"].unique()) - set(EXPECTED_LEADS.tolist()))
    if unexpected_leads:
        raise ValueError(f"{spec.name} contains unexpected leads: {unexpected_leads}")

    issue_times = np.sort(work["data_available_kst_dtm"].unique()).astype("datetime64[ns]")
    expected_index = pd.MultiIndex.from_tuples(
        [
            (
                pd.Timestamp(issue_time) + pd.Timedelta(hours=int(step)),
                pd.Timestamp(issue_time),
            )
            for issue_time in issue_times
            for step in EXPECTED_LEADS
        ],
        names=list(TIME_KEY_COLS),
    )
    grid_ids = sorted(spec.layout)
    pivot = work.pivot(
        index=list(TIME_KEY_COLS),
        columns="grid_id",
        values=list(spec.raw_channels),
    ).reindex(index=expected_index)
    expected_columns = pd.MultiIndex.from_product([spec.raw_channels, grid_ids])
    pivot = pivot.reindex(columns=expected_columns)
    if len(pivot) != len(issue_times) * len(EXPECTED_LEADS):
        raise ValueError(f"{spec.name} issue row coverage differs from 24-hour contract")

    raw = pivot.to_numpy(np.float32).reshape(
        len(issue_times), len(EXPECTED_LEADS), len(spec.raw_channels), len(grid_ids)
    )
    raw_missing = np.isnan(raw)
    raw_filled = _interpolate_issue_time(raw)

    raw_index = {channel: index for index, channel in enumerate(spec.raw_channels)}
    channel_arrays = [raw_filled]
    missing_arrays = [raw_missing]
    for vector in spec.vectors:
        u_index = raw_index[vector.u]
        v_index = raw_index[vector.v]
        speed = np.hypot(raw_filled[:, :, u_index], raw_filled[:, :, v_index])
        speed_missing = raw_missing[:, :, u_index] | raw_missing[:, :, v_index]
        channel_arrays.append(speed[:, :, None, :].astype(np.float32))
        missing_arrays.append(speed_missing[:, :, None, :])
    flat_values = np.concatenate(channel_arrays, axis=2)
    flat_missing = np.concatenate(missing_arrays, axis=2)

    spatial_mask = _build_spatial_mask(spec.layout)
    values = np.zeros(
        (*flat_values.shape[:3], *spatial_mask.shape), dtype=np.float32
    )
    missing_mask = np.zeros_like(values, dtype=bool)
    for grid_index, grid_id in enumerate(grid_ids):
        row, column = spec.layout[grid_id]
        values[:, :, :, row, column] = flat_values[:, :, :, grid_index]
        missing_mask[:, :, :, row, column] = flat_missing[:, :, :, grid_index]

    forecast_times = np.asarray(
        [
            [pd.Timestamp(issue) + pd.Timedelta(hours=int(step)) for step in EXPECTED_LEADS]
            for issue in issue_times
        ],
        dtype="datetime64[ns]",
    )
    year_matrix = pd.to_datetime(forecast_times.reshape(-1)).year.to_numpy().reshape(
        forecast_times.shape
    )
    tensor = SourceIssueTensor(
        source=spec.name,
        values=values,
        missing_mask=missing_mask,
        spatial_mask=spatial_mask,
        channel_names=spec.output_channels,
        forecast_times=forecast_times,
        issue_times=issue_times,
        years=year_matrix.astype(np.int16),
        leads=EXPECTED_LEADS.copy(),
        time_features=_time_features(forecast_times, issue_times),
        targets=_target_array(labels, forecast_times),
        fallback_flags=np.zeros(len(issue_times), dtype=bool),
    )
    tensor.validate()
    return tensor


def build_ldaps_blh_ratio_tensor(
    frame: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> SourceIssueTensor:
    required = ldaps_blh_ratio_required_columns()
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{LDAPS_BLH_RATIO_SPEC.name} missing columns: {missing_columns}")

    work = frame[list(required)].copy()
    blh = pd.to_numeric(work.pop(LDAPS_BLH_COLUMN), errors="coerce")
    work[LDAPS_HUB_OVER_BLH_CHANNEL] = TURBINE_HUB_HEIGHT_M / np.maximum(
        blh,
        LDAPS_BLH_FLOOR_M,
    )
    return build_grid_source_core_tensor(work, LDAPS_BLH_RATIO_SPEC, labels=labels)


def build_ldaps_pressure_tendency_tensor(
    frame: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> SourceIssueTensor:
    pressure_tensor = build_grid_source_core_tensor(
        frame,
        LDAPS_SURFACE_PRESSURE_SPEC,
        labels=labels,
    )
    pressure_index = pressure_tensor.channel_names.index(LDAPS_SURFACE_PRESSURE_COLUMN)
    step_indices = np.arange(len(EXPECTED_LEADS))
    left_indices = np.maximum(step_indices - LDAPS_PRESSURE_TENDENCY_HOURS, 0)
    right_indices = np.minimum(
        step_indices + LDAPS_PRESSURE_TENDENCY_HOURS,
        len(EXPECTED_LEADS) - 1,
    )
    elapsed_hours = (
        EXPECTED_LEADS[right_indices] - EXPECTED_LEADS[left_indices]
    ).astype(np.float32)

    pressure = pressure_tensor.values[:, :, pressure_index]
    pressure_missing = pressure_tensor.missing_mask[:, :, pressure_index]
    tendency = (
        pressure[:, right_indices] - pressure[:, left_indices]
    ) / elapsed_hours[None, :, None, None]
    tendency_missing = (
        pressure_missing[:, right_indices] | pressure_missing[:, left_indices]
    )

    value_channels: list[np.ndarray] = []
    missing_channels: list[np.ndarray] = []
    for channel in LDAPS_PRESSURE_TENDENCY_SPEC.output_channels:
        if channel == LDAPS_PRESSURE_TENDENCY_CHANNEL:
            value_channels.append(tendency)
            missing_channels.append(tendency_missing)
            continue
        source_index = pressure_tensor.channel_names.index(channel)
        value_channels.append(pressure_tensor.values[:, :, source_index])
        missing_channels.append(pressure_tensor.missing_mask[:, :, source_index])

    tensor = replace(
        pressure_tensor,
        source=LDAPS_PRESSURE_TENDENCY_SPEC.name,
        values=np.stack(value_channels, axis=2).astype(np.float32),
        missing_mask=np.stack(missing_channels, axis=2),
        channel_names=LDAPS_PRESSURE_TENDENCY_SPEC.output_channels,
    )
    tensor.validate()
    return tensor


def _ldaps_safe_alignment(
    left_u: np.ndarray,
    left_v: np.ndarray,
    right_u: np.ndarray,
    right_v: np.ndarray,
) -> np.ndarray:
    denominator = np.maximum(
        np.hypot(left_u, left_v) * np.hypot(right_u, right_v),
        0.25,
    )
    return np.clip(
        (left_u * right_u + left_v * right_v) / denominator,
        -1.0,
        1.0,
    ).astype(np.float32)


def _ldaps_centered_slope(values: np.ndarray, radius: int) -> np.ndarray:
    steps = np.arange(values.shape[1])
    left = np.maximum(steps - radius, 0)
    right = np.minimum(steps + radius, values.shape[1] - 1)
    elapsed = (right - left).astype(np.float32)
    return ((values[:, right] - values[:, left]) / elapsed[None, :, None, None]).astype(
        np.float32
    )


def _ldaps_centered_missing(missing: np.ndarray, radius: int) -> np.ndarray:
    steps = np.arange(missing.shape[1])
    left = np.maximum(steps - radius, 0)
    right = np.minimum(steps + radius, missing.shape[1] - 1)
    return missing[:, left] | missing[:, right]


def _ldaps_lag_delta(values: np.ndarray) -> np.ndarray:
    delta = np.empty_like(values, dtype=np.float32)
    delta[:, 1:] = values[:, 1:] - values[:, :-1]
    delta[:, 0] = values[:, 1] - values[:, 0]
    return delta


def _ldaps_lag_missing(missing: np.ndarray) -> np.ndarray:
    paired = np.empty_like(missing, dtype=bool)
    paired[:, 1:] = missing[:, 1:] | missing[:, :-1]
    paired[:, 0] = missing[:, 0] | missing[:, 1]
    return paired


def _ldaps_roll3(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(values, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
    windows = np.stack(
        [padded[:, offset : offset + values.shape[1]] for offset in range(3)],
        axis=0,
    )
    return windows.mean(axis=0).astype(np.float32), windows.std(axis=0).astype(np.float32)


def _ldaps_broadcast_spatial(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*values.shape, *mask.shape), dtype=np.float32)
    output[..., mask] = values[..., None]
    return output


def _ldaps_spatial_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values[..., mask]


def _ldaps_spatial_plane(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.where(mask)
    design = np.column_stack(
        [columns.astype(np.float32), rows.astype(np.float32), np.ones(len(rows))]
    )
    pseudo_inverse = np.linalg.pinv(design).astype(np.float32)
    coefficients = np.einsum(
        "itn,kn->itk",
        _ldaps_spatial_values(values, mask),
        pseudo_inverse,
    )
    return (
        _ldaps_broadcast_spatial(coefficients[..., 0], mask),
        _ldaps_broadcast_spatial(coefficients[..., 1], mask),
    )


def _ldaps_spatial_missing(missing: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return _ldaps_broadcast_spatial(
        _ldaps_spatial_values(missing, mask).any(axis=-1),
        mask,
    ).astype(bool)


def build_ldaps_derived_family_tensor(
    frame: pd.DataFrame,
    family: str,
    labels: pd.DataFrame | None = None,
) -> SourceIssueTensor:
    if family not in LDAPS_DERIVED_FAMILY_CHANNELS:
        raise ValueError(f"Unknown LDAPS derived family: {family}")
    required = ldaps_derived_family_required_columns(family)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"LDAPS {family} missing columns: {missing_columns}")

    if family == "density_spatial":
        density = build_ldaps_derived_family_tensor(frame, "density_power", labels=labels)
        spatial = build_ldaps_derived_family_tensor(frame, "spatial_flow", labels=labels)
        if not np.array_equal(density.forecast_times, spatial.forecast_times):
            raise ValueError("LDAPS density/spatial forecast times differ")
        core_channels = len(LDAPS_CORE_SPEC.output_channels)
        tensor = replace(
            density,
            source="ldaps_density_spatial_derived_core",
            values=np.concatenate(
                [density.values, spatial.values[:, :, core_channels:]], axis=2
            ).astype(np.float32),
            missing_mask=np.concatenate(
                [
                    density.missing_mask,
                    spatial.missing_mask[:, :, core_channels:],
                ],
                axis=2,
            ),
            channel_names=(
                *LDAPS_CORE_SPEC.output_channels,
                *LDAPS_DERIVED_FAMILY_CHANNELS["density_spatial"],
            ),
        )
        tensor.validate()
        return tensor

    extra_vectors = (
        (LDAPS_5M_CORE_SPEC.vectors[-1],) if family == "vertical_profile" else ()
    )
    vector_raw = {
        channel for vector in extra_vectors for channel in (vector.u, vector.v)
    }
    extra_scalars = tuple(
        channel
        for channel in LDAPS_DERIVED_RAW_CHANNELS[family]
        if channel not in vector_raw
    )
    parent_spec = GridSourceSpec(
        name=f"ldaps_{family}_derived_parent",
        layout=LDAPS_CORE_SPEC.layout,
        vectors=(*LDAPS_CORE_SPEC.vectors, *extra_vectors),
        scalar_channels=extra_scalars,
    )
    parent = build_grid_source_core_tensor(frame, parent_spec, labels=labels)
    channel_index = {
        channel: index for index, channel in enumerate(parent.channel_names)
    }

    def value(channel: str) -> np.ndarray:
        return parent.values[:, :, channel_index[channel]]

    def missing(channel: str) -> np.ndarray:
        return parent.missing_mask[:, :, channel_index[channel]]

    def combined_missing(*channels: str) -> np.ndarray:
        result = np.zeros_like(parent.values[:, :, 0], dtype=bool)
        for channel in channels:
            result |= missing(channel)
        return result

    umax_name = "heightAboveGround_50_50MUmax"
    vmax_name = "heightAboveGround_50_50MVmax"
    umin_name = "heightAboveGround_50_50MUmin"
    vmin_name = "heightAboveGround_50_50MVmin"
    u10_name = "heightAboveGround_10_10u"
    v10_name = "heightAboveGround_10_10v"
    envelope_inputs = (umax_name, vmax_name, umin_name, vmin_name)
    umax, vmax = value(umax_name), value(vmax_name)
    umin, vmin = value(umin_name), value(vmin_name)
    u10, v10 = value(u10_name), value(v10_name)
    ws10 = value("wind_10m_speed")
    mid_u = 0.5 * (umax + umin)
    mid_v = 0.5 * (vmax + vmin)
    ws50mid = np.hypot(mid_u, mid_v).astype(np.float32)
    envelope_u = umax - umin
    envelope_v = vmax - vmin
    envelope_norm = np.hypot(envelope_u, envelope_v).astype(np.float32)
    envelope_missing = combined_missing(*envelope_inputs)
    wind_missing = combined_missing(*envelope_inputs, u10_name, v10_name)
    mask = parent.spatial_mask
    derived: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(
        name: str,
        values: np.ndarray,
        missing_values: np.ndarray,
    ) -> None:
        derived[name] = (
            np.asarray(values, dtype=np.float32),
            np.asarray(missing_values, dtype=bool),
        )

    if family == "envelope":
        add("ldaps_50m_mid_u", mid_u, combined_missing(umax_name, umin_name))
        add("ldaps_50m_mid_v", mid_v, combined_missing(vmax_name, vmin_name))
        add("ldaps_50m_mid_speed", ws50mid, envelope_missing)
        add("ldaps_50m_envelope_u", envelope_u, combined_missing(umax_name, umin_name))
        add("ldaps_50m_envelope_v", envelope_v, combined_missing(vmax_name, vmin_name))
        add("ldaps_50m_envelope_norm", envelope_norm, envelope_missing)
        add(
            "ldaps_50m_relative_envelope",
            envelope_norm / np.maximum(ws50mid, 1.0),
            envelope_missing,
        )

    elif family == "vertical_profile":
        u5_name = "heightAboveGround_5_XBLWS"
        v5_name = "heightAboveGround_5_YBLWS"
        u5, v5 = value(u5_name), value(v5_name)
        ws5 = value("wind_5m_speed")
        five_missing = combined_missing(u5_name, v5_name)
        all_vertical_missing = wind_missing | five_missing
        add("ldaps_50m_mid_minus_10m_speed", ws50mid - ws10, wind_missing)
        add(
            "ldaps_50m_mid_10m_log_shear",
            np.log(np.maximum(ws50mid, 0.5) / np.maximum(ws10, 0.5))
            / np.log(5.0),
            wind_missing,
        )
        add(
            "ldaps_50m_mid_10m_alignment",
            _ldaps_safe_alignment(mid_u, mid_v, u10, v10),
            wind_missing,
        )
        add("ldaps_10m_minus_5m_speed", ws10 - ws5, all_vertical_missing)
        add(
            "ldaps_10m_5m_log_shear",
            np.log(np.maximum(ws10, 0.5) / np.maximum(ws5, 0.5)) / np.log(2.0),
            all_vertical_missing,
        )
        add(
            "ldaps_10m_5m_alignment",
            _ldaps_safe_alignment(u10, v10, u5, v5),
            all_vertical_missing,
        )

    elif family == "density_power":
        temperature_name = "heightAboveGround_2_t"
        humidity_name = "heightAboveGround_2_q"
        density_inputs = (temperature_name, humidity_name, LDAPS_SURFACE_PRESSURE_COLUMN)
        temperature = np.maximum(value(temperature_name), 180.0)
        specific_humidity = np.clip(value(humidity_name), 0.0, 0.1)
        virtual_temperature = temperature * (1.0 + 0.61 * specific_humidity)
        density = np.maximum(value(LDAPS_SURFACE_PRESSURE_COLUMN), 10000.0) / (
            287.05 * virtual_temperature
        )
        density_factor = np.cbrt(np.maximum(density, 0.1) / 1.225)
        density_missing = combined_missing(*density_inputs)
        add("ldaps_virtual_temperature", virtual_temperature, density_missing)
        add("ldaps_air_density", density, density_missing)
        add(
            "ldaps_density_equivalent_ws10",
            ws10 * density_factor,
            density_missing | missing("wind_10m_speed"),
        )
        add(
            "ldaps_density_equivalent_ws50mid",
            ws50mid * density_factor,
            density_missing | envelope_missing,
        )
        add(
            "ldaps_power_density_ws10",
            0.5 * density * np.power(ws10, 3),
            density_missing | missing("wind_10m_speed"),
        )
        add(
            "ldaps_power_density_ws50mid",
            0.5 * density * np.power(ws50mid, 3),
            density_missing | envelope_missing,
        )

    elif family == "pbl_stability":
        blh = np.maximum(value(LDAPS_BLH_COLUMN), LDAPS_BLH_FLOOR_M)
        blh_missing = missing(LDAPS_BLH_COLUMN)
        hub_over_blh = TURBINE_HUB_HEIGHT_M / blh
        shear = np.log(np.maximum(ws50mid, 0.5) / np.maximum(ws10, 0.5)) / np.log(
            5.0
        )
        add("ldaps_hub_over_blh_family", hub_over_blh, blh_missing)
        add(
            "ldaps_blh_tendency_3h_family",
            _ldaps_centered_slope(blh, 3),
            _ldaps_centered_missing(blh_missing, 3),
        )
        add(
            "ldaps_envelope_x_hub_over_blh",
            envelope_norm * hub_over_blh,
            envelope_missing | blh_missing,
        )
        add(
            "ldaps_shear_x_hub_over_blh",
            shear * hub_over_blh,
            wind_missing | blh_missing,
        )
        add(
            "ldaps_ventilation_proxy",
            blh * ws10,
            blh_missing | missing("wind_10m_speed"),
        )

    elif family == "spatial_flow":
        ws10_flat = _ldaps_spatial_values(ws10, mask)
        mid_flat = _ldaps_spatial_values(ws50mid, mask)
        mean_ws10 = ws10_flat.mean(axis=-1)
        mean_mid = mid_flat.mean(axis=-1)
        spatial_missing = _ldaps_spatial_missing(wind_missing, mask)
        gradient_x, gradient_y = _ldaps_spatial_plane(ws10, mask)
        mean_u10 = _ldaps_spatial_values(u10, mask).mean(axis=-1)
        mean_v10 = _ldaps_spatial_values(v10, mask).mean(axis=-1)
        add("ldaps_spatial_ws10_mean", _ldaps_broadcast_spatial(mean_ws10, mask), spatial_missing)
        add("ldaps_spatial_ws10_std", _ldaps_broadcast_spatial(ws10_flat.std(axis=-1), mask), spatial_missing)
        add(
            "ldaps_spatial_ws10_p90_p10",
            _ldaps_broadcast_spatial(
                np.quantile(ws10_flat, 0.9, axis=-1)
                - np.quantile(ws10_flat, 0.1, axis=-1),
                mask,
            ),
            spatial_missing,
        )
        add(
            "ldaps_spatial_ws10_coherence",
            _ldaps_broadcast_spatial(
                np.hypot(mean_u10, mean_v10) / np.maximum(mean_ws10, 0.1),
                mask,
            ),
            spatial_missing,
        )
        add("ldaps_spatial_ws10_gradient_x", gradient_x, spatial_missing)
        add("ldaps_spatial_ws10_gradient_y", gradient_y, spatial_missing)
        add("ldaps_spatial_ws50mid_mean", _ldaps_broadcast_spatial(mean_mid, mask), spatial_missing)
        add("ldaps_spatial_ws50mid_std", _ldaps_broadcast_spatial(mid_flat.std(axis=-1), mask), spatial_missing)
        add(
            "ldaps_spatial_ws50mid_p90_p10",
            _ldaps_broadcast_spatial(
                np.quantile(mid_flat, 0.9, axis=-1)
                - np.quantile(mid_flat, 0.1, axis=-1),
                mask,
            ),
            spatial_missing,
        )

    elif family == "terrain_interaction":
        terrain_name = "surface_0_h"
        terrain = value(terrain_name)
        terrain_missing = missing(terrain_name)
        terrain_summary_missing = _ldaps_spatial_missing(terrain_missing, mask)
        terrain_mean = _ldaps_broadcast_spatial(
            _ldaps_spatial_values(terrain, mask).mean(axis=-1), mask
        )
        slope_x, slope_y = _ldaps_spatial_plane(terrain, mask)
        slope_magnitude = np.hypot(slope_x, slope_y).astype(np.float32)
        alignment = np.where(
            slope_magnitude > 1e-6,
            (u10 * slope_x + v10 * slope_y)
            / np.maximum(ws10 * slope_magnitude, 1e-6),
            0.0,
        )
        alignment = np.clip(alignment, -1.0, 1.0)
        terrain_wind_missing = terrain_summary_missing | combined_missing(u10_name, v10_name)
        add("ldaps_terrain_relative_height", terrain - terrain_mean, terrain_missing | terrain_summary_missing)
        add("ldaps_terrain_slope_x", slope_x, terrain_summary_missing)
        add("ldaps_terrain_slope_y", slope_y, terrain_summary_missing)
        add("ldaps_terrain_slope_magnitude", slope_magnitude, terrain_summary_missing)
        add("ldaps_terrain_wind_alignment", alignment, terrain_wind_missing)
        add(
            "ldaps_terrain_wind_exposure",
            ws10 * alignment * slope_magnitude,
            terrain_wind_missing,
        )

    elif family == "temporal_trajectory":
        previous_u10 = np.empty_like(u10)
        previous_v10 = np.empty_like(v10)
        previous_u10[:, 1:] = u10[:, :-1]
        previous_v10[:, 1:] = v10[:, :-1]
        previous_u10[:, 0] = u10[:, 1]
        previous_v10[:, 0] = v10[:, 1]
        temporal_missing = _ldaps_lag_missing(combined_missing(u10_name, v10_name))
        ws10_roll_mean, ws10_roll_std = _ldaps_roll3(ws10)
        ws50_roll_mean, _ = _ldaps_roll3(ws50mid)
        add("ldaps_ws10_ramp_1h", _ldaps_lag_delta(ws10), temporal_missing)
        add(
            "ldaps_ws10_vector_ramp_1h",
            np.hypot(u10 - previous_u10, v10 - previous_v10),
            temporal_missing,
        )
        add(
            "ldaps_ws10_direction_turn_1h",
            1.0 - _ldaps_safe_alignment(u10, v10, previous_u10, previous_v10),
            temporal_missing,
        )
        add("ldaps_ws50mid_ramp_1h", _ldaps_lag_delta(ws50mid), _ldaps_lag_missing(envelope_missing))
        add("ldaps_envelope_ramp_1h", _ldaps_lag_delta(envelope_norm), _ldaps_lag_missing(envelope_missing))
        add("ldaps_ws10_roll3_mean", ws10_roll_mean, missing("wind_10m_speed"))
        add("ldaps_ws10_roll3_std", ws10_roll_std, missing("wind_10m_speed"))
        add("ldaps_ws50mid_roll3_mean", ws50_roll_mean, envelope_missing)
        issue_range = ws10.max(axis=1) - ws10.min(axis=1)
        issue_range = np.broadcast_to(issue_range[:, None], ws10.shape)
        issue_missing = np.broadcast_to(
            combined_missing(u10_name, v10_name).any(axis=1)[:, None], ws10.shape
        )
        peak_index = ws10.argmax(axis=1).astype(np.float32)
        current_index = np.arange(ws10.shape[1], dtype=np.float32)[None, :, None, None]
        hours_to_peak = peak_index[:, None] - current_index
        add("ldaps_ws10_issue_range", issue_range, issue_missing)
        add("ldaps_ws10_hours_to_peak", hours_to_peak, issue_missing)

    elif family == "thermodynamic_regime":
        temperature_name = "heightAboveGround_2_t"
        dewpoint_name = "heightAboveGround_2_dpt"
        rh_name = "heightAboveGround_2_r"
        q_name = "heightAboveGround_2_q"
        temperature = value(temperature_name)
        dewpoint = value(dewpoint_name)
        rh = value(rh_name)
        pressure = value(LDAPS_SURFACE_PRESSURE_COLUMN)
        mslp = value(LDAPS_MSLP_COLUMN)
        pressure_mean = _ldaps_broadcast_spatial(
            _ldaps_spatial_values(pressure, mask).mean(axis=-1), mask
        )
        q = value(q_name)
        q_mean = _ldaps_broadcast_spatial(
            _ldaps_spatial_values(q, mask).mean(axis=-1), mask
        )
        pressure_gradient_x, pressure_gradient_y = _ldaps_spatial_plane(pressure, mask)
        temperature_gradient_x, temperature_gradient_y = _ldaps_spatial_plane(
            temperature, mask
        )
        thermo_missing = combined_missing(
            temperature_name,
            dewpoint_name,
            rh_name,
            q_name,
            LDAPS_SURFACE_PRESSURE_COLUMN,
            LDAPS_MSLP_COLUMN,
        )
        spatial_thermo_missing = _ldaps_spatial_missing(thermo_missing, mask)
        add("ldaps_dewpoint_depression", temperature - dewpoint, thermo_missing)
        add("ldaps_relative_humidity_deficit", 100.0 - np.clip(rh, 0.0, 100.0), thermo_missing)
        add("ldaps_mslp_surface_gap", mslp - pressure, thermo_missing)
        add("ldaps_surface_pressure_anomaly", pressure - pressure_mean, thermo_missing | spatial_thermo_missing)
        add("ldaps_surface_pressure_gradient_x", pressure_gradient_x, spatial_thermo_missing)
        add("ldaps_surface_pressure_gradient_y", pressure_gradient_y, spatial_thermo_missing)
        add(
            "ldaps_temperature_gradient_magnitude",
            np.hypot(temperature_gradient_x, temperature_gradient_y),
            spatial_thermo_missing,
        )
        add("ldaps_specific_humidity_anomaly", q - q_mean, thermo_missing | spatial_thermo_missing)

    elif family == "weather_regime":
        direct_name = "heightAboveGround_2_SWDIR"
        diffuse_name = "heightAboveGround_2_SWDIF"
        ndnsw_name = "surface_0_NDNSW"
        ndnlw_name = "surface_0_NDNLW"
        cloud_names = ("etc_0_hcc", "etc_0_mcc", "etc_0_lcc", "etc_0_VLCDC")
        precip_names = ("surface_0_avg_lsprate", "surface_0_lssrate", "surface_0_ncpcp")
        snow_names = ("surface_0_snol", "surface_0_SNOM")
        direct, diffuse = value(direct_name), value(diffuse_name)
        shortwave = direct + diffuse
        clouds = [np.maximum(value(channel), 0.0) for channel in cloud_names]
        cloud_sum = np.sum(clouds, axis=0)
        low_cloud = clouds[2] + clouds[3]
        precip_positive = [np.maximum(value(channel), 0.0) for channel in precip_names]
        snow_positive = [np.maximum(value(channel), 0.0) for channel in snow_names]
        precip_activity = np.sum(precip_positive, axis=0)
        snow_activity = np.sum(snow_positive, axis=0)
        shortwave_missing = combined_missing(direct_name, diffuse_name)
        radiation_missing = combined_missing(ndnsw_name, ndnlw_name)
        cloud_missing = combined_missing(*cloud_names)
        precip_missing = combined_missing(*precip_names)
        snow_missing = combined_missing(*snow_names)
        add("ldaps_shortwave_total", shortwave, shortwave_missing)
        add(
            "ldaps_shortwave_direct_fraction",
            np.where(shortwave > 1.0, direct / np.maximum(shortwave, 1.0), 0.0),
            shortwave_missing,
        )
        add("ldaps_net_radiation_proxy", value(ndnsw_name) + value(ndnlw_name), radiation_missing)
        add("ldaps_cloud_layer_mean", cloud_sum / float(len(cloud_names)), cloud_missing)
        add("ldaps_low_cloud_dominance", low_cloud / np.maximum(cloud_sum, 1e-6), cloud_missing)
        add("ldaps_precip_log_activity", np.log1p(precip_activity), precip_missing)
        add("ldaps_precip_occurrence", (precip_activity > 0.0).astype(np.float32), precip_missing)
        add("ldaps_snow_log_activity", np.log1p(snow_activity), snow_missing)
        add("ldaps_snow_occurrence", (snow_activity > 0.0).astype(np.float32), snow_missing)

    expected_channels = LDAPS_DERIVED_FAMILY_CHANNELS[family]
    if tuple(derived) != expected_channels:
        raise RuntimeError(
            f"LDAPS {family} derived channels differ: {tuple(derived)}"
        )
    value_channels = [
        parent.values[:, :, channel_index[channel]]
        for channel in LDAPS_CORE_SPEC.output_channels
    ]
    missing_channels = [
        parent.missing_mask[:, :, channel_index[channel]]
        for channel in LDAPS_CORE_SPEC.output_channels
    ]
    value_channels.extend(derived[channel][0] for channel in expected_channels)
    missing_channels.extend(derived[channel][1] for channel in expected_channels)
    values = np.stack(value_channels, axis=2).astype(np.float32)
    values *= mask.reshape(1, 1, 1, *mask.shape)
    tensor = replace(
        parent,
        source=f"ldaps_{family}_derived_core",
        values=values,
        missing_mask=np.stack(missing_channels, axis=2),
        channel_names=(*LDAPS_CORE_SPEC.output_channels, *expected_channels),
    )
    tensor.validate()
    return tensor


def _nearest_axis(values: Iterable[float], target: float, count: int) -> np.ndarray:
    unique = np.unique(np.asarray(list(values), dtype=np.float64))
    if len(unique) < count:
        raise ValueError(f"grid axis has {len(unique)} points, fewer than crop size {count}")
    selected = unique[np.argsort(np.abs(unique - float(target)))[:count]]
    return np.sort(selected)


def _gefs_component_tensor(
    frame: pd.DataFrame,
    source: str,
    raw_channels: tuple[str, ...],
    vectors: tuple[VectorChannels, ...],
    crop_size: int,
    labels: pd.DataFrame | None = None,
) -> SourceIssueTensor:
    required = ("run_date", "fhour", "lat", "lon", *raw_channels)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{source} missing columns: {missing_columns}")
    work = frame[list(required)].copy()
    work["run_date"] = pd.to_datetime(work["run_date"]).dt.normalize()
    work["fhour"] = pd.to_numeric(work["fhour"], errors="raise").astype(np.int16)
    work["lat"] = pd.to_numeric(work["lat"], errors="raise")
    work["lon"] = pd.to_numeric(work["lon"], errors="raise")
    unexpected_fhours = sorted(set(work["fhour"].unique()) - set(GEFS_FHOURS.tolist()))
    if unexpected_fhours:
        raise ValueError(f"{source} contains unexpected forecast hours: {unexpected_fhours}")

    latitudes = _nearest_axis(work["lat"].unique(), FARM_LATITUDE, crop_size)[::-1]
    longitudes = _nearest_axis(work["lon"].unique(), FARM_LONGITUDE, crop_size)
    cropped = work.loc[
        work["lat"].isin(latitudes) & work["lon"].isin(longitudes)
    ].copy()
    if cropped.duplicated(["run_date", "fhour", "lat", "lon"]).any():
        raise ValueError(f"{source} contains duplicate run/hour/grid rows")

    run_dates = np.sort(cropped["run_date"].unique()).astype("datetime64[ns]")
    expected_index = pd.MultiIndex.from_product(
        [pd.to_datetime(run_dates), GEFS_FHOURS], names=["run_date", "fhour"]
    )
    grid_pairs = [(float(lat), float(lon)) for lat in latitudes for lon in longitudes]
    pivot = cropped.pivot(
        index=["run_date", "fhour"],
        columns=["lat", "lon"],
        values=list(raw_channels),
    ).reindex(index=expected_index)
    expected_columns = pd.MultiIndex.from_tuples(
        [(channel, lat, lon) for channel in raw_channels for lat, lon in grid_pairs]
    )
    pivot = pivot.reindex(columns=expected_columns)
    raw = pivot.to_numpy(np.float32).reshape(
        len(run_dates), len(GEFS_FHOURS), len(raw_channels), crop_size, crop_size
    )
    source_missing = np.isnan(raw)
    source_filled = _interpolate_issue_time(raw)

    source_leads = GEFS_FHOURS.astype(np.float32) - 10.0
    target_parts = []
    missing_parts = []
    for target_lead in EXPECTED_LEADS.astype(np.float32):
        upper = int(np.searchsorted(source_leads, target_lead, side="left"))
        if upper == 0 or upper == len(source_leads):
            raise ValueError(f"{source} cannot interpolate target lead {target_lead}")
        if source_leads[upper] == target_lead:
            lower = upper
            weight = 0.0
        else:
            lower = upper - 1
            weight = float(
                (target_lead - source_leads[lower])
                / (source_leads[upper] - source_leads[lower])
            )
        target_parts.append(
            (1.0 - weight) * source_filled[:, lower] + weight * source_filled[:, upper]
        )
        missing_parts.append(source_missing[:, lower] | source_missing[:, upper])
    hourly = np.stack(target_parts, axis=1).astype(np.float32)
    hourly_missing = np.stack(missing_parts, axis=1)

    raw_index = {channel: index for index, channel in enumerate(raw_channels)}
    channel_arrays = [hourly]
    missing_arrays = [hourly_missing]
    for vector in vectors:
        u_index = raw_index[vector.u]
        v_index = raw_index[vector.v]
        channel_arrays.append(
            np.hypot(hourly[:, :, u_index], hourly[:, :, v_index])[:, :, None]
        )
        missing_arrays.append(
            (hourly_missing[:, :, u_index] | hourly_missing[:, :, v_index])[:, :, None]
        )
    values = np.concatenate(channel_arrays, axis=2).astype(np.float32)
    missing_mask = np.concatenate(missing_arrays, axis=2)
    channel_names = (*raw_channels, *(f"{vector.name}_speed" for vector in vectors))

    issue_times = (pd.to_datetime(run_dates) + pd.Timedelta(days=1, hours=13)).to_numpy(
        dtype="datetime64[ns]"
    )
    forecast_times = np.asarray(
        [
            [pd.Timestamp(issue) + pd.Timedelta(hours=int(step)) for step in EXPECTED_LEADS]
            for issue in issue_times
        ],
        dtype="datetime64[ns]",
    )
    year_matrix = pd.to_datetime(forecast_times.reshape(-1)).year.to_numpy().reshape(
        forecast_times.shape
    )
    tensor = SourceIssueTensor(
        source=source,
        values=values,
        missing_mask=missing_mask,
        spatial_mask=np.ones((crop_size, crop_size), dtype=bool),
        channel_names=channel_names,
        forecast_times=forecast_times,
        issue_times=issue_times,
        years=year_matrix.astype(np.int16),
        leads=EXPECTED_LEADS.copy(),
        time_features=_time_features(forecast_times, issue_times),
        targets=_target_array(labels, forecast_times),
        fallback_flags=np.zeros(len(issue_times), dtype=bool),
        latitudes=latitudes.astype(np.float32),
        longitudes=longitudes.astype(np.float32),
    )
    tensor.validate()
    return tensor


def _build_gefs_core_tensor(
    pressure: pd.DataFrame,
    gust: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    include_700: bool = False,
    include_spread: bool = False,
    spread_vectors: tuple[VectorChannels, ...] = (),
    include_gust_spread: bool = False,
    variant: str | None = None,
) -> GEFSIssueTensor:
    mean_vectors = (
        *GEFS_PRESSURE_MEAN_VECTORS,
        *((GEFS_PRESSURE_700_VECTOR,) if include_700 else ()),
    )
    vectors = (*mean_vectors, *spread_vectors)
    selected_spread_channels = (
        GEFS_PRESSURE_SPREAD_CHANNELS
        if include_spread
        else tuple(
            channel
            for vector in spread_vectors
            for channel in (vector.u, vector.v)
        )
    )
    pressure_raw = (
        *(channel for vector in mean_vectors for channel in (vector.u, vector.v)),
        *selected_spread_channels,
    )
    if variant is None:
        variant = "spread_core" if include_spread else "mean_core"
    pressure_tensor = _gefs_component_tensor(
        pressure,
        source=f"gefs_pressure_{variant}",
        raw_channels=pressure_raw,
        vectors=vectors,
        crop_size=7,
        labels=labels,
    )
    gust_raw = (
        *GEFS_GUST_MEAN_CHANNELS,
        *(
            GEFS_GUST_SPREAD_CHANNELS
            if include_spread or include_gust_spread
            else ()
        ),
    )
    gust_tensor = _gefs_component_tensor(
        gust,
        source=f"gefs_gust_{variant}",
        raw_channels=gust_raw,
        vectors=(),
        crop_size=9,
        labels=labels,
    )
    combined = GEFSIssueTensor(pressure=pressure_tensor, gust=gust_tensor)
    combined.validate()
    return combined


def build_gefs_mean_core_tensor(
    pressure: pd.DataFrame,
    gust: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    include_700: bool = False,
) -> GEFSIssueTensor:
    return _build_gefs_core_tensor(
        pressure,
        gust,
        labels=labels,
        include_700=include_700,
        include_spread=False,
    )


def build_gefs_spread_core_tensor(
    pressure: pd.DataFrame,
    gust: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> GEFSIssueTensor:
    return _build_gefs_core_tensor(
        pressure,
        gust,
        labels=labels,
        include_700=False,
        include_spread=True,
    )


def build_gefs_mean700_core_tensor(
    pressure: pd.DataFrame,
    gust: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> GEFSIssueTensor:
    return _build_gefs_core_tensor(
        pressure,
        gust,
        labels=labels,
        include_700=True,
        variant="mean700_core",
    )


def build_gefs_near_spread_core_tensor(
    pressure: pd.DataFrame,
    gust: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> GEFSIssueTensor:
    return _build_gefs_core_tensor(
        pressure,
        gust,
        labels=labels,
        spread_vectors=GEFS_NEAR_SPREAD_VECTORS,
        include_gust_spread=True,
        variant="near_spread_core",
    )


def build_gefs_upper_spread_core_tensor(
    pressure: pd.DataFrame,
    gust: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> GEFSIssueTensor:
    return _build_gefs_core_tensor(
        pressure,
        gust,
        labels=labels,
        spread_vectors=GEFS_UPPER_SPREAD_VECTORS,
        variant="upper_spread_core",
    )


def gefs_publication_audit(
    root: Path | str,
    kind: str = "geavg",
) -> pd.DataFrame:
    root = Path(root)
    paths = sorted((root / "meta").glob("gefs_meta_*.csv"))
    if not paths:
        raise FileNotFoundError(f"GEFS metadata not found under {root}")
    meta = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in paths],
        ignore_index=True,
    )
    required = ("run_date", "product", "kind", "fhour", "last_modified")
    missing = [column for column in required if column not in meta.columns]
    if missing:
        raise ValueError(f"GEFS metadata missing columns: {missing}")
    selected = meta.loc[
        meta["kind"].eq(kind)
        & meta["product"].isin(["pa", "s"])
        & pd.to_numeric(meta["fhour"], errors="coerce").isin(GEFS_FHOURS)
    ].copy()
    selected["run_date"] = pd.to_datetime(selected["run_date"]).dt.normalize()
    selected["published_kst"] = (
        pd.to_datetime(selected["last_modified"], utc=True, errors="coerce")
        .dt.tz_convert("Asia/Seoul")
        .dt.tz_localize(None)
    )
    selected["object_key"] = (
        selected["product"].astype(str)
        + ":"
        + selected["kind"].astype(str)
        + ":"
        + selected["fhour"].astype(str)
    )
    grouped = selected.groupby("run_date", as_index=False).agg(
        object_count=("object_key", "nunique"),
        published_kst=("published_kst", "max"),
        missing_publication=("published_kst", lambda values: int(values.isna().sum())),
    )
    grouped["data_available_kst_dtm"] = grouped["run_date"] + pd.Timedelta(
        days=1, hours=13
    )
    grouped["publication_margin_hours"] = (
        grouped["data_available_kst_dtm"] - grouped["published_kst"]
    ).dt.total_seconds() / 3600.0
    expected_objects = len(GEFS_FHOURS) * 2
    grouped["safe"] = (
        grouped["object_count"].eq(expected_objects)
        & grouped["missing_publication"].eq(0)
        & grouped["publication_margin_hours"].gt(0.0)
    )
    grouped["kind"] = kind
    return grouped.sort_values("run_date").reset_index(drop=True)


def _apply_fallback_to_tensor(
    tensor: SourceIssueTensor,
    safe_by_issue: dict[pd.Timestamp, bool],
) -> SourceIssueTensor:
    values = tensor.values.copy()
    missing_mask = tensor.missing_mask.copy()
    fallback = np.zeros(len(tensor.issue_times), dtype=bool)
    last_safe_index: int | None = None
    for index, issue_time in enumerate(pd.to_datetime(tensor.issue_times)):
        safe = bool(safe_by_issue.get(pd.Timestamp(issue_time), False))
        if safe:
            last_safe_index = index
            continue
        if last_safe_index is None:
            raise ValueError(f"{tensor.source}: no prior safe GEFS issue for {issue_time}")
        values[index] = values[last_safe_index]
        missing_mask[index] = missing_mask[last_safe_index]
        fallback[index] = True
    out = replace(
        tensor,
        values=values,
        missing_mask=missing_mask,
        fallback_flags=fallback,
    )
    out.validate()
    return out


def apply_gefs_publication_fallback(
    tensor: GEFSIssueTensor,
    publication: pd.DataFrame,
) -> GEFSIssueTensor:
    required = ("data_available_kst_dtm", "safe")
    missing = [column for column in required if column not in publication.columns]
    if missing:
        raise ValueError(f"GEFS publication table missing columns: {missing}")
    safe_by_issue = {
        pd.Timestamp(issue): bool(safe)
        for issue, safe in zip(publication["data_available_kst_dtm"], publication["safe"])
    }
    out = GEFSIssueTensor(
        pressure=_apply_fallback_to_tensor(tensor.pressure, safe_by_issue),
        gust=_apply_fallback_to_tensor(tensor.gust, safe_by_issue),
    )
    out.validate()
    return out


def select_source_issues(
    tensor: SourceIssueTensor,
    issue_times: Iterable[np.datetime64 | pd.Timestamp],
) -> SourceIssueTensor:
    requested = pd.to_datetime(list(issue_times)).to_numpy(dtype="datetime64[ns]")
    positions = pd.Index(tensor.issue_times).get_indexer(requested)
    if (positions < 0).any():
        missing = requested[positions < 0]
        raise ValueError(f"{tensor.source}: missing requested issues {missing[:5].tolist()}")
    out = replace(
        tensor,
        values=tensor.values[positions],
        missing_mask=tensor.missing_mask[positions],
        forecast_times=tensor.forecast_times[positions],
        issue_times=tensor.issue_times[positions],
        years=tensor.years[positions],
        time_features=tensor.time_features[positions],
        targets=None if tensor.targets is None else tensor.targets[positions],
        fallback_flags=(
            None if tensor.fallback_flags is None else tensor.fallback_flags[positions]
        ),
    )
    out.validate()
    return out


def select_gefs_issues(
    tensor: GEFSIssueTensor,
    issue_times: Iterable[np.datetime64 | pd.Timestamp],
) -> GEFSIssueTensor:
    out = GEFSIssueTensor(
        pressure=select_source_issues(tensor.pressure, issue_times),
        gust=select_source_issues(tensor.gust, issue_times),
    )
    out.validate()
    return out


def fit_source_channel_scaler(
    tensor: SourceIssueTensor,
    issue_mask: np.ndarray | None = None,
) -> SourceChannelScaler:
    if issue_mask is None:
        issue_mask = np.ones(len(tensor.issue_times), dtype=bool)
    issue_mask = np.asarray(issue_mask, dtype=bool)
    if issue_mask.shape != (len(tensor.issue_times),) or not issue_mask.any():
        raise ValueError("issue_mask must select at least one source issue")
    spatial = tensor.spatial_mask.astype(bool)
    values = tensor.values[issue_mask][..., spatial]
    missing = tensor.missing_mask[issue_mask][..., spatial]
    means = []
    scales = []
    for channel_index in range(len(tensor.channel_names)):
        channel = values[:, :, channel_index].reshape(-1)
        channel_missing = missing[:, :, channel_index].reshape(-1)
        observed = channel[~channel_missing]
        if len(observed) == 0:
            raise ValueError(f"{tensor.source}: no observed values for {tensor.channel_names[channel_index]}")
        mean = float(observed.mean(dtype=np.float64))
        scale = float(observed.std(dtype=np.float64))
        means.append(mean)
        scales.append(scale if scale >= 1e-6 else 1.0)
    return SourceChannelScaler(
        channel_names=tensor.channel_names,
        mean=np.asarray(means, dtype=np.float32),
        scale=np.asarray(scales, dtype=np.float32),
    )


def transform_source_channels(
    tensor: SourceIssueTensor,
    scaler: SourceChannelScaler,
) -> np.ndarray:
    if tensor.channel_names != scaler.channel_names:
        raise ValueError(f"{tensor.source}: scaler channels differ from tensor channels")
    transformed = (
        tensor.values - scaler.mean.reshape(1, 1, -1, 1, 1)
    ) / scaler.scale.reshape(1, 1, -1, 1, 1)
    return (transformed * tensor.spatial_mask.reshape(1, 1, 1, *tensor.spatial_mask.shape)).astype(
        np.float32
    )


def load_gefs_core_frames(
    root: Path | str,
    include_spread: bool = False,
    include_700: bool = False,
    spread_vectors: tuple[VectorChannels, ...] = (),
    include_gust_spread: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    pressure_paths = sorted((root / "parquet").glob("gefs_pa_*.parquet"))
    gust_paths = sorted((root / "parquet").glob("gefs_gust_*.parquet"))
    if not pressure_paths or not gust_paths:
        raise FileNotFoundError(f"GEFS parquet files not found under {root}")
    mean_vectors = (
        *GEFS_PRESSURE_MEAN_VECTORS,
        *((GEFS_PRESSURE_700_VECTOR,) if include_700 else ()),
    )
    selected_spread_channels = (
        GEFS_PRESSURE_SPREAD_CHANNELS
        if include_spread
        else tuple(
            channel
            for vector in spread_vectors
            for channel in (vector.u, vector.v)
        )
    )
    pressure_columns = tuple(dict.fromkeys((
        "run_date",
        "fhour",
        "lat",
        "lon",
        *(channel for vector in mean_vectors for channel in (vector.u, vector.v)),
        *selected_spread_channels,
    )))
    gust_columns = tuple(dict.fromkeys((
        "run_date",
        "fhour",
        "lat",
        "lon",
        *GEFS_GUST_MEAN_CHANNELS,
        *(
            GEFS_GUST_SPREAD_CHANNELS
            if include_spread or include_gust_spread
            else ()
        ),
    )))
    pressure = pd.concat(
        [pd.read_parquet(path, columns=list(pressure_columns)) for path in pressure_paths],
        ignore_index=True,
    )
    gust = pd.concat(
        [pd.read_parquet(path, columns=list(gust_columns)) for path in gust_paths],
        ignore_index=True,
    )
    return pressure, gust
