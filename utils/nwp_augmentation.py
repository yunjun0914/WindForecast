from __future__ import annotations

import numpy as np
import pandas as pd


LDAPS_WIND_PAIRS = (
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
    ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin"),
    ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
)

GFS_WIND_PAIRS = (
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    ("heightAboveGround_80_u", "heightAboveGround_80_v"),
    ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
    ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
    ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
    ("isobaricInhPa_500_u", "isobaricInhPa_500_v"),
)


def _issue_parameters(
    issue_times: pd.Series,
    seed: int,
    speed_scale_std: float,
    direction_std_deg: float,
    max_speed_scale_delta: float,
    max_direction_deg: float,
) -> pd.DataFrame:
    issues = pd.DatetimeIndex(pd.to_datetime(issue_times).dropna().unique()).sort_values()
    if issues.empty:
        raise ValueError("No NWP issue times were found")
    rng = np.random.default_rng(seed)
    scales = 1.0 + rng.normal(0.0, speed_scale_std, size=len(issues))
    scales = np.clip(
        scales,
        1.0 - max_speed_scale_delta,
        1.0 + max_speed_scale_delta,
    )
    angles = rng.normal(0.0, direction_std_deg, size=len(issues))
    angles = np.clip(angles, -max_direction_deg, max_direction_deg)
    return pd.DataFrame(
        {
            "data_available_kst_dtm": issues,
            "speed_scale": scales,
            "direction_deg": angles,
        }
    )


def _perturb_frame(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    wind_pairs: tuple[tuple[str, str], ...],
    scalar_speed_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    missing = [
        column
        for pair in wind_pairs
        for column in pair
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"NWP wind columns missing: {sorted(set(missing))}")

    output = frame.copy()
    issue_time = pd.to_datetime(output["data_available_kst_dtm"])
    lookup = parameters.set_index("data_available_kst_dtm")
    scale = issue_time.map(lookup["speed_scale"]).to_numpy(float)
    angle = np.deg2rad(issue_time.map(lookup["direction_deg"]).to_numpy(float))
    if not np.isfinite(scale).all() or not np.isfinite(angle).all():
        raise ValueError("Some NWP rows did not receive perturbation parameters")
    cosine = np.cos(angle)
    sine = np.sin(angle)

    for u_col, v_col in wind_pairs:
        u = pd.to_numeric(output[u_col], errors="coerce").to_numpy(float)
        v = pd.to_numeric(output[v_col], errors="coerce").to_numpy(float)
        output[u_col] = scale * (cosine * u - sine * v)
        output[v_col] = scale * (sine * u + cosine * v)

    for column in scalar_speed_cols:
        if column not in output.columns:
            raise ValueError(f"NWP scalar wind column missing: {column}")
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(float)
        output[column] = scale * values
    return output


def perturb_nwp_issues(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    *,
    seed: int,
    speed_scale_std: float,
    direction_std_deg: float,
    max_speed_scale_delta: float,
    max_direction_deg: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply one shared speed scale and direction rotation to each NWP issue."""
    all_issues = pd.concat(
        [
            pd.to_datetime(ldaps["data_available_kst_dtm"]),
            pd.to_datetime(gfs["data_available_kst_dtm"]),
        ],
        ignore_index=True,
    )
    parameters = _issue_parameters(
        all_issues,
        seed,
        speed_scale_std,
        direction_std_deg,
        max_speed_scale_delta,
        max_direction_deg,
    )
    augmented_ldaps = _perturb_frame(ldaps, parameters, LDAPS_WIND_PAIRS)
    augmented_gfs = _perturb_frame(
        gfs,
        parameters,
        GFS_WIND_PAIRS,
        scalar_speed_cols=("surface_0_gust",),
    )
    return augmented_ldaps, augmented_gfs, parameters
