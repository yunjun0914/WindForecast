from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class ExtraTreesScadaStack:
    raw_train_wind: np.ndarray
    calibrated_train_wind: np.ndarray
    raw_validation_wind: np.ndarray
    calibrated_validation_wind: np.ndarray
    wind_scales: np.ndarray
    diagnostics: pd.DataFrame


def scada_wind_matrix(
    panel: pd.DataFrame,
    scada_hourly: pd.DataFrame,
    turbines: tuple[str, ...] | list[str],
) -> np.ndarray:
    required_panel = ["forecast_kst_dtm"]
    required_scada = ["forecast_kst_dtm", "turbine_id", "scada_ws_cubic"]
    missing_panel = [column for column in required_panel if column not in panel]
    missing_scada = [column for column in required_scada if column not in scada_hourly]
    if missing_panel or missing_scada:
        raise ValueError(
            f"Missing SCADA alignment columns: panel={missing_panel} scada={missing_scada}"
        )

    source = scada_hourly[required_scada].copy()
    source["forecast_kst_dtm"] = pd.to_datetime(source["forecast_kst_dtm"])
    if source.duplicated(["forecast_kst_dtm", "turbine_id"]).any():
        raise ValueError("Duplicate hourly SCADA wind rows")
    wide = source.pivot(
        index="forecast_kst_dtm",
        columns="turbine_id",
        values="scada_ws_cubic",
    ).reindex(columns=list(turbines))
    aligned = wide.reindex(pd.DatetimeIndex(pd.to_datetime(panel["forecast_kst_dtm"])))
    return aligned.to_numpy(dtype=np.float32)


def cubic_target(wind: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(wind, dtype=float)
    scales = np.asarray(scales, dtype=float)
    return np.power(np.clip(values, 0.0, None) / scales, 3.0)


def inverse_cubic_target(target: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(target, dtype=float)
    scales = np.asarray(scales, dtype=float)
    return np.cbrt(np.clip(values, 0.0, None)) * scales


def _wind_scales(wind: np.ndarray) -> np.ndarray:
    values = np.asarray(wind, dtype=float)
    scales = []
    for turbine_index in range(values.shape[1]):
        valid = values[:, turbine_index]
        valid = valid[np.isfinite(valid)]
        if len(valid) < 100:
            raise ValueError(
                f"Too few wind targets for turbine {turbine_index}: {len(valid)}"
            )
        scales.append(max(float(np.quantile(valid, 0.99)), 5.0))
    return np.asarray(scales, dtype=np.float32)


def _fit_imputer(features: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(np.asarray(features, dtype=float), axis=0)
    return np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)


def _impute(features: np.ndarray, medians: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    return np.where(np.isfinite(values), values, medians).astype(np.float32)


def _new_extra_trees(
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    n_jobs: int,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_depth=None,
        min_samples_leaf=int(min_samples_leaf),
        max_features=float(max_features),
        bootstrap=False,
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )


def _fit_predict(
    features: np.ndarray,
    wind: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    n_jobs: int,
) -> np.ndarray:
    fit_indices = np.asarray(fit_indices, dtype=int)
    predict_indices = np.asarray(predict_indices, dtype=int)
    if len(fit_indices) < 200:
        raise ValueError(f"Too few Extra Trees fit rows: {len(fit_indices)}")
    scales = _wind_scales(wind[fit_indices])
    medians = _fit_imputer(features[fit_indices])
    model = _new_extra_trees(
        seed,
        n_estimators,
        min_samples_leaf,
        max_features,
        n_jobs,
    )
    model.fit(
        _impute(features[fit_indices], medians),
        cubic_target(wind[fit_indices], scales),
    )
    prediction = model.predict(_impute(features[predict_indices], medians))
    return inverse_cubic_target(prediction, scales).astype(np.float32)


def _crossfit_holdouts(
    observed_indices: np.ndarray,
    years: np.ndarray,
    issue_times: np.ndarray,
    within_year_folds: int,
) -> list[np.ndarray]:
    observed_indices = np.asarray(observed_indices, dtype=int)
    present_years = sorted(np.unique(years[observed_indices]).astype(int).tolist())
    if len(present_years) >= 2:
        return [
            observed_indices[years[observed_indices] == year]
            for year in present_years
        ]

    unique_issues = np.unique(issue_times[observed_indices])
    unique_issues.sort()
    chunks = [
        chunk
        for chunk in np.array_split(unique_issues, int(within_year_folds))
        if len(chunk)
    ]
    return [
        observed_indices[np.isin(issue_times[observed_indices], chunk)]
        for chunk in chunks
    ]


def _fit_wind_isotonic(
    raw_wind: np.ndarray,
    actual_wind: np.ndarray,
    scale: float,
) -> IsotonicRegression:
    valid = np.isfinite(raw_wind) & np.isfinite(actual_wind)
    if int(valid.sum()) < 100:
        raise ValueError(f"Too few wind Isotonic rows: {int(valid.sum())}")
    calibrator = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        out_of_bounds="clip",
    )
    calibrator.fit(
        np.power(np.clip(raw_wind[valid], 0.0, None) / scale, 3.0),
        np.power(np.clip(actual_wind[valid], 0.0, None) / scale, 3.0),
    )
    return calibrator


def _apply_wind_isotonic(
    calibrator: IsotonicRegression,
    raw_wind: np.ndarray,
    scale: float,
) -> np.ndarray:
    raw_cube = np.power(np.clip(raw_wind, 0.0, None) / scale, 3.0)
    return (np.cbrt(np.clip(calibrator.predict(raw_cube), 0.0, None)) * scale).astype(
        np.float32
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return np.nan
    if np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return np.nan
    return float(spearmanr(left[valid], right[valid]).statistic)


def _diagnostic_rows(
    scope: str,
    turbines: tuple[str, ...],
    actual: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for turbine_index, turbine in enumerate(turbines):
        target = actual[:, turbine_index]
        raw_one = raw[:, turbine_index]
        calibrated_one = calibrated[:, turbine_index]
        valid = np.isfinite(target) & np.isfinite(raw_one) & np.isfinite(calibrated_one)
        if not valid.any():
            continue
        scale = float(scales[turbine_index])
        target_cube = np.power(np.clip(target[valid], 0.0, None) / scale, 3.0)
        raw_cube = np.power(np.clip(raw_one[valid], 0.0, None) / scale, 3.0)
        calibrated_cube = np.power(
            np.clip(calibrated_one[valid], 0.0, None) / scale, 3.0
        )
        raw_unique = int(np.unique(raw_one[valid]).size)
        calibrated_unique = int(np.unique(calibrated_one[valid]).size)
        mapping = (
            pd.DataFrame(
                {"raw": raw_one[valid], "calibrated": calibrated_one[valid]}
            )
            .groupby("raw", sort=True)["calibrated"]
            .first()
            .to_numpy(float)
        )
        rows.append(
            {
                "stage": "wind_isotonic",
                "scope": scope,
                "turbine_id": turbine,
                "n_rows": int(valid.sum()),
                "wind_scale_p99": scale,
                "raw_wind_mae": float(np.mean(np.abs(raw_one[valid] - target[valid]))),
                "calibrated_wind_mae": float(
                    np.mean(np.abs(calibrated_one[valid] - target[valid]))
                ),
                "raw_cubic_mae": float(np.mean(np.abs(raw_cube - target_cube))),
                "calibrated_cubic_mae": float(
                    np.mean(np.abs(calibrated_cube - target_cube))
                ),
                "raw_spearman": _safe_spearman(raw_one[valid], target[valid]),
                "calibrated_spearman": _safe_spearman(
                    calibrated_one[valid], target[valid]
                ),
                "mapping_spearman": _safe_spearman(
                    raw_one[valid], calibrated_one[valid]
                ),
                "unique_retention": calibrated_unique / max(raw_unique, 1),
                "monotonic_inversions": int(np.sum(np.diff(mapping) < -1e-8)),
            }
        )
    return rows


def build_extra_trees_scada_stack(
    features: np.ndarray,
    wind: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    years: np.ndarray,
    issue_times: np.ndarray,
    turbines: tuple[str, ...] | list[str],
    seed: int,
    n_estimators: int = 500,
    min_samples_leaf: int = 2,
    max_features: float = 0.80,
    within_year_folds: int = 3,
    n_jobs: int = -1,
) -> ExtraTreesScadaStack:
    features = np.asarray(features, dtype=np.float32)
    wind = np.asarray(wind, dtype=np.float32)
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    years = np.asarray(years)
    issue_times = np.asarray(issue_times)
    turbine_order = tuple(turbines)
    if features.ndim != 2 or wind.ndim != 2:
        raise ValueError("Extra Trees SCADA stack expects 2-D feature and wind matrices")
    if len(features) != len(wind) or wind.shape[1] != len(turbine_order):
        raise ValueError("Extra Trees SCADA feature/target shape mismatch")
    if not 0.0 < float(max_features) <= 1.0:
        raise ValueError("max_features must be in (0, 1]")

    complete_target = np.isfinite(wind).all(axis=1)
    observed_train = train_indices[complete_target[train_indices]]
    if len(observed_train) < 500:
        raise ValueError(f"Too few complete SCADA wind rows: {len(observed_train)}")
    scales = _wind_scales(wind[observed_train])
    raw_train = np.full((len(train_indices), wind.shape[1]), np.nan, dtype=np.float32)
    train_positions = {row: position for position, row in enumerate(train_indices)}

    holdouts = _crossfit_holdouts(
        observed_train,
        years,
        issue_times,
        within_year_folds,
    )
    for fold_index, heldout in enumerate(holdouts):
        fit = observed_train[~np.isin(observed_train, heldout)]
        heldout_prediction = _fit_predict(
            features,
            wind,
            fit,
            heldout,
            seed + fold_index * 1009,
            n_estimators,
            min_samples_leaf,
            max_features,
            n_jobs,
        )
        positions = np.asarray([train_positions[row] for row in heldout], dtype=int)
        raw_train[positions] = heldout_prediction

    missing_train_positions = np.flatnonzero(~np.isfinite(raw_train).all(axis=1))
    full_predict_indices = np.concatenate(
        [train_indices[missing_train_positions], validation_indices]
    )
    full_prediction = _fit_predict(
        features,
        wind,
        observed_train,
        full_predict_indices,
        seed + 900001,
        n_estimators,
        min_samples_leaf,
        max_features,
        n_jobs,
    )
    if len(missing_train_positions):
        raw_train[missing_train_positions] = full_prediction[: len(missing_train_positions)]
    raw_validation = full_prediction[len(missing_train_positions) :]
    if not np.isfinite(raw_train).all() or not np.isfinite(raw_validation).all():
        raise ValueError("Incomplete Extra Trees SCADA predictions")

    calibrated_train = np.full_like(raw_train, np.nan)
    calibrated_validation = np.empty_like(raw_validation)
    actual_train = wind[train_indices]
    observed_train_positions = np.asarray(
        [train_positions[row] for row in observed_train], dtype=int
    )
    for turbine_index in range(wind.shape[1]):
        for heldout in holdouts:
            heldout_positions = np.asarray(
                [train_positions[row] for row in heldout], dtype=int
            )
            fit_positions = observed_train_positions[
                ~np.isin(observed_train_positions, heldout_positions)
            ]
            crossfit_calibrator = _fit_wind_isotonic(
                raw_train[fit_positions, turbine_index],
                actual_train[fit_positions, turbine_index],
                float(scales[turbine_index]),
            )
            calibrated_train[heldout_positions, turbine_index] = (
                _apply_wind_isotonic(
                    crossfit_calibrator,
                    raw_train[heldout_positions, turbine_index],
                    float(scales[turbine_index]),
                )
            )

        final_calibrator = _fit_wind_isotonic(
            raw_train[:, turbine_index],
            actual_train[:, turbine_index],
            float(scales[turbine_index]),
        )
        missing_calibrated = ~np.isfinite(calibrated_train[:, turbine_index])
        calibrated_train[missing_calibrated, turbine_index] = _apply_wind_isotonic(
            final_calibrator,
            raw_train[missing_calibrated, turbine_index],
            float(scales[turbine_index]),
        )
        calibrated_validation[:, turbine_index] = _apply_wind_isotonic(
            final_calibrator,
            raw_validation[:, turbine_index],
            float(scales[turbine_index]),
        )

    diagnostics = _diagnostic_rows(
        "outer_train_oof",
        turbine_order,
        actual_train,
        raw_train,
        calibrated_train,
        scales,
    )
    diagnostics.extend(
        _diagnostic_rows(
            "outer_validation",
            turbine_order,
            wind[validation_indices],
            raw_validation,
            calibrated_validation,
            scales,
        )
    )
    return ExtraTreesScadaStack(
        raw_train_wind=raw_train,
        calibrated_train_wind=calibrated_train,
        raw_validation_wind=raw_validation,
        calibrated_validation_wind=calibrated_validation,
        wind_scales=scales,
        diagnostics=pd.DataFrame(diagnostics),
    )


def cubic_feature_channels(wind: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return cubic_target(wind, scales).astype(np.float32)
