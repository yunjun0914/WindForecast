import numpy as np
import pandas as pd

from utils.extra_trees_scada_stack import (
    build_extra_trees_scada_stack,
    cubic_feature_channels,
    cubic_target,
    inverse_cubic_target,
    scada_wind_matrix,
)


def test_cubic_target_round_trip() -> None:
    wind = np.array([[2.0, 5.0], [7.0, 11.0]], dtype=np.float32)
    scales = np.array([10.0, 12.0], dtype=np.float32)
    transformed = cubic_target(wind, scales)
    np.testing.assert_allclose(inverse_cubic_target(transformed, scales), wind)
    np.testing.assert_allclose(cubic_feature_channels(wind, scales), transformed)


def test_scada_wind_matrix_preserves_panel_order() -> None:
    panel = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(
                ["2024-01-01 02:00", "2024-01-01 01:00"]
            )
        }
    )
    scada = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(
                [
                    "2024-01-01 01:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 02:00",
                ]
            ),
            "turbine_id": ["T1", "T2", "T1", "T2"],
            "scada_ws_cubic": [1.0, 2.0, 3.0, 4.0],
        }
    )
    matrix = scada_wind_matrix(panel, scada, ("T1", "T2"))
    np.testing.assert_allclose(matrix, [[3.0, 4.0], [1.0, 2.0]])


def test_extra_trees_predictions_are_crossfit_and_isotonic_is_monotone() -> None:
    rng = np.random.default_rng(20260720)
    n_rows = 800
    features = rng.normal(size=(n_rows, 8)).astype(np.float32)
    years = np.repeat([2023, 2024], n_rows // 2)
    issue_times = pd.date_range("2023-01-01", periods=n_rows, freq="h").to_numpy()
    wind = np.column_stack(
        [
            7.0 + 1.2 * features[:, 0] - 0.4 * features[:, 1],
            8.0 + 0.8 * features[:, 2] + 0.3 * features[:, 3],
        ]
    ).astype(np.float32)
    wind = np.clip(wind, 0.2, None)
    train_indices = np.arange(700)
    validation_indices = np.arange(700, n_rows)

    result = build_extra_trees_scada_stack(
        features,
        wind,
        train_indices,
        validation_indices,
        years,
        issue_times,
        ("T1", "T2"),
        seed=17,
        n_estimators=8,
        within_year_folds=2,
        n_jobs=1,
    )
    assert result.raw_train_wind.shape == (700, 2)
    assert result.raw_validation_wind.shape == (100, 2)
    assert np.isfinite(result.calibrated_train_wind).all()
    validation_diagnostics = result.diagnostics.loc[
        result.diagnostics["scope"].eq("outer_validation")
    ]
    assert (validation_diagnostics["monotonic_inversions"] == 0).all()

    changed = wind.copy()
    changed[years == 2024] *= 1.8
    changed_result = build_extra_trees_scada_stack(
        features,
        changed,
        train_indices,
        validation_indices,
        years,
        issue_times,
        ("T1", "T2"),
        seed=17,
        n_estimators=8,
        within_year_folds=2,
        n_jobs=1,
    )
    heldout_positions = np.flatnonzero(years[train_indices] == 2024)
    np.testing.assert_allclose(
        result.raw_train_wind[heldout_positions],
        changed_result.raw_train_wind[heldout_positions],
    )
