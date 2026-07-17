import numpy as np
import pandas as pd
import pytest

from utils.tcn_scada_cutin_features import attach_cutin_features


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "data_available_kst_dtm": pd.to_datetime(
                ["2023-12-31 18:00", "2023-12-31 18:00"]
            ),
            "turbine_id": ["t1", "t2"],
            "base": [1.0, 2.0],
        }
    )


def test_attach_cutin_features_uses_signed_margin_and_soft_gate():
    teacher = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "turbine_id": ["t1", "t2"],
            "teacher_ws_cubic": [3.0, 5.0],
        }
    )
    out = attach_cutin_features(
        _features(), teacher, cut_in_speed=4.0, tau=1.0
    )
    assert out["scada_cutin_margin"].tolist() == [-1.0, 1.0]
    np.testing.assert_allclose(
        out["scada_cutin_gate"].to_numpy(),
        [1.0 / (1.0 + np.exp(1.0)), 1.0 / (1.0 + np.exp(-1.0))],
    )
    assert "teacher_ws_cubic" not in out


def test_attach_cutin_features_rejects_duplicate_teacher_rows():
    teacher = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "turbine_id": ["t1", "t1"],
            "teacher_ws_cubic": [3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        attach_cutin_features(_features(), teacher, cut_in_speed=4.0)
