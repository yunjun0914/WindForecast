import numpy as np
import pandas as pd

from utils.nwp_augmentation import _perturb_frame


def test_nwp_perturbation_rotates_and_scales_wind_vectors_by_issue():
    issue_time = pd.Timestamp("2024-01-01 00:00")
    frame = pd.DataFrame(
        {
            "data_available_kst_dtm": [issue_time],
            "u": [1.0],
            "v": [0.0],
            "unchanged": [7.0],
        }
    )
    parameters = pd.DataFrame(
        {
            "data_available_kst_dtm": [issue_time],
            "speed_scale": [2.0],
            "direction_deg": [90.0],
        }
    )

    augmented = _perturb_frame(frame, parameters, (("u", "v"),))

    assert np.isclose(augmented.loc[0, "u"], 0.0, atol=1e-12)
    assert np.isclose(augmented.loc[0, "v"], 2.0, atol=1e-12)
    assert augmented.loc[0, "unchanged"] == 7.0
