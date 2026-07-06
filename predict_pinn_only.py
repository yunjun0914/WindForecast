import numpy as np
import pandas as pd

from predict_final_tree_ensemble import GROUP1, GROUP2, GROUP3, GROUPS, build_pinn_test_predictions
from utils.metrics import GROUP_CAPACITY_KWH

SUBMISSION_PATH = "results/submission.csv"
ARCHIVE_PATH = "results/submission_pinn_only.csv"


def main():
    pinn = build_pinn_test_predictions()
    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"])

    prediction = pd.DataFrame({"forecast_kst_dtm": pinn[GROUP1]["time"]})
    for group in GROUPS:
        if not prediction["forecast_kst_dtm"].equals(pinn[group]["time"]):
            raise ValueError(f"time mismatch for {group}")
        prediction[group] = np.clip(pinn[group]["pred"], 0, GROUP_CAPACITY_KWH[group])

    merged = submission[["forecast_id", "forecast_kst_dtm"]].merge(prediction, on="forecast_kst_dtm", how="left")
    if merged[GROUPS].isna().any().any():
        missing = merged[merged[GROUPS].isna().any(axis=1)].head()
        raise ValueError(f"submission has missing predictions:\n{missing}")
    if len(merged) != len(submission):
        raise ValueError(f"row count mismatch: {len(merged)} vs {len(submission)}")

    merged = merged[["forecast_id", "forecast_kst_dtm", GROUP1, GROUP2, GROUP3]]
    merged.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    merged.to_csv(ARCHIVE_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {SUBMISSION_PATH}: {merged.shape}")
    print(f"saved {ARCHIVE_PATH}: {merged.shape}")
    print(merged.head())
    print(merged.tail())
    return merged


if __name__ == "__main__":
    main()
