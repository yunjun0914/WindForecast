import numpy as np
from sklearn.metrics import make_scorer

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

GROUP_CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}


def group_nmae_ficr(actual, forecast, capacity, min_output_ratio=0.10):
    """Per-group NMAE and FICR (fraction of the max possible settlement actually captured).
    FICR pays a step-function unit price by error band (<=6%: 4.0, <=8%: 3.0, else: 0.0),
    weighted by actual generation -- so it does not move in lockstep with NMAE."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    valid = actual >= capacity * min_output_ratio
    actual = actual[valid]
    forecast = forecast[valid]

    error_rate = np.abs(forecast - actual) / capacity
    nmae = np.mean(error_rate)

    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    earned_settlement = np.sum(actual * unit_price)
    max_settlement = np.sum(actual * 4.0)
    ficr = earned_settlement / max_settlement

    return nmae, ficr


def group_score(actual, forecast, capacity):
    """Single-group version of the official score: 0.5*(1-NMAE) + 0.5*FICR."""
    nmae, ficr = group_nmae_ficr(actual, forecast, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr


def make_group_scorer(capacity):
    """sklearn scorer for RandomizedSearchCV/GridSearchCV, tuned for one KPX group's capacity."""
    return make_scorer(lambda y_true, y_pred: group_score(y_true, y_pred, capacity), greater_is_better=True)


def total_score(group_nmaes, group_ficrs):
    one_minus_nmae = 1 - np.mean(group_nmaes)
    ficr = np.mean(group_ficrs)
    return 0.5 * one_minus_nmae + 0.5 * ficr, one_minus_nmae, ficr


def metric(answer_df, pred_df):
    """Official competition scoring function: total_score, one_minus_nmae, ficr."""
    group_nmaes, group_ficrs = [], []
    for col in TARGET_COLS:
        nmae, ficr = group_nmae_ficr(answer_df[col], pred_df[col], GROUP_CAPACITY_KWH[col])
        group_nmaes.append(nmae)
        group_ficrs.append(ficr)
    return total_score(group_nmaes, group_ficrs)
