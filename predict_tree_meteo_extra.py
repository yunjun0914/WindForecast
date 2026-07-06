import argparse

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor

import predict_tree_meteo

DEFAULT_OUTPUT = "results/submission_tree_meteo_extra.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="all_meteo", choices=["baseline", "thermo", "radiation_cloud", "all_meteo", "lead_all_meteo"])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predict_tree_meteo.MODELS.clear()
    predict_tree_meteo.MODELS.update(
        {
            "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
            "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            "xgb": XGBRegressor(random_state=42, n_jobs=-1),
            "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        }
    )
    args_list = ["--block", args.block, "--output", args.output]
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["predict_tree_meteo.py"] + args_list
        return predict_tree_meteo.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
