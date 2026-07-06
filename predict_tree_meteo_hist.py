import argparse

from sklearn.ensemble import HistGradientBoostingRegressor

import predict_tree_meteo


DEFAULT_OUTPUT = "results/submission_tree_meteo_hist.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="all_meteo", choices=["baseline", "thermo", "radiation_cloud", "all_meteo", "lead_all_meteo"])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predict_tree_meteo.MODELS.clear()
    predict_tree_meteo.MODELS.update(
        {
            "hist_gbr": HistGradientBoostingRegressor(
                max_iter=350,
                learning_rate=0.04,
                l2_regularization=0.02,
                random_state=42,
            )
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
