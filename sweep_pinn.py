"""Optuna sweep over PINN loss hyperparameters (lambda_*).

lambda_hod is an AdamW decoupled weight_decay coefficient (applied to hod_bias's own
param group in train_manufacturer's stage-2 optimizer) rather than an explicit L2 loss
term -- this keeps its meaning independent of Adam's gradient normalization, so it
lives in the same everyday range as typical NN weight_decay values.
lambda_moy controls the repeating month-of-year bias. lambda_hour/lambda_year are the
same idea for train-only residual absorption.
"""

import argparse

import optuna

from train_pinn import GAMMA, LAMBDA, load_training_data, run_all_manufacturers

SWEEP_EPOCHS = 200  # reduced from the full 500+2000 for search speed; best params get a full retrain after


def objective(trial, corrected_weather, labels):
    lam = {
        "betz": trial.suggest_float("lambda_betz", 1e-3, 10.0, log=True),
        "bc": trial.suggest_float("lambda_bc", 1e-3, 10.0, log=True),
        "flat": trial.suggest_float("lambda_flat", 1e-3, 10.0, log=True),
        "smooth": trial.suggest_float("lambda_smooth", 1e-4, 1.0, log=True),
        "hod": trial.suggest_float("lambda_hod", 1e-5, 1.0, log=True),
        "moy": trial.suggest_float("lambda_moy", 1e-5, 1.0, log=True),
        "hour": trial.suggest_float("lambda_hour", 1e-5, 1.0, log=True),
        "year": trial.suggest_float("lambda_year", 1e-5, 1.0, log=True),
    }
    gamma = trial.suggest_float("gamma", 1e-3, 0.1, log=True)

    results = run_all_manufacturers(
        corrected_weather, labels, lam=lam, gamma=gamma,
        stage1_epochs=SWEEP_EPOCHS, stage2_epochs=SWEEP_EPOCHS, verbose=False, save=False,
    )
    with_bias = results[results["stage"] == "with_bias"]
    return with_bias["score"].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--study-name", default="pinn_lambda_sweep")
    args = parser.parse_args()

    corrected_weather, labels = load_training_data()

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    # seed the search with the current hand-picked defaults as a baseline trial
    default_params = {f"lambda_{k}": v for k, v in LAMBDA.items()}
    default_params["gamma"] = GAMMA
    study.enqueue_trial(default_params)

    study.optimize(lambda t: objective(t, corrected_weather, labels), n_trials=args.trials)

    print("\n=== Best trial ===")
    print(f"score: {study.best_value:.4f}")
    print(f"params: {study.best_params}")

    df = study.trials_dataframe().sort_values("value", ascending=False)
    df.to_csv("results/pinn_sweep_results.csv", index=False, encoding="utf-8-sig")
    print("\nTop 5 trials:")
    print(df[["number", "value"] + [c for c in df.columns if c.startswith("params_")]].head(5))

    importance = optuna.importance.get_param_importances(study)
    print("\nParameter importance:")
    for name, score in importance.items():
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
