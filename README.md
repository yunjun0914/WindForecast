# WindForecast

KPX 풍력발전량 예측 프로젝트입니다.

현재 코드는 세 가지 메인 블록만 남겼습니다.

| Block | File | Output |
|---|---|---|
| PINN | `predict_pinn_effective_grid_g1_year_bagging.py` | `results/submission_pinn_effective_grid_g1_year_bagging.csv` |
| TREE | `predict_tree_compact_v2_metric_valid_lgbm_mean.py` | `results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv` |
| PINN50:TREE50 | `blend_submission_files.py` | final blend submission |

실험 기록은 `docs/exp_logs.md`에 요약만 남겼습니다.

## Data

`data/`는 git에 포함하지 않습니다. 실행 전 아래 구조가 필요합니다.

```text
data/
├── sample_submission.csv
├── train/
│   ├── ldaps_train.csv
│   ├── gfs_train.csv
│   ├── train_labels.csv
│   ├── scada_vestas_train.csv
│   └── scada_unison_train.csv
└── test/
    ├── ldaps_test.csv
    └── gfs_test.csv
```

## Environment

권장 conda 환경 이름은 `WindForecast`입니다.

```bash
conda env create -f environment.yml
conda activate WindForecast
```

이미 환경이 있으면:

```bash
conda run -n WindForecast python --version
```

## Reproduce Current Outputs

### 1. PINN

```bash
conda run -n WindForecast python predict_pinn_effective_grid_g1_year_bagging.py
```

### 2. TREE

```bash
conda run -n WindForecast python predict_tree_compact_v2_metric_valid_lgbm_mean.py
```

### 3. PINN50:TREE50 Blend

```bash
conda run -n WindForecast python blend_submission_files.py ^
  --base results/submission_pinn_effective_grid_g1_year_bagging.csv ^
  --extra results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv ^
  --extra-weight 0.5 ^
  --out results/submission.csv
```

PowerShell 한 줄:

```powershell
conda run -n WindForecast python blend_submission_files.py --base results/submission_pinn_effective_grid_g1_year_bagging.csv --extra results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv --extra-weight 0.5 --out results/submission.csv
```

## Validation

| Purpose | Command |
|---|---|
| PINN OOF | `conda run -n WindForecast python experiments/evaluate_pinn_effective_grid_g1_year_bagging_oof.py` |
| TREE year-fold | `conda run -n WindForecast python experiments/evaluate_tree_compact_v2_multi_year_models.py --models lgbm_tuned,lgbm_regularized --train-policies metric_valid --stem tree_compact_v2_multi_year_lgbm_policy` |
| PINN/TREE blend | `conda run -n WindForecast python experiments/evaluate_pinn_tree_compact_v2_metric_valid_blend.py --tree-predictions results/tree_compact_v2_multi_year_lgbm_policy_predictions.csv --pinn-oof results/pinn_effective_grid_g1_year_bagging_oof_predictions.csv --tree-weights 0,0.25,0.4,0.5,0.6,0.75,1.0` |

## Current Reference Scores

| Model | Mean score | Mean nMAE | Mean FICR |
|---|---:|---:|---:|
| TREE only, metric-valid LGBM mean | `0.61157` | `0.13045` | `0.35358` |
| PINN/TREE validation best near tree=0.4 | `0.62481` | `0.12908` | `0.37870` |
| PINN50:TREE50 public candidate | `0.62423` | - | - |

## Next Priority

Tree hyperparameter optimization is the next main track.

Do not create a new test submission unless validation improves by roughly `+0.01` or the user explicitly asks for a diagnostic submission.
