# PINN + TREE + TCN Family OOF Blend

작성일: 2026-07-09 04:09:49 +09:00

## 목적

PINN, TREE, SeqNN을 같은 leave-one-year-out OOF 기준으로 섞었을 때 실제로 상호보완이 생기는지 확인한다.

이번 실험은 검증 전용이다. test submission은 생성하지 않았다.

## 입력 Branch

| Branch | File | 역할 |
|---|---|---|
| PINN | `results/standard_oof_pinn_lgbm_time_oof_stage2_es_hod_v1.csv` | 물리/피크/FICR 보조 |
| TREE | `results/standard_oof_tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1.csv` | tabular LGBM 주력 |
| TCN W24 | `results/oof_seqnn_short_tcn_w24_v1.csv` | short ramp |
| TCN W72 | `results/oof_seqnn_mid_tcn_w72_v1.csv` | mid weather regime, best TCN |
| TCN W168 | `results/oof_seqnn_long_tcn_w168_v1.csv` | long context diversity |

## TCN Family

사용자 지정 고정 비율:

| Component | Weight |
|---|---:|
| TCN W24 | `0.30` |
| TCN W72 | `0.40` |
| TCN W168 | `0.30` |

생성 파일:

- `results/oof_tcn_family_w24_0.30_w72_0.40_w168_0.30.csv`

## 결과

PINN/TREE/TCN-family 비율을 `0.05` 간격 grid로 탐색했다.

| Variant | PINN | TREE | TCN family | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|---:|---:|---:|
| best | `0.25` | `0.40` | `0.35` | `0.63088` | `0.12760` | `0.38937` | `0.61667` |
| runner-up | `0.10` | `0.50` | `0.40` | `0.63068` | `0.12691` | `0.38826` | `0.61687` |
| TREE only | `0.00` | `1.00` | `0.00` | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| TCN family only | `0.00` | `0.00` | `1.00` | `0.62084` | `0.13797` | `0.37966` | `0.61141` |
| PINN only | `1.00` | `0.00` | `0.00` | `0.61293` | `0.14197` | `0.36783` | `0.60596` |

Best variant group/year:

| Group | Year | Score | nMAE | FICR |
|---|---:|---:|---:|---:|
| group1 | 2022 | `0.62674` | `0.12661` | `0.38010` |
| group1 | 2023 | `0.61776` | `0.12756` | `0.36308` |
| group1 | 2024 | `0.64188` | `0.11202` | `0.39578` |
| group2 | 2022 | `0.64707` | `0.12558` | `0.41973` |
| group2 | 2023 | `0.65307` | `0.12995` | `0.43610` |
| group2 | 2024 | `0.67830` | `0.11858` | `0.47518` |
| group3 | 2023 | `0.57917` | `0.15218` | `0.31053` |
| group3 | 2024 | `0.59705` | `0.12978` | `0.32388` |

## 산출물

| File | 내용 |
|---|---|
| `results/oof_pinn_tree_tcnfamily_w24_030_w72_040_w168_030_scores.csv` | group/year별 점수 |
| `results/oof_pinn_tree_tcnfamily_w24_030_w72_040_w168_030_summary.csv` | 가중치별 평균 요약 |
| `results/oof_pinn_tree_tcnfamily_w24_030_w72_040_w168_030_best_oof.csv` | 최고 3-branch blend OOF 예측 |
| `results/oof_tcn_family_w24_0.30_w72_0.40_w168_0.30.csv` | 고정 TCN family OOF 예측 |

## 판단

세 branch를 섞는 방향은 유효하다.

- TREE only 대비 `+0.00692`
- TCN W72 단일 blend best였던 `TREE 60% + TCN 40% = 0.63021`보다도 `+0.00068`
- FICR이 `0.37602 -> 0.38937`로 크게 올라간다.
- group3는 여전히 약점이다. 개선은 있지만 순위권 점프를 위해서는 group3 전용 데이터/feature/teacher 개선이 별도 필요하다.

실전 후보로는 `PINN 0.25 / TREE 0.40 / TCN-family 0.35`가 가장 높다. 다만 public 제출은 아직 보류한다. 상승폭은 의미 있지만, 제출 기회가 제한되어 있으므로 다음 단계는 best OOF 파일 기반 잔차/그룹별 플롯 확인이 좋다.

## 실행 명령

```powershell
conda run -n WindForecast python experiments\evaluate_three_branch_oof_blend.py `
  --pinn-csv results\standard_oof_pinn_lgbm_time_oof_stage2_es_hod_v1.csv `
  --tree-csv results\standard_oof_tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1.csv `
  --tcn24-csv results\oof_seqnn_short_tcn_w24_v1.csv `
  --tcn72-csv results\oof_seqnn_mid_tcn_w72_v1.csv `
  --tcn168-csv results\oof_seqnn_long_tcn_w168_v1.csv `
  --stem oof_pinn_tree_tcnfamily_w24_030_w72_040_w168_030
```
