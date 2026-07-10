# Current Best Structure For Review

Updated: 2026-07-11 KST

이 문서는 현재 최고 public 구조를 빠르게 리뷰하기 위한 요약본이다. 상세 재현 명령, 하이퍼파라미터, 시간별 best timeline은 `docs/best_model_usage.md`를 기준으로 한다.

## Current Best

| Item | Value |
|---|---|
| File | `results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv` |
| Public score | `0.6386205415` |
| Public 1-nMAE | `0.8682636645` |
| Public FiCR | `0.4089774184` |
| Submit id | `1484025` |
| Submitted at | `2026-07-11 01:21:21 KST` |

## Structure

```text
TCN_family =
    0.30 * TCN_W24
  + 0.40 * TCN_W72
  + 0.30 * TCN_W168

PINN_floor = clip(PINN, 0.35 * capacity, capacity)

final_raw =
    0.25 * PINN_floor
  + 0.20 * TREE
  + 0.55 * TCN_family

final = clip(final_raw, 0.10 * capacity, capacity)
```

## Branch Files

| Component | File |
|---|---|
| PINN | `results/submission_pinn_lgbm_teacher_year_bagging.csv` |
| TREE | `results/submission_tree_lgbm_best_v2_l1.csv` |
| TCN W24 | `results/submission_seqnn_short_tcn_w24_v1.csv` |
| TCN W72 | `results/submission_seqnn_mid_tcn_w72_v1.csv` |
| TCN W168 | `results/submission_seqnn_long_tcn_w168_v1.csv` |

Important correction:

- The current best does not use `TREE g3 pseudo2022`.
- The previous `PINN25 + TREE40 + TCN35 + group3 pseudo2022` public best is now superseded.
- The current best uses the weighted-L1 TCN family, not FiCR-only TCN.

## Public Comparison

| Candidate | Public score | 1-nMAE | FiCR | Judgment |
|---|---:|---:|---:|---|
| `PINN25/TREE40/TCN35 + TREE g3 pseudo2022` | `0.6370788926` | `0.8701764551` | `0.4039813302` | Previous best |
| `TCN FiCR-only heavy blend` | `0.6361412210` | `0.8615048788` | `0.4107775632` | Higher FiCR, worse total |
| `PINN floor35 + PINN25/TREE20/TCN55 + final floor10` | `0.6386205415` | `0.8682636645` | `0.4089774184` | Current best |

## Reproduction

Use `docs/best_model_usage.md`.

Fast rebuild:

1. Floor PINN branch at 35%.
2. Blend floored PINN, plain TREE, and TCN family at `0.25 / 0.20 / 0.55`.
3. Apply final 10% floor.

The reconstruction was checked against the current best CSV with max absolute difference `3.64e-12`.
