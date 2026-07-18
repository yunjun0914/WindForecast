# 2026-07-18 - Global SCADA-wind to three-group TCN, non-nested fold-best OOF

## Purpose

Replace the previous per-turbine power Stage2 with one joint group model:

```text
weather/NWP
  -> TCN1
  -> 17 turbine SCADA cubic-wind predictions
  -> TCN2
  -> 3 official group-power predictions
```

No turbine-power pseudo target, turbine-power head, turbine sum, or DOY bias is used.

## User-confirmed validation protocol

- No nested OOF and no inner validation split.
- Each outer validation year selects the actual best checkpoint.
- The selected checkpoint is reloaded and used directly; no fixed-best-epoch refit.
- OOF predictions are the concatenation of the held-out predictions from the fold-best checkpoints.
- Future test inference must average the preserved fold models.

Detailed protocol: `docs/validation_protocol_non_nested.md`.

## TCN1

- Shared per-group/per-turbine IssueBlockTCN with turbine identity channels.
- Loss and checkpoint metric: `mean(abs(v_true^3-v_pred^3) / s^3)`.
- `s`: fold-train SCADA wind p99.
- Fold-best epochs and wind MAE:

| Group | Validation year | Best epoch | Wind MAE (m/s) |
|---|---:|---:|---:|
| G1 | 2022 | 5 | 1.5458 |
| G1 | 2023 | 6 | 1.3140 |
| G1 | 2024 | 4 | 1.2301 |
| G2 | 2022 | 3 | 1.4465 |
| G2 | 2023 | 5 | 1.3797 |
| G2 | 2024 | 4 | 1.3277 |
| G3 | 2023 | 9 | 1.3498 |
| G3 | 2024 | 6 | 1.2531 |

G3 2022 has no observed SCADA wind. Its five input channels are the mean of the G3 validation-2023-best and validation-2024-best TCN1 predictions; no pseudo target is created.

Pooled TCN1 wind diagnostics:

| Model | Wind MAE | Cubic MAE |
|---|---:|---:|
| calibrated optimal-grid anchor | 1.639797 | 0.076150 |
| fold-best predicted SCADA wind | 1.358230 | 0.063006 |

The preceding nested/refit TCN1 wind MAE was 1.383983, so the corrected fold-best protocol improved wind MAE by 0.025753 m/s.

## TCN2

- Input: `[issue, 24, 17]` predicted SCADA-wind channels in fixed turbine order.
- Output: `[issue, 24, 3]` capacity-normalized official group power.
- IssueBlockTCN: hidden 32, two layers, kernel 3, dropout 0.05.
- No DOY features or bias.
- Output: `0.10 + 0.90 * sigmoid(raw)`.
- Loss: masked pure-FiCR reward calculated separately per group and averaged equally.
- G3 2022 official target is fully masked; G1/G2 2022 remain in training/scoring.
- Checkpoint: mean hard FiCR across groups with an official validation target.

Fold-best epochs:

| Validation year | Best epoch | Checkpoint FiCR | Groups in checkpoint metric |
|---|---:|---:|---:|
| 2022 | 16 | 0.401924 | 2 (G1/G2) |
| 2023 | 17 | 0.397851 | 3 |
| 2024 | 35 | 0.415771 | 3 |

Because G3 has no 2022 validation target, the TCN2 validation-2022 checkpoint is excluded from the future G3 test ensemble. The saved manifest uses validation-2023/2024 checkpoints for G3.

## Standalone pooled OOF

| Variant | Score | nMAE | FiCR |
|---|---:|---:|---:|
| Previous normalized-MAE + DOY two-stage | 0.616361 | 0.134300 | 0.367022 |
| Previous pure-band no-DOY two-stage | 0.608422 | 0.151238 | 0.368083 |
| Global 17-wind -> 3-group fold-best TCN | **0.628680** | 0.141292 | 0.398653 |

Group results:

| Group | Score | nMAE | FiCR |
|---|---:|---:|---:|
| G1 | 0.632914 | 0.135325 | 0.401154 |
| G2 | 0.657418 | 0.133874 | 0.448709 |
| G3 | 0.595709 | 0.154678 | 0.346095 |

## Current-best fixed-weight replacement diagnostic

The current-best OOF structure was held fixed. Its existing TCN branch was fully replaced using:

```text
replacement = clip(current_final + 0.45 * (new_tcn - old_tcn), 0.10 * capacity, capacity)
```

| Variant | Score | nMAE | FiCR | Delta score |
|---|---:|---:|---:|---:|
| Current-best OOF | 0.638013 | 0.132610 | 0.408636 | 0 |
| Full replacement with global fold-best TCN | 0.636712 | 0.133088 | 0.406511 | -0.001302 |

This is an observed full-replacement diagnostic only. Adoption/rejection and any partial blend experiment remain the user's decision.

## Files

- Runner: `experiments/evaluate_global_scada_wind_to_group_tcn_foldbest_oof_v2.py`
- Core implementation: `experiments/evaluate_global_scada_wind_to_group_tcn_foldbest_oof_v1.py`
- OOF summary: `results/global_17wind_3group_foldbest_pure_band_tcn_oof_v2_summary.csv`
- OOF predictions: `results/global_17wind_3group_foldbest_pure_band_tcn_oof_v2_predictions.csv`
- Fold diagnostics: `results/global_17wind_3group_foldbest_pure_band_tcn_oof_v2_fold_scores.csv`
- Wind diagnostics: `results/global_17wind_3group_foldbest_pure_band_tcn_oof_v2_wind_scores.csv`
- Training diagnostics: `results/global_17wind_3group_foldbest_pure_band_tcn_oof_v2_training.csv`
- Ensemble manifest: `results/global_17wind_3group_foldbest_pure_band_tcn_oof_v2_ensemble_manifest.csv`
- Checkpoints: `results/global_17wind_3group_foldbest_pure_band_tcn_oof_v2_checkpoints/`
- Current-best replacement summary: `results/current_best_global_17wind_foldbest_tcn_replacement_oof_v2_summary.csv`

No submission CSV was created.

