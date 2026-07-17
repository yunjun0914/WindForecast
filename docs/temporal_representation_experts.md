# Temporal MIN/MEDIAN/MAX Expert TCNs

Updated: 2026-07-17 KST

## Purpose

Test whether deliberately different temporal views of the existing strongest
TCN input can create useful ensemble diversity without target-bin routing or
external forecast data.

## Fixed Baseline Contract

The experiment keeps the reproduced 29-hour per-turbine TCN contract fixed:

```text
input               = mixed LDAPS + GFS weather-only
feature variant     = optimal_grid_issue_context
feature count       = 75
target              = share50 turbine target
point loss          = weighted L1 with actual_sqrt group-output weight
window / model      = W72 / h128 / L3 / kernel3
effective field     = 29 hours
validation          = strict outer-year OOF
baseline pooled OOF = 0.635758
```

GEFS, source-specific models, teacher inputs, loss changes, hyperparameter
changes, postprocessing, and submissions are excluded.

## Representation Transform

The normal 75-column feature table is built first. For each turbine and NWP
issue, scalar feature values are then replaced by a centered seven-horizon
summary:

```text
MIN expert    : rolling minimum over t-3 ... t+3
MEDIAN expert : rolling median over t-3 ... t+3
MAX expert    : rolling maximum over t-3 ... t+3
```

Aggregation never crosses `data_available_kst_dtm`, so every value comes from
forecast horizons published in the same NWP issue. It does not use future
observations or later forecast issues.

Group 1/2 transform 54 of 75 columns and group 3 transforms 53. Calendar
sin/cos, wind-direction sin/cos, and raw u/v vector components remain in their
original representation because component-wise min/max is not physically
coherent.

The three statistics replace the scalar input view; they are not appended as
extra channels. Feature count therefore remains 75 for every expert.

## OOF Workload

Each representation trains 46 turbine-fold models:

```text
group1 = 6 turbines x 3 outer years = 18
group2 = 6 turbines x 3 outer years = 18
group3 = 5 turbines x 2 outer years = 10
total  = 46 fits per representation
```

MIN, MEDIAN, and MAX therefore require 138 fits. The fixed ensemble is the
unweighted mean of the three OOF predictions. No blend-weight search is run.

## Results

| Variant | Pooled Score | NMAE | FiCR |
|---|---:|---:|---:|
| Original baseline | 0.635758 | 0.129608 | 0.401124 |
| Temporal MIN | 0.633587 | 0.130740 | 0.397915 |
| Temporal MEDIAN | 0.634133 | 0.129815 | 0.398082 |
| Temporal MAX | 0.628217 | 0.131586 | 0.388021 |
| Equal MIN/MEDIAN/MAX | 0.635232 | 0.128964 | 0.399428 |

The equal representation ensemble is `-0.000526` below the original overall.
It improves average NMAE by `0.000644`, but FiCR drops by `0.001696`.

### Output Regimes

| Actual output | Original | Equal experts | Delta |
|---|---:|---:|---:|
| 0.10-0.30C | 0.668069 | 0.670014 | +0.001945 |
| 0.30-0.80C | 0.603139 | 0.607382 | +0.004243 |
| 0.80-0.90C | 0.702389 | 0.697029 | -0.005360 |
| 0.90-1.00C | 0.658100 | 0.640286 | -0.017814 |

The intended middle-output improvement is real, but shoulder and peak FiCR
losses outweigh it. MIN-to-MAX normalized residual correlation is `0.9487`, so
the views create some diversity but remain strongly related.

## Conclusion

The three temporal representations should not replace the original model as a
standalone family. They are a valid continuous middle-output component: the
ensemble improves middle Score without predicting the actual target regime or
using a hard gate. The original model must remain the peak/FiCR anchor.

No original-to-representation blend weight was evaluated in this experiment.

## Reproduction Files

```text
utils/temporal_representation.py
tests/test_temporal_representation.py
experiments/evaluate_per_turbine_tcn_interval_head_oof.py
experiments/analyze_temporal_representation_experts.py
experiments/run_temporal_representation_experts_bear.sh
```

Bear artifacts:

```text
/home/yunjun0914/windforecast_runs/temporal_representation_experts_20260717_v1/
```

Local ignored artifacts:

```text
results/temporal_representation_experts_20260717_v1/
```
