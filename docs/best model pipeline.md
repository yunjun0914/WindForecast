# Best Model Pipeline

Status: archived pointer.

This file used to describe older best-model pipelines such as `PINN50 + TREE50` and later `PINN25 + TREE40 + TCN35`. It is no longer the source of truth.

Use these documents instead:

- `docs/best_model_usage.md` for the current public best, exact reproduction commands, hyperparameters, and public timeline.
- `docs/current_pipeline_map.md` for the current branch/file flow.
- `.agents/windforecast_agent_context.md` for the short handoff summary.

Current public best as of 2026-07-11 KST:

```text
results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv
score  = 0.6386205415
1-nMAE = 0.8682636645
FiCR   = 0.4089774184
```
