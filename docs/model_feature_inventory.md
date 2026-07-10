# Model Feature Inventory

Status: archived pointer.

The previous contents described the older `PINN25 + TREE40 + TCN35 + group3 pseudo2022` best. That is no longer the current public best.

Current public best:

```text
results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv
```

Current best structure:

```text
PINN_floor = clip(PINN, 0.35 * capacity, capacity)
TCN_family = 0.30 * TCN_W24 + 0.40 * TCN_W72 + 0.30 * TCN_W168
final_raw = 0.25 * PINN_floor + 0.20 * TREE + 0.55 * TCN_family
final = clip(final_raw, 0.10 * capacity, capacity)
```

Use `docs/best_model_usage.md` as the source of truth for branch files and reproduction. Regenerate this inventory before using it for current-best feature analysis.
