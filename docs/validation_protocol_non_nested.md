# WindForecast validation protocol: non-nested fold-best OOF

> **과거 규칙:** 이 문서는 2026-07-19에 `docs/validation_protocol_fixed_epoch.md`로 대체되었다. 아래 내용은 기존 fold-best 실험 재현용이며 새 실험의 기본 검증 규칙으로 사용하지 않는다.

User-confirmed protocol, 2026-07-18 KST.

## Required behavior

- Do not use nested OOF or an inner validation split unless the user explicitly asks for it.
- Use each outer validation fold for early stopping and model selection.
- Preserve the actual best checkpoint from that fold.
- Do not discard the checkpoint and refit a fresh model on the full outer-train data for the selected number of epochs.
- Build OOF predictions by concatenating each fold-best checkpoint's held-out predictions.
- For test inference, average the predictions from the preserved fold-best models.
- Pool all outer-fold predictions first, then calculate each group metric once and weight the three groups equally.

## Two-stage TCN application

```text
weather -> TCN1 fold-best checkpoints -> turbine SCADA-wind OOF
        -> TCN2 fold-best checkpoints -> three-group power OOF
```

For group 3 in 2022, where no SCADA wind or official power target exists:

- do not create a pseudo target;
- infer the five 2022 turbine-wind channels by averaging the group-3 validation-2023 and validation-2024 TCN1 fold-best models;
- mask the group-3 2022 official power target in the TCN2 loss and OOF metric.

