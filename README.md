# WindForecast

KPX 풍력발전량 예측 프로젝트입니다.

현재 최고 모델이나 제출 구조는 README에서 관리하지 않습니다. 모델 및 제출 기록이 필요한 작업은 실제 코드, Git 이력, 제출 파일과 사용자가 제공한 리더보드 기록을 직접 확인합니다.

## 데이터

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

## 실행 환경

권장 conda 환경 이름은 `WindForecast`입니다.

```bash
conda env create -f environment.yml
conda activate WindForecast
```

이미 환경이 있으면:

```bash
conda run -n WindForecast python --version
```

## 작업 규칙

- 대회 규칙은 `docs/rules.md`를 기준으로 합니다.
- 작업 방식은 `AGENTS.md`와 `docs/project_workflow.md`를 기준으로 합니다.
- 사용자가 명시적으로 요청하지 않으면 test submission이나 experiment log를 만들지 않습니다.
