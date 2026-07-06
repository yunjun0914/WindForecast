# WindForecast

강원도 풍력발전단지(태백가덕산·태백원동)의 **KPX 그룹별 시간당 발전량**을 예측하는 프로젝트입니다. LDAPS/GFS 기상예보와 터빈 SCADA 데이터를 사용해 2025~2026년 발전량(kWh)을 예측하고, 실제 정산금 구조(NMAE + FICR)에 맞춰 평가받습니다.

## 예측 대상

| 컬럼 | 터빈 구성 | 설비용량 |
|---|---|---:|
| `kpx_group_1` | VESTAS V126 1~6호기 | 21.6 MW |
| `kpx_group_2` | VESTAS V126 7~12호기 | 21.6 MW |
| `kpx_group_3` | UNISON U136 1~5호기 | 21.0 MW |

**평가지표**: `total_score = 0.5*(1-NMAE) + 0.5*FICR`
- NMAE: 설비용량 대비 평균절대오차 (발전량이 설비용량 10% 이상인 시간대만 평가)
- FICR: 오차구간별 정산단가(≤6%: 100%, ≤8%: 75%, 초과: 0%)를 실제 발전량으로 가중평균한, 이론상 최대 정산금 대비 실제 획득 비율

## 폴더 구조

```
WindForecast/
├── data/                    # 원본 데이터 (LDAPS/GFS 예보, SCADA, 라벨, 제출양식)
│   ├── train/               #   2022-01-01 ~ 2025-01-01
│   └── test/                #   2025-01-01 ~ 2026-01-01 (예측 대상)
├── utils/
│   ├── preprocessing.py     # 격자 집계 + 파생피처 + 시간주기 인코딩
│   ├── power_curve.py       # SCADA 기반 그룹별 경험적 파워커브
│   └── metrics.py           # 실제 채점식(NMAE+FICR) 구현
├── models/
│   ├── random_forest.py     # RandomForestRegressor 학습 래퍼
│   ├── lgbm.py               # LGBMRegressor 학습 래퍼
│   ├── xgb.py                 # XGBRegressor 학습 래퍼
│   └── artifacts/            # 학습된 모델(.pkl) 저장 위치
├── train.py                  # 그룹×모델 학습 + 검증(NMAE/FICR) — 실험용
├── tune.py                   # 하이퍼파라미터 탐색 (RandomizedSearchCV)
├── evaluate_time_holdout.py  # 시간기준 홀드아웃 검증 + permutation importance
├── diagnose_bias.py          # 출력구간별 예측 편향 진단
├── calibrate_and_evaluate.py # Isotonic 후보정 검증
├── predict.py                 # ★ 최종 제출 파일 생성 스크립트
├── results/                   # 검증 결과, 제출 파일(submission.csv)
├── docs/
│   ├── 데이터 변수 구조.md      # 원본 데이터 전체 컬럼 설명
│   ├── 현재 사용 변수 목록.md    # 실제 모델에 들어가는 피처 목록
│   └── exp_logs.md              # 스크립트별 실험 기록/의사결정 로그
└── environment.yml / requirements.txt
```

## 파이프라인

```
LDAPS(16격자) ─┐
               ├─ 격자 집계 → 파생피처(풍속) → 시간주기 인코딩 ─┐
GFS(9격자,최근접1개) ┘                                          │
                                                                ▼
SCADA(풍속-발전량) → 그룹별 경험적 파워커브 피처 ──────────→ 최종 피처 테이블(43개)
                                                                │
                                                                ▼
                                    그룹별(1/2/3) RF + LGBM + XGB 학습
                                                                │
                                                                ▼
                                              단순평균 앙상블 (그룹 내부에서만)
                                                                │
                                                                ▼
                                    3그룹 OOF 예측 풀링 → Isotonic 후보정
                                       (저출력 과대예측·고출력 과소예측 교정)
                                                                │
                                                                ▼
                                                results/submission.csv
```

## 실행 방법

```bash
# 1. conda 환경 생성 (최초 1회)
conda env create -f environment.yml

# 2. 그룹×모델 검증 (선택, 실험용)
conda run -n WindForecast python train.py

# 3. 최종 제출 파일 생성
conda run -n WindForecast python predict.py
```

## 핵심 설계 결정

- **피처는 바람 관련 변수 위주** (풍속/풍향 원본+파생, 42개) + **SCADA 기반 파워커브 피처** 1개. 기온·습도·기압 등은 permutation importance로 검증 후 제외
- **검증은 시간기준 홀드아웃**(2022~2023 학습 / 2024 검증) 사용. 랜덤 K-fold는 미래 연도 일반화를 테스트하지 못해 실제 리더보드와 방향이 안 맞았음
- **하이퍼파라미터 튜닝은 보류**: 랜덤 K-fold 기준 튜닝이 실제로는 과적합을 유발함을 확인, 디폴트 하이퍼파라미터 유지
- **Isotonic 후보정**으로 트리 앙상블의 "평균 수렴" 편향(저출력 과대·고출력 과소예측)을 교정 — FICR 개선에 직접 기여

자세한 과정과 각 결정의 근거는 [docs/exp_logs.md](docs/exp_logs.md)에 정리되어 있습니다.

## 결과

| 제출 | total_score | 1-NMAE | FICR |
|---|---:|---:|---:|
| 최초 베이스라인 | 0.6005 | 0.8660 | 0.3349 |
| 바람 피처 전용 | 0.6068 | 0.8659 | 0.3478 |
| **최종 (피처 정리 + 앙상블 + Isotonic 보정)** | **0.6087** | 0.8654 | **0.3520** |
