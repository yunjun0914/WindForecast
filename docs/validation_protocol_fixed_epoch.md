# WindForecast 검증 규칙: 공통 하이퍼파라미터·중앙값 epoch

2026-07-19부터 사용하는 기본 검증 규칙이다. Validation은 최종 공통 설정을 정하는 데 사용하고, submission은 모든 사용 가능한 연도를 다시 학습해 만든다.

## 전체 절차

1. 모든 fold가 공유할 하이퍼파라미터 후보를 validation으로 비교한다.
2. 선택된 하나의 하이퍼파라미터 세트를 모든 fold에 공통 적용한다.
3. 각 validation fold에서 best epoch를 측정한다.
4. fold별 best epoch의 중앙값을 최종 공통 epoch로 확정한다.
5. 공통 하이퍼파라미터와 중앙값 epoch로 모든 outer fold를 다시 학습해 fixed-epoch OOF를 만든다.
6. OOF 예측을 모두 이어 붙인 뒤 group별 metric을 한 번씩 계산하고 세 group을 동일 가중한다.
7. 설정이 확정되면 모든 사용 가능한 학습 연도를 합쳐 같은 하이퍼파라미터와 같은 epoch로 새 모델을 학습하고 test를 예측한다.

## Validation의 역할

- Validation은 공통 하이퍼파라미터를 선택하고 fold별 best epoch를 관측하는 데 사용한다.
- fold마다 서로 다른 하이퍼파라미터를 사용하지 않는다.
- fold마다 서로 다른 best checkpoint를 최종 OOF나 submission 모델로 사용하지 않는다.
- fold-best checkpoint 점수는 epoch 선택을 위한 진단값이며 최종 fixed-epoch OOF 점수와 구분해서 기록한다.
- 중앙값 epoch가 확정된 뒤에는 모든 fold를 해당 epoch까지 정확히 학습해 다시 평가한다.
- 같은 OOF를 보고 설정을 반복 변경했다면 탐색 결과임을 명시하고, 최종 설정을 고정한 OOF와 구분한다.

예시:

```text
fold별 best epoch = [18, 26, 31]
최종 공통 epoch   = median(...) = 26

최종 OOF:
  fold 1 -> epoch 26 checkpoint
  fold 2 -> epoch 26 checkpoint
  fold 3 -> epoch 26 checkpoint
```

## TCN1·TCN2 적용

TCN1과 TCN2는 stage별로 하이퍼파라미터와 epoch를 각각 결정하되, 같은 stage의 모든 outer fold는 하나의 공통 설정을 사용한다.

```text
weather/NWP
  -> 공통 하이퍼파라미터 + TCN1 중앙값 epoch
  -> 터빈별 SCADA 풍속 fixed-epoch OOF
  -> 공통 하이퍼파라미터 + TCN2 중앙값 epoch
  -> 발전량 fixed-epoch OOF
```

- TCN2 OOF 입력은 해당 outer validation을 학습하지 않은 fixed-epoch TCN1 fold 모델의 예측이어야 한다.
- Group 3의 2022처럼 정답이 없는 구간은 pseudo target을 만들지 않고 loss와 metric에서 mask한다.
- 정답이 없는 구간의 TCN 입력 생성 규칙은 실험 전에 고정하고 결과에 명시한다.

## Submission 학습

- OOF fold checkpoint를 평균해서 제출하지 않는다.
- TCN1은 모든 사용 가능한 SCADA 풍속 학습 연도를 합쳐 공통 하이퍼파라미터와 TCN1 중앙값 epoch로 새로 학습한다.
- TCN2는 모든 사용 가능한 공식 발전량 학습 연도를 합쳐 공통 하이퍼파라미터와 TCN2 중앙값 epoch로 새로 학습한다.
- test는 이 full-data 모델의 예측을 사용한다.
- multi-seed를 사용할 경우 seed 목록과 평균 방식을 validation 전에 고정한다.

## 과거 규칙

`docs/validation_protocol_non_nested.md`의 outer-fold best checkpoint 보존 및 fold-model test 평균 방식은 과거 실험 재현용으로만 유지한다. 새 실험의 기본 규칙으로 사용하지 않는다.
