# Analyzer 작업 지침

## 역할

저장 model folder를 검증하고 general/solvent/ISA analyzer를 선택해
row-preserving inference와 해석 결과를 제공한다.

## 주요 계약

- model 선택은 folder 이름이 아니라 saved config/manager contract를 사용한다.
- 필수 core artifact는 `config.yaml`, `network.py`, `final.pth`다.
- non-identity scaler는 `scaler.pkl`, ISA는 saved `functional_group.csv`가
  필요하다.
- `predict()`의 historical mapping output은 유지한다.
- 새 코드는 duplicates와 invalid row를 보존하는 `predict_rows()` 및
  `predict_csv()`를 우선한다.
- 모든 input column 길이와 numeric finite 여부를 확인한다.

## 수정 규칙

- 학습 당시 DataManager, feature schema, scaler, fragmentation rule을 재사용한다.
- ISA fragment/atom index와 score/feature alignment를 함께 검증한다.
- legacy pickle mapping은 허용된 과거 class에만 좁게 적용한다.
- inference dummy row를 추가하지 않는다.

## 검증

- `tests/test_t4_analyzer_core.py`
- `tests/test_example_model_compatibility.py`

## 변경 기록

- 2026-07-20: callable Analyzer factory, PredictionResult, CSV, uncertainty,
  ensemble, ISA aligned analysis를 추가했다.
