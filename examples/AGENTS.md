# Example 작업 지침

## 역할

README 사용 흐름의 실행 가능한 예시와 legacy saved-model compatibility
fixture를 제공한다.

## 수정 규칙

- Python script를 우선하고 notebook은 같은 public API를 사용하게 유지한다.
- CPU 예시를 기본으로 하며 GPU 전용 가정을 숨기지 않는다.
- `assets/models`와 ISA `assets/Models`는 loading regression fixture다.
  원본 weight를 재저장하거나 자동 upgrade하지 않는다.
- saved fixture의 `network.py`, `config.yaml`, `final.pth`, `scaler.pkl`,
  `functional_group.csv` 구성과 historical 이름을 보존한다.
- generated `_Models`, `_Graphs`, image, checkpoint를 source에 추가하지 않는다.

## 검증

- `tests/test_examples_syntax.py`
- `tests/test_example_model_compatibility.py`

## 변경 기록

- 2026-07-20: CPU training/prediction/custom-network script와 PyG-compatible
  read-only legacy fixture를 정리했다.
- 2026-07-20: 학습, 모델 변형, 저장 모델 모드, inference, 불확실성, ISA 해석,
  실험 비교·최적화, callback, custom network 및 CLI를 기능별 Python 예제로
  분리하고 `examples/README.md`에 실행 비용과 산출물 경계를 기록했다.
- 2026-07-21: `integration/`에 solvent 5종 train-save-Analyzer와
  general/solvent/ISA transfer-save-Analyzer public workflow를 추가했다.
  `run_all.py`는 각 workflow를 임시 cwd subprocess로 격리하고 선택적 artifact
  보존, 실행 시간 요약과 실패 exit status를 제공한다.
