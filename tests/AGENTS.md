# Test suite 작업 지침

## 역할

public API, saved asset, graph/network 수치, training, Analyzer, CI/artifact
계약의 회귀 근거를 제공한다.

## 실행 계층

- dependency-light:
  `python -m unittest discover -s tests -p "test_*.py" -v`
- heavy PyG:
  `D4CMPP2_RUN_HEAVY=1 python -m unittest discover -s tests -p "test_*.py" -v`
- static: `bash tests/ci_static_checks.sh`
- artifact: `bash tests/ci_artifact_checks.sh`

## 수정 규칙

- heavy test는 `markers.heavy_test`를 사용하고 fast 환경에서 import 가능한
  구조를 유지한다.
- test data는 작고 repository-local이며 외부 download가 없어야 한다.
- 저장 example fixture는 read-only로 load하고 복사본에서만 mutation한다.
- 실패를 숨기기 위해 assertion을 약화하거나 test를 삭제하지 않는다.
- 새 public/default/registry/cache 계약에는 dependency-light 정적 회귀도
  가능한 범위에서 추가한다.

## Snapshot 정책

- registry와 state-dict snapshot 변경은 구현 변경 근거와 승인이 있어야 한다.
- 수치 snapshot은 platform 차이를 고려하되 의미 없는 넓은 tolerance를 쓰지 않는다.

## 변경 기록

- 2026-07-20: fast/heavy 2계층, 13-model smoke, legacy model loading,
  clean artifact quickstart를 포함하는 199-test suite로 정리했다.
