# Utility 작업 지침

## 역할

config resolution/validation, path, cache, checkpoint, split, reproducibility,
output, transfer, leaderboard 등 여러 public flow가 공유하는 경계를 담는다.

## 수정 규칙

- helper가 입력 config를 mutate하는지 명시하고 가능하면 isolated copy를 쓴다.
- path-like 입력을 basename으로 조용히 재해석하지 않는다.
- 자동 model folder에는 CSV 전체 경로가 아니라 file stem을 사용한다.
- atomic write는 같은 filesystem의 staging file과 `os.replace`를 사용한다.
- graph cache validator는 recipe, graph type, tensor shape/dtype/finite 값,
  edge-index/feature alignment를 확인한다.
- fallback 실행은 warning으로 알리고 원인을 숨기는 bare `except`를 쓰지 않는다.
- routine 출력은 `OutputAdapter`; warning/error/debug 역할을 구분한다.

## 주요 검증

- `tests/test_config_validation.py`
- `tests/test_path_utils.py`
- `tests/test_runtime_environment.py`
- `tests/test_t3_graph_cache_integrity.py`
- `tests/test_t5_config_resolution.py`
- `tests/test_t5_errors_logging.py`

## 변경 기록

- 2026-07-20: config provenance, deterministic policy, cache v2, atomic
  checkpoint, explicit path resolution, typed output/error 경계를 확립했다.
- 2026-07-20: cache edge-index/feature 행 수 불일치 검증을 추가했다.
- 2026-07-21: v2 cache payload 검증은 필수 identity/alignment 검사와 선택 가능한
  graph tensor 검사를 분리했다. 기본값은 전체 검사다.
