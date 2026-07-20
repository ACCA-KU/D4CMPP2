# DataManager 작업 지침

## 역할

CSV schema, graph 생성/cache, row alignment, split, scaler, Dataset/DataLoader
구성을 책임진다.

## 핵심 불변조건

- molecule, solvent, numeric input, target, set label, graph,
  `original_row_index`는 항상 같은 row mask를 사용한다.
- target scaler 기본 fit 범위는 train row다. legacy saved config는 기록된
  full-data 동작을 경고와 함께 보존한다.
- split seed와 80/10/10 의미를 바꾸지 않는다.
- graph error는 model folder의 `graph_error.csv`에 원본 row index와 함께
  기록한다.
- feature dimension은 NetworkManager 생성 전에 working config에 존재해야 한다.

## 수정 규칙

- CSV column과 numeric validation은 graph 생성 전에 실패시킨다.
- invalid graph를 필터링할 때 관련 배열 전체에 하나의 mask를 적용한다.
- cache recipe/schema 변경은 승인과 migration 설명이 필요하다.
- custom manager metadata는 additive이며 metadata 없는 기존 구현도 허용한다.

## 검증

- `tests/test_csv_*`
- `tests/test_graph_alignment.py`
- `tests/test_scaler_alignment.py`
- `tests/test_t3_data_quality.py`
- `tests/test_t3_scaffold_split.py`

## 변경 기록

- 2026-07-20: row alignment, data-quality report, train-only scaler,
  scaffold split, PyG cache v2를 확립했다.
