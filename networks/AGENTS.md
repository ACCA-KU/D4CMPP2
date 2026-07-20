# Network 구현 지침

## 역할

이 폴더는 public network class, model metadata, registry, saved-source
snapshot의 기준 구현을 담는다.

## 주요 계약

- built-in ID는 `network_refer.yaml` 및 `tests/snapshots/network_registry.json`
  과 일치해야 한다.
- model은 `MolecularNetwork`를 상속하고 `model_name`, `input_contract`,
  `hyperparameters`, `default_optimization_space`를 선언한다.
- 기본 loss는 NaN-masked MSE이며 model-specific loss는 `compute_loss()`로
  구현한다.
- parameter 이름과 shape는 저장 `final.pth` 및 transfer compatibility의
  일부다.
- `ISATPM_model.py`는 historical saved name을 위한 deprecated adapter다.

## 수정 규칙

- forward 입력은 DataManager/Dataset contract와 함께 검토한다.
- layer 수, aggregation, activation, normalization, loss term 변경은 수치
  변경이므로 사전 승인을 받는다.
- 새 built-in model은 registry snapshot, state-dict snapshot, smoke matrix,
  README ID 목록을 함께 갱신한다.
- custom model source는 module scope에서 import 가능해야 저장 snapshot으로
  재로딩할 수 있다.

## 검증

- `tests/test_network_abc.py`
- `tests/test_state_dict_contract.py`
- `tests/test_t3_registry_smoke_matrix.py`
- model family별 `tests/test_pyg_*.py`

## 변경 기록

- 2026-07-20: 13개 PyG model과 model-owned metadata/optimization contract로
  정리했다.
- 2026-07-20: ISATPN single-score variance만 0으로 처리하고 다중 score의
  기존 unbiased variance는 보존했다.
