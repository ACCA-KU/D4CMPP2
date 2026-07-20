# Dataset 작업 지침

## 역할

PyG graph와 numeric/target/smiles/row-index를 sample 및 batch로 묶고
network `forward(**kwargs)` 형식으로 unwrap한다.

## Batch 계약

- 모든 batch에는 `target`이 있다.
- general molecule `<name>`은 `_graphs`, `_node_feature`, `_edge_feature`,
  `_smiles` key를 사용한다.
- numeric input은 `<name>_var`를 사용한다.
- ISA는 `_r_node`, `_r2r_edge`, `_i_node`, `_i2i_edge`, `_d_node`,
  `_d2d_edge`를 제공한다.
- legacy single/solvent/ISA key alias를 유지한다.

## 수정 규칙

- `collate`, `unwrapper`, 모든 built-in network forward를 함께 검증한다.
- subset/split 후에도 original row index와 smiles 순서를 보존한다.
- device 이동은 graph, tensor, target 전체에 일관되게 적용한다.

## 검증

- `tests/test_pyg_data_contract.py`
- `tests/test_t3_dataset_contract_matrix.py`

## 변경 기록

- 2026-07-20: generalized multi-molecule/numeric PyG batch contract와 legacy
  key adapter를 확립했다.
