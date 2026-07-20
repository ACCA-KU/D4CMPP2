# GraphGenerator 작업 지침

## 역할

SMILES를 homogeneous PyG `Data` 또는 ISA `HeteroData`로 변환하고 선언된
feature dimension을 제공한다.

## Graph 계약

- normal graph: node feature `x`, directed `edge_index`, aligned `edge_attr`.
- ISA graph: `r_nd`, `i_nd`, `d_nd`와 여섯 relation을 유지한다.
- 모든 edge feature 행 수는 해당 `edge_index.shape[1]`과 같아야 한다.
- empty graph도 명시적인 2차원 shape와 dtype을 가진다.
- ISA bare single atom은 real-node self-loop 1개와 `(1, edge_dim)` zero
  feature를 가진다.

## 수정 규칙

- atom/bond feature 순서나 폭을 바꾸면 cache와 saved model에 영향이 있다.
- invalid SMILES는 원인을 보존해 DataManager가 row report를 만들 수 있게 한다.
- explicit hydrogen, disconnected graph, single atom, empty edge를 테스트한다.

## 검증

- `tests/test_pyg_data_contract.py`
- `tests/test_t3_dataset_contract_matrix.py`
- `tests/test_t3_graph_cache_integrity.py`

## 변경 기록

- 2026-07-20: PyG graph schema와 ISA relation topology를 고정했다.
- 2026-07-20: ISA bare single-atom edge feature의 `(2, 10)` malformed
  출력을 `(1, 8)` 선언 계약으로 수정했다.
