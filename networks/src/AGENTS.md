# Network layer 작업 지침

## 역할

이 폴더는 PyG message passing, pooling, ISA heterogeneous relation 연산과
공통 linear/dropout layer를 구현한다.

## 불변조건

- DGL-era 수식의 aggregation 방향과 정규화 의미를 보존한다.
- `edge_index` 방향, destination aggregation, batch pooling을 임의로
  바꾸지 않는다.
- parameter key/shape는 saved weight compatibility의 일부다.
- ISA relation 이름은 `r_nd/r2r`, `r2i`, `i_nd/i2i`, `i2d`,
  `d_nd/d2d`, `d2r`을 유지한다.

## 수정 규칙

- 연산 변경은 작은 고정 graph에서 직접 계산한 parity test를 먼저 추가한다.
- empty edge, single node, batched graph, CPU backward를 함께 확인한다.
- dependency가 제공하는 유사 layer로 교체하기 전에 현재 custom 수식과
  state-dict shape가 같은지 증명한다.

## 변경 기록

- 2026-07-20: DGL layer를 PyG tensor/scatter 연산으로 전환했다.
- 2026-07-20: ISATPN single-value variance의 undefined NaN만 제거했다.
