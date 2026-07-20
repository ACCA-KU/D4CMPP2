# NetworkManager 작업 지침

## 역할

network source를 선택·load하고 config dimension을 검증하며 optimizer,
transfer parameter group, saved weight를 연결한다.

## 주요 계약

- saved model은 model folder의 `network.py` snapshot을 current registry보다
  우선한다.
- built-in module과 custom file loading을 구분하고 원래 import 오류를 보존한다.
- transfer는 same-name/same-shape entry만 복사하고 결과를
  `transfer_report.json`에 기록한다.
- `lr_dict`는 parameter name의 dot-separated component와 정확히 매칭한다.

## 수정 규칙

- checkpoint는 parameter 이름뿐 아니라 shape를 확인한다.
- custom module directory를 `sys.path`에 영구 추가하지 않는다.
- 동일 short module name의 서로 다른 file이 충돌하지 않게 path-derived
  identity를 유지한다.

## 검증

- `tests/test_network_persistence.py`
- `tests/test_t3_transfer_learning.py`
- `tests/test_t5_import_discovery.py`

## 변경 기록

- 2026-07-20: saved snapshot 우선 loading, isolated custom imports,
  shape-aware transfer report를 확립했다.
