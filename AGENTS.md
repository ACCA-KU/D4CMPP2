# D4CMPP2 배포 소스 작업 지침

## 적용 범위

이 파일은 `D4CMPP2/` 배포 소스 전체에 적용된다. 하위 폴더의 `AGENTS.md`가
더 구체적인 계약을 정의하면 그 지침을 함께 따른다. workspace 루트의
`AGENTS.md`와 `TASK{N}.md` 기록 규칙은 계속 유효하다.

## 현재 구현

- package version: `0.4.0`
- backend: PyTorch Geometric only
- public entry points: `train`, `grid_search`, `optimize`,
  `compare_experiments`, `Analyzer`, `Segmentator`, `Data`
- built-in network ID: 13개
- primary saved contract: `config.yaml`, `network.py`, `final.pth`,
  `scaler.pkl`; full resume는 `checkpoints/*.ckpt`
- graph cache: fingerprinted PyG schema v2

## 배포 문서 정책

- 최종 사용자 설명은 루트 `README.md`와 `CHANGELOG.md`에 둔다.
- 조사 보고서, 승인 제안서, 단계별 개발 문서는 package 내부에 만들지 않고
  workspace 루트 `TASK{N}.md`에 기록한다.
- 구현자가 알아야 할 계약과 수정 이력은 가장 가까운 `AGENTS.md`에 남긴다.
- 새 기능이 사용자 호출법이나 오류를 바꾸면 README와 해당 폴더
  `AGENTS.md`를 함께 갱신한다.

## 호환성 불변조건

- public import, config key, network ID, CSV column, result filename을
  승인 없이 삭제하거나 이름 변경하지 않는다.
- 저장 모델 loading은 model folder의 source snapshot과 자산을 우선한다.
- row alignment, split seed, scaler fit scope, checkpoint 의미의 변경은 수치
  호환성 변경으로 취급한다.
- 정상 graph의 feature 폭과 순서는 바꾸지 않는다. 잘못된 cache는 원인을
  포함해 거부하고 명시적으로 재생성한다.

## 기본 검증

```sh
python -m unittest discover -s tests -p "test_*.py" -v
bash tests/ci_static_checks.sh
bash tests/ci_artifact_checks.sh
```

ML dependency가 필요한 검증은
`D4CMPP2_RUN_HEAVY=1`로 PyG 전용 환경에서 수행한다.

## 변경 기록

- 2026-07-20: 단계 0~7 개편 완료. PyG-only 전환, config/data/cache/train,
  Analyzer, packaging/CI 계약을 확립했다.
- 2026-07-20: 개발 과정의 `docs/`를 제거하고 사용자 정보는 README로,
  유지보수 계약은 영역별 `AGENTS.md`로 재배치했다.
