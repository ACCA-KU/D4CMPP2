# Packaged data 작업 지침

## 역할

quickstart와 example에서 사용하는 작은 CSV 및 dataset namespace를 제공한다.

## 수정 규칙

- CSV column 이름과 의미를 임의로 바꾸지 않는다.
- `test.csv`는 offline CPU quickstart가 짧게 끝나는 크기를 유지한다.
- 개인정보, credential, 라이선스가 불명확한 데이터를 추가하지 않는다.
- dataset row 변경은 graph cache, example 출력, artifact quickstart에 영향을
  줄 수 있으므로 관련 테스트를 다시 실행한다.
- 대형 연구 dataset을 package data로 추가하지 않는다.

## 검증

- `tests/test_data_fixture.py`
- `tests/ci_artifact_checks.sh`

## 변경 기록

- 2026-07-20: wheel에 포함되는 offline quickstart data와 schema 검증을
  확립했다.
