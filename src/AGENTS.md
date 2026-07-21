# Core pipeline 작업 지침

## 역할

`src/`는 DataManager → NetworkManager → TrainManager → PostProcessor와
Analyzer가 공유하는 실행 계층이다.

## 초기화 순서

1. config를 격리해 resolve한다.
2. DataManager와 Dataset contract를 선택한다.
3. graph feature dimension을 working config에 기록한다.
4. NetworkManager와 network를 만든다.
5. CSV, split, scaler, loader를 초기화한다.
6. TrainManager가 학습하고 PostProcessor가 결과를 저장한다.

이 순서를 바꾸면 custom manager와 saved model loading에 영향이 있으므로
구조 변경 승인이 필요하다.

## 공통 규칙

- caller config와 loaded YAML을 예기치 않게 mutate하지 않는다.
- 오류를 다른 유형으로 바꿀 때 원인을 `raise ... from exc`로 보존한다.
- 저장 실패가 부분 결과를 정상 결과처럼 보이게 하지 않는다.
- 사용자 출력은 `src/utils/output.py` 경계를 사용한다.

## 변경 기록

- 2026-07-20: config provenance, typed errors, callbacks, atomic checkpoint와
  row-preserving Analyzer 흐름을 정립했다.
- 2026-07-21: fitted target scaler는 prediction CSV 저장 여부와 분리해
  PostProcessor가 항상 atomic `scaler.pkl`로 저장하며 Analyzer 필수 artifact
  계약을 보장한다.
- 2026-07-21: public training/optimization/CLI/error 구현은 `src/api/`가 canonical
  위치이며 저장소 root 파일은 compatibility alias만 담당한다.
