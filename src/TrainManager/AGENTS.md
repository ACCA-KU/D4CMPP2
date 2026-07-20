# TrainManager 작업 지침

## 역할

epoch loop, NaN-masked loss, optimizer, 두 scheduler, best/latest checkpoint,
early stopping, callbacks를 관리한다.

## 수치 계약

- best metric과 scheduler step 순서를 바꾸지 않는다.
- learning-rate 감소 시 기존 best weight 복원 정책을 유지한다.
- early stopping 비교와 patience 경계를 테스트 없이 바꾸지 않는다.
- `LOAD_PATH`는 weight-only continue, `RESUME_PATH`는 full-state resume다.
- full resume는 optimizer/scheduler/epoch/early-stop/RNG를 복원한다.

## Callback 계약

- callback은 관찰 전용 immutable event를 받는다.
- callback object는 config/checkpoint에 저장하지 않는다.
- callback failure는 학습을 실패시키되 원래 failure를 숨기지 않는다.

## 검증

- `tests/test_t3_checkpoint_policy.py`
- `tests/test_t5_callbacks.py`
- `tests/test_failure_behavior.py`

## 변경 기록

- 2026-07-20: atomic full checkpoint, exact resume, observation callback
  event 순서를 확립했다.
