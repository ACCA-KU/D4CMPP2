# D4CMPP2 기능별 예제

이 디렉터리는 현재 공개 기능을 작은 단위로 나눈 실행 예제 모음이다. 처음에는
`training/01_basic_cpu.py`와 `inference/01_prediction.py`를 실행하고, 필요한 기능만
아래 표에서 골라 보는 것을 권장한다.

모든 학습 예제는 기본적으로 CPU를 사용한다. GPU를 사용하려면 `device="cuda:0"`로
바꾸며, D4CMPP2는 사용할 수 없는 CUDA 장치를 CPU로 조용히 대체하지 않는다.
학습·최적화 예제는 실행 위치 아래에 `_Models`, `_Graphs` 또는 지정한 출력 폴더를
만들 수 있으므로 source tree가 아닌 별도 작업 디렉터리에서 실행하는 것이 좋다.

## 빠른 시작

저장소의 상위 디렉터리에서 package를 editable install한 뒤 실행한다.

```sh
python -m pip install -e .
python examples/training/01_basic_cpu.py
python examples/inference/01_prediction.py path/to/model
```

예제의 `data="test"`는 package에 포함된 작은 데이터다. 실제 연구에서는 CSV 경로와
target 이름을 자신의 데이터에 맞게 바꾼다.

## 기능 지도

| 기능 | 예제 | 설명 |
|---|---|---|
| 기본 CPU 학습 | `training/01_basic_cpu.py` | GCN 2 epoch 학습과 저장 모델 경로 |
| solvent 모델 | `training/02_solvent.py` | compound/solvent 두 분자 열 |
| multi-target | `training/03_multitarget.py` | target별 NaN mask를 유지한 공동 학습 |
| split·재현성 | `training/04_splitting_reproducibility.py` | random/scaffold/predefined와 seed |
| ISA 학습 | `training/05_isa.py` | ISAT/ISATPN/GC 및 fragmentation 규칙 |
| resume·continue·transfer | `training/06_saved_model_modes.py` | 세 저장 모델 모드의 차이 |
| 13개 모델 선택 | `training/07_model_families.py` | 지원 ID와 필요한 입력 열 |
| 일반 예측 | `inference/01_prediction.py` | legacy mapping과 row-preserving 결과 |
| CSV 예측 | `inference/02_csv_prediction.py` | invalid row 보존과 atomic CSV 출력 |
| 불확실성 | `inference/03_uncertainty_ensemble.py` | MC dropout과 ensemble |
| ISA 해석 | `inference/04_isa_interpretation.py` | fragment/atom 정렬 score·feature |
| 실험 비교 | `experiments/01_compare.py` | 여러 저장 모델의 leaderboard |
| model-aware 최적화 | `experiments/02_optimize.py` | grid/Bayesian search와 resume |
| legacy grid search | `experiments/03_legacy_grid_search.py` | 호환용 조합 탐색 API |
| callback | `extensions/01_callbacks.py` | 학습 loop를 바꾸지 않는 관찰자 |
| custom network | `custom_network.py`, `extensions/02_custom_network_training.py` | 등록·학습·재로딩 가능한 사용자 모델 |
| numeric input | `extensions/03_numeric_inputs.py` | named numeric batch key와 predefined split |
| heavy 통합 실행 | `integration/run_all.py` | solvent Analyzer와 계열별 transfer를 실제 학습·저장·재로딩으로 일괄 검증 |
| CLI | `cli/README.md` | 설치, 학습, resume/load/transfer 명령 |
| ISA segmentation | `ISA/segment.py` | 학습 전 fragmentation 규칙 확인 |

기존 notebook과 `general/assets/models`, `ISA/assets/Models`는 이전 저장 모델의
호환성 검증 자료다. 해당 model directory의 `config.yaml`, `network.py`,
`final.pth`, `scaler.pkl`, `functional_group.csv`를 수정하거나 다시 저장하지 않는다.

## 공통 데이터 규약

- 일반 모델: `compound`와 하나 이상의 numeric target 열
- `wS` 모델: `compound`, `solvent`, target 열
- 선택 열 `set`: `train`, `val`, `test`
- 선택 numeric 입력: `numeric_input_columns`에 이름을 명시
- explicit hydrogen: 필요한 molecule 열을 `explicit_h_columns`에 명시
- invalid SMILES가 있으면 row-preserving inference에서는 `invalid` 상태로 남고,
  학습에서는 data-quality 보고서에 제외 이유가 기록된다.

`split_strategy`는 `auto`, `random`, `predefined`, `scaffold`를 지원한다.
`random_seed`는 학습 난수 흐름을, `split_random_seed`는 분할만 제어한다.
`assets/tiny_numeric.csv`는 numeric input과 predefined split 예제 전용의 작은
데이터이며 실제 성능 평가용 데이터가 아니다.

## 저장 모델 모드

- `RESUME_PATH`: `latest.ckpt`의 optimizer, scheduler, epoch와 RNG까지 정확히 재개
- `LOAD_PATH`: `final.pth`만 읽고 optimizer와 epoch를 새로 시작
- `TRANSFER_PATH`: 이름과 shape가 호환되는 parameter만 새 모델로 복사

세 모드는 서로 배타적이다. 예제에서도 한 번에 하나만 선택한다.

## 실행 비용

`01_basic_cpu.py`와 기존 saved fixture 기반 inference는 빠른 smoke 용도다.
ISA 학습, scaffold split, grid search와 Bayesian optimization은 데이터 크기와
trial 수에 따라 오래 걸린다. 최적화 예제의 기본 trial/grid는 설명을 위해 작게
설정했지만 각 trial은 완전한 학습·저장 작업이다.

## Heavy integration runner

단위 테스트가 manager, 저장, Analyzer 경계를 우회하는 문제를 막기 위해 실제
public API만 사용하는 무거운 workflow는 `integration/`에 둔다. 다음 한 명령은
각 workflow를 별도 subprocess와 임시 작업 폴더에서 실행하고 성공/실패 및
소요 시간을 요약한다. 기본 산출물은 종료 시 자동 제거된다.

```sh
python examples/integration/run_all.py
```

모델, `transfer_report.json`, manifest를 직접 확인하려면 보존 경로를 지정한다.

```sh
python examples/integration/run_all.py --keep-output integration-output
```

세부 workflow와 선택 실행 방법은 `integration/README.md`를 참고한다.
개별 실행 파일은 `integration/solvent_analyzer.py`와
`integration/transfer_learning.py`다.
