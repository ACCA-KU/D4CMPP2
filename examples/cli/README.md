# CLI 예제

Python API와 CLI는 같은 학습 경로와 검증을 사용한다.

```sh
# 도움말과 설치 확인
python -m D4CMPP2 --help
d4cmpp2 --help

# 기본 CPU 학습
d4cmpp2 --data test --target Abs --network GCN --device cpu --epoch 2

# 두 target은 쉼표로 구분
d4cmpp2 --data test --target Abs,Emi --network GCN --device cpu --epoch 2

# 정확한 checkpoint resume
d4cmpp2 --resume path/to/model --device cpu --epoch 4

# final.pth만 불러와 optimizer와 epoch를 새로 시작
d4cmpp2 --load path/to/model --device cpu --epoch 2

# compatible parameter transfer
d4cmpp2 --transfer path/to/model --data test --target Abs \
  --network GCN --device cpu --epoch 2

# routine 출력과 progress를 숨김
d4cmpp2 --data test --target Abs --network GCN --device cpu --epoch 2 --quiet
```

Windows PowerShell에서는 여러 줄의 `\` 대신 한 줄로 입력하거나 PowerShell
backtick을 사용한다. `--cuda 0`은 호환용 deprecated alias이므로 새 명령에서는
`--device cuda:0`을 사용한다.

