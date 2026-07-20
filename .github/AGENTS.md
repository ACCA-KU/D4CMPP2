# CI 작업 지침

## 역할

`.github/workflows/ci.yml`은 Linux/Windows fast matrix, PyG CPU full suite,
정적 검사, clean artifact 검증을 실행한다. `release.yml`은 GitHub Release의
tag/version 일치를 확인하고 검증한 sdist/wheel만 PyPI Trusted Publishing으로
게시한다.

## 수정 규칙

- repository permission은 `contents: read`를 유지한다.
- 지원 Python 3.10-3.12와 최소 PyTorch 2.2/PyG 2.8 경계를 보존한다.
- dependency 상한 또는 지원 OS를 바꾸면 `pyproject.toml`, `requirements.txt`,
  README, CI contract test를 함께 갱신한다.
- GPU CI는 필수가 아니며 release-time smoke 결과는 workspace TASK 기록에
  남긴다.
- publish 권한은 `release.yml`의 publish job에만 `id-token: write`로 부여한다.
  API token secret이나 account password를 workflow에 추가하지 않는다.
- PyPI publisher의 workflow 이름은 `release.yml`, environment는 `pypi`와
  정확히 일치시킨다.

## 검증

- `tests/test_t6_ci_contract.py`
- `tests/ci_static_checks.sh`
- `tests/ci_artifact_checks.sh`

## 변경 기록

- 2026-07-20: read-only matrix, minimum/latest PyG CPU job, Ruff/Pyright,
  isolated wheel quickstart를 확립했다.
- 2026-07-20: GitHub Release tag와 package version을 대조하고 build/publish를
  분리한 PyPI OIDC Trusted Publishing workflow를 추가했다.
