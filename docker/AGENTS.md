# Docker image instructions

## Scope

This directory owns distributable CPU and NVIDIA CUDA runtime image definitions.

## Rules

- Build a package wheel in a separate stage; do not copy the source tree into the
  final runtime image.
- Never install an NVIDIA host driver inside an image.
- Pin the PyTorch/runtime pair and label the actual CUDA runtime truthfully.
- Dockerfile names, image tags, labels, and `torch.version.cuda` must identify
  the same CUDA runtime.
- Build must verify `D4CMPP2.__version__`, PyG import, and `torch.version.cuda`.
- Runtime uses a non-root user and `/workspace` for user-owned outputs.
- Do not publish a variant until its image builds and the documented smoke command
  passes on the corresponding CPU/GPU runner.

## Change history

- 2026-07-21: Added CPU, CUDA 12.8, and CUDA 13.0 images.
