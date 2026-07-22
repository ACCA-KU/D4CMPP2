# D4CMPP2 container images

Build from the repository root so the package source is available to the wheel
builder stage. Each image installs a normal D4CMPP2 wheel and runs as the
non-root `d4cmpp2` user with `/workspace` as its working directory.

## Variants

| Dockerfile | PyTorch | Actual CUDA runtime | Intended host |
|---|---:|---:|---|
| `Dockerfile.cpu` | 2.11.0 | none | CPU-only systems |
| `Dockerfile.cuda128` | 2.11.0 | 12.8 | NVIDIA driver compatible with CUDA 12.8 |
| `Dockerfile.cuda130` | 2.11.0 | 13.0 | NVIDIA driver compatible with CUDA 13.0 |

The host must have an NVIDIA driver and NVIDIA Container Toolkit. Do not install
the host driver inside these images. See NVIDIA's
[CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
and [container toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/).

## Build

Replace `DOCKERHUB_USER` with the Docker Hub namespace:

```sh
docker build -f docker/Dockerfile.cpu \
  -t DOCKERHUB_USER/d4cmpp2:1.0.0-cpu .

docker build -f docker/Dockerfile.cuda128 \
  -t DOCKERHUB_USER/d4cmpp2:1.0.0-cuda128 .

docker build -f docker/Dockerfile.cuda130 \
  -t DOCKERHUB_USER/d4cmpp2:1.0.0-cuda130 .
```

For a later D4CMPP2 release, pass the version used by the package metadata:

```sh
docker build -f docker/Dockerfile.cpu \
  --build-arg D4CMPP2_VERSION=1.1.0 \
  -t DOCKERHUB_USER/d4cmpp2:1.1.0-cpu .
```

The build fails if `D4CMPP2.__version__` differs from
`D4CMPP2_VERSION`, preventing a misleading image tag.

## Verify locally

CPU:

```sh
docker run --rm DOCKERHUB_USER/d4cmpp2:1.0.0-cpu \
  python -c "import D4CMPP2, torch; print(D4CMPP2.__version__, torch.__version__, torch.version.cuda)"
```

CUDA 12.8:

```sh
docker run --rm --gpus all DOCKERHUB_USER/d4cmpp2:1.0.0-cuda128 \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
```

CUDA 13.0:

```sh
docker run --rm --gpus all DOCKERHUB_USER/d4cmpp2:1.0.0-cuda130 \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
```

Mount a local directory to preserve model and graph outputs:

```sh
docker run --rm --gpus all \
  -v "$PWD/work:/workspace" \
  DOCKERHUB_USER/d4cmpp2:1.0.0-cuda128 \
  d4cmpp2 --data /workspace/data.csv --target target --network GCN --device cuda:0
```

## Push to Docker Hub

```sh
docker login
docker push DOCKERHUB_USER/d4cmpp2:1.0.0-cpu
docker push DOCKERHUB_USER/d4cmpp2:1.0.0-cuda128
docker push DOCKERHUB_USER/d4cmpp2:1.0.0-cuda130
```

Use explicit variant tags instead of a single ambiguous `latest` tag. Optional
moving aliases such as `latest-cpu` and `latest-cuda128` can be maintained
separately.
