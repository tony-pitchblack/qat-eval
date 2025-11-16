# QAT Eval - Docker usage

## Prebuilt image (recommended)

Pull the latest image:

```bash
# For Linux (x86_64/amd64):
docker pull --platform=linux/amd64 tonypitchblack/qat-eval:latest

# For Mac (Apple Silicon M1/M2, or if on arm64):
docker pull --platform=linux/arm64 tonypitchblack/qat-eval:latest
```

If the arm64 pull fails with "no matching manifest", the registry image hasn't been published as multi-arch yet. Build locally for arm64 using the steps below.

```bash
# Apple Silicon fallback: build and run locally (arm64)
docker buildx build --platform linux/arm64 -t qat-eval:arm64 --load .
docker run --rm -it -v "$(pwd)":/work qat-eval:arm64
```

Run with GPU:

```bash
docker run --rm -it --gpus all -v "$(pwd)":/work tonypitchblack/qat-eval:latest
```

JupyterLab:

```bash
docker run --rm -it --gpus all -p 8888:8888 -v "$(pwd)":/work tonypitchblack/qat-eval:latest \
  jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```

## Build (linux/amd64)

Run from this directory (where `environment.yml` is located):

```bash
docker buildx build --platform linux/amd64 -t qat-eval:amd64 --load .
```

## Build (linux/arm64, Apple Silicon / ARM servers)

Run from this directory (where `environment.yml` is located):

```bash
docker buildx build --platform linux/arm64 -t qat-eval:arm64 --load .
```

## Run (interactive shell)

```bash
# amd64 image (most Linux x86_64 hosts)
docker run --rm -it -v "$(pwd)":/work qat-eval:amd64

# arm64 image (Apple Silicon Macs, ARM hosts)
docker run --rm -it -v "$(pwd)":/work qat-eval:arm64
```

## Run JupyterLab

```bash
# Replace the image tag with the one you built/pulled (qat-eval:amd64, qat-eval:arm64, or tonypitchblack/qat-eval:latest)
docker run --rm -it --gpus all -p 8888:8888 -v "$(pwd)":/work qat-eval:amd64 \
  jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```

## Publish multi-arch image (maintainers)

Build and push a manifest that supports both `linux/amd64` and `linux/arm64`:

```bash
# optional: ensure a builder is available
docker buildx create --name multiarch --use || docker buildx use multiarch

docker login
docker buildx build --platform linux/amd64,linux/arm64 \
  -t tonypitchblack/qat-eval:latest \
  --push .
```

Validate platforms:

```bash
docker buildx imagetools inspect tonypitchblack/qat-eval:latest
```

## Notes

- The image installs the conda env from `environment.yml` using micromamba and exposes port `8888` for Jupyter.
- GPU runs require the NVIDIA Container Toolkit and a host with CUDA-capable GPUs.
