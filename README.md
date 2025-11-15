# QAT Eval - Docker usage

## Prebuilt image (recommended)

Pull the latest image:

```bash
docker pull tonypitchblack/qat-eval:latest
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

## Run (interactive shell)

```bash
docker run --rm -it --gpus all -v "$(pwd)":/work qat-eval:amd64
```

## Run JupyterLab

```bash
docker run --rm -it --gpus all -p 8888:8888 -v "$(pwd)":/work qat-eval:amd64 \
  jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```

## Notes

- The image installs the conda env from `environment.yml` using micromamba and exposes port `8888` for Jupyter.
- GPU runs require the NVIDIA Container Toolkit and a host with CUDA-capable GPUs.
