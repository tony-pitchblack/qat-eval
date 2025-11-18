# Quantization Aware Training: comprehensive evaluation

## Evaluation Results

| Method   | ROCAUC | NDCG@10 | PSNR |
|----------|--------|---------|------|
| No QAT   |        |         |      |
| LSQ      |        |         |      |
| PACT     |        |         |      |
| AdaRound |        |         |      |
| APoT     |        |         |      |
| QIL      |        |         |      |

*QIL = Quantization Interval Learning: https://arxiv.org/pdf/1808.05779*

## Pull latest Docker container
Linux or MacOS (non M-series):
```bash
docker pull --platform=linux/amd64 tonypitchblack/qat-eval:latest
```

MacOS (M-series):
```bash
docker pull --platform=linux/arm64 tonypitchblack/qat-eval:latest
```

## Run Jupyter kernel & MLflow inside Docker
To use Jupyter or MLflow make sure to use `.env` file with:
```bash
JUPYTER_PORT=8888 # choose a free port
MLFLOW_PORT=5000 # choose a free port
```

First time launch (Jupyter + MLflow):
```bash
source .env
docker run -it --name qat-eval \
  --env-file .env \
  -p $JUPYTER_PORT:$JUPYTER_PORT \
  -p $MLFLOW_PORT:$MLFLOW_PORT \
  -v "$HOME/qat-eval":/workspace \
  -w /workspace \
  tonypitchblack/qat-eval:latest \
  bash -lc "
    micromamba run -n qat-eval jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token='' --port=\$JUPYTER_PORT &
    micromamba run -n qat-eval mlflow ui --host 0.0.0.0 --port \$MLFLOW_PORT &
    wait
  "
```

Reuse existing container:
```bash
docker start -ai qat-eval
```

## Docker build (linux + macos)
```bash
docker buildx create --name multiarch --use || docker buildx use multiarch   # ensure buildx builder
docker login                                                                 # NOTE: use PAT for following push to proceed
docker buildx build --platform linux/amd64,linux/arm64 -t tonypitchblack/qat-eval:latest \
  --push . --progress=plain  # build & push multi-arch
```
