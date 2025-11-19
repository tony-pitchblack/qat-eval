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

## Run Docker container
The container automatically starts Jupyter and MLflow servers in dedicated tmux sessions.
Make sure to pass free ports `JUPYTER_PORT` and `MLFLOW_PORT` as env vars (or put them in .env).

To start new container run:
```bash
docker run -it --rm --env-file .env --name qat-eval tonypitchblack/qat-eval:latest
```

To reuse existing container run:
```bash
docker start -ai qat-eval
```

To open repo inside the container in VS Code / Cursor IDE use `Dev Containers: Attach to Running Container` and open `/work/qat-eval` folder.

## Docker build (linux + macos)
```bash
docker buildx create --name multiarch --use || docker buildx use multiarch   # ensure buildx builder
docker login                                                                 # NOTE: use PAT for following push to proceed
docker buildx build --platform linux/amd64,linux/arm64 -t tonypitchblack/qat-eval:latest \
  --push . --progress=plain  # build & push multi-arch
```
