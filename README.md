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
```bash
docker pull tonypitchblack/qat-eval:latest # linux
docker pull --platform=linux/arm64 tonypitchblack/qat-eval:latest # macos
```

## Run Jupyter kernel inside docker
First time launch:
```bash
docker run -it --name qat-eval \
  -p 8888:8888 \
  -v "$HOME/qat-eval":/workspace \
  -w /workspace \
  qat-eval \
  bash -lc "micromamba run -n qat-eval jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"
```

Reuse existing container:
```bash
docker start -ai qat-eval
```

## Docker build (linux + macos)
```bash
docker buildx create --name multiarch --use || docker buildx use multiarch   # ensure buildx builder
docker login                                                                 # NOTE: use PAT for following push to proceed
docker buildx build --platform linux/amd64,linux/arm64 -t tonypitchblack/qat-eval:latest --push .  # build & push multi-arch
```
