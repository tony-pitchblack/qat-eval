# Quantization Aware Training: comprehensive evaluation

## Evaluation Results

| Method   | LSTM      |          | SASRec    |          | ESPCN     |          |
|----------|-----------|----------|-----------|----------|-----------|----------|
|          | Metric Δ  | Size Δ   | Metric Δ  | Size Δ   | Metric Δ  | Size Δ   |
| LSQ      |           |          |           |          |           |          |
| PACT     |           |          |           |          |           |          |
| AdaRound |           |          |           |          |           |          |
| APoT     |           |          |           |          |           |          |
| QIL      |           |          |           |          |           |          |

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

## Basic Docker container usage
To start the pulled container, run:
```bash
docker run -it --name qat-eval tonypitchblack/qat-eval:latest
```

To reuse an existing container, run:
```bash
docker start -ai qat-eval
```

To open repo inside the container in VS Code / Cursor IDE use `Dev Containers: Attach to Running Container` and open `/work/qat-eval` folder.

## Run training
From inside the container (or a local environment with dependencies installed), you can launch training with `main.py`:
```bash
python main.py \
  --model {sasrec,espcn,lstm,simple_cnn} \
  --quantizer {no_quant,lsq,pact,adaround,apot,qil} \
  [--model-config PATH] \
  [--quantizer-config PATH] \
  [--device cpu|cuda|mps] \
  [--logging-backend none|mlflow]
```

Basic usage (automatically use default configs):
```bash
python main.py --model sasrec --quantizer no_quant
```

To run with custom configs:
```bash
python main.py \
  --model sasrec \
  --quantizer no_quant \
  --device cpu \
  --model-config configs/sasrec_custom.yaml \
  --quantizer-config configs/no_quant_custom.yaml
```

To use MLflow logging see section "Advanced Docker container usage with Jupyter & MLflow" and then enable it via `--logging-backend mlflow` option:
```bash
python main.py --model sasrec --quantizer no_quant --logging-backend mlflow
```

## Advanced Docker container usage with Jupyter & MLflow
To start Jupyter and MLflow servers in dedicated tmux sessions inside the container, first set up hosts/ports in `.env` file:
```bash
JUPYTER_PORT=8888
MLFLOW_PORT=5000
JUPYTER_HOST=127.0.0.1
MLFLOW_HOST=127.0.0.1
```

To start docker container with port forwarding run:
```bash
source .env && docker run -it --rm \
  --env-file .env \
  --name qat-eval \
  -p $JUPYTER_PORT:$JUPYTER_PORT \
  -p $MLFLOW_PORT:$MLFLOW_PORT \
  tonypitchblack/qat-eval:latest
```

To start Jupyter in tmux session run:
```bash
micromamba activate qat-eval && \
tmux new -s jupyter -d \
  "jupyter notebook \
  --ip=$JUPYTER_HOST \
  --port=$JUPYTER_PORT \
  --no-browser \
  --allow-root \
  --NotebookApp.token="
```

To start MLflow in tmux session run:
```bash
micromamba activate qat-eval && \
tmux new -s mlflow -d \
  "mlflow server \
  --host ${MLFLOW_HOST} \
  --port ${MLFLOW_PORT}"
```

To reuse an existing container, run:
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
