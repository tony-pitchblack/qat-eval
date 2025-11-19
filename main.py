#!/usr/bin/env python3

# Basic imports
import argparse
import warnings
from typing import Any, Dict, Tuple
import hashlib
import json
import torch
import yaml
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
from functools import partial

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import models & datasets
from models.sasrec import SASRecModel
from datasets.next_item_dataset import NextItemDataset
from metrics import ndcg_at_k

from models.lstm import LSTMModel
from datasets.dummy_lstm_dataset import DummyLSTMDataset

from models.espcn import ESPCNModel
from datasets.dummy_espcn_dataset import DummyESPCNDataset

# Uncomment models & datasets as they get implemented
model_name_to_model_dataset_class = {
    "sasrec": {
        "model_class": SASRecModel,
        "dataset_class": NextItemDataset
    },
    # "lstm": {
    #     "model_class": LSTMModel,
    #     "dataset_class": DummyLSTMDataset
    # },
    # "espcn": {
    #     "model_class": ESPCNModel,
    #     "dataset_class": DummyESPCNDataset
    # },
}

# Import quantizers
from quantizers.dummy import DummyQuantizer
from quantizers.lsq import LSQQuantizer
from quantizers.pact import PACTQuantizer
from quantizers.adaround import AdaRoundQuantizer
from quantizers.apot import APoTQuantizer
from quantizers.qil import QILQuantizer

# Uncomment quantizers as they get implemented
quantizer_name_to_quantizer_class = {
    "dummy": DummyQuantizer,
    "lsq": LSQQuantizer,
    # "pact": PACTQuantizer,
    # "adaround": AdaRoundQuantizer,
    # "apot": APoTQuantizer,
    # "qil": QILQuantizer,
}

model_name_to_metric = {
    "sasrec": {
        "metric_name": f"NDCG_at_{10}",
        "metric_fn": partial(ndcg_at_k, k=10),
    },

    # "lstm": {
    #     "metric_name": "ROCAUC",
    #     "metric_fn": rocauc  # import rocauc when enabling LSTM
    # },

    # "espcn": {
    #     "metric_name": "PSNR",
    #     "metric_fn": psnr    # import psnr when enabling ESPCN
    # },
}

def load_yaml_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        content = f.read().strip()
        if not content or content.startswith("# TODO"):
            return {}
        return yaml.safe_load(content) or {}


def resolve_default_model_config_path(model: str) -> str:
    return os.path.join("configs", "model_configs", f"{model}_default.yml")


def resolve_default_quantizer_config_path(quantizer: str) -> str:
    return os.path.join("configs", "quantizer_configs", f"{quantizer}_default.yml")


def _load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def get_mlflow_tracking_uri(env_path: str = ".env") -> str:
    _load_env_file(env_path)
    host = os.environ.get("MLFLOW_HOST")
    port = os.environ.get("MLFLOW_PORT")
    if not host or not port:
        raise ValueError("MLFLOW_HOST and MLFLOW_PORT must be set in the .env file or environment")
    return f"http://{host}:{port}"


def fit_sasrec(
    model: SASRecModel,
    quantizer_module: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: Dict[str, Any],
    metric_fn,
    metric_name: str,
    device: torch.device,
    logging_backend: str = "none",
    mlflow_client=None,
) -> None:
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("epochs", 1))

    use_mlflow = logging_backend == "mlflow" and mlflow_client is not None

    model = model.to(device)
    quantizer_module = quantizer_module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_train_metric = float("-inf")
    best_val_metric = float("-inf")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0
        train_count = 0
        for inputs, input_lengths, targets in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            logits = model(inputs, input_lengths)
            logits = quantizer_module(logits)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = targets.size(0)
            train_loss_sum += loss.item() * batch_size
            train_metric_sum += metric_fn(logits.detach(), targets)
            train_count += batch_size

        val_loss_avg = 0.0
        val_metric_avg = 0.0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_metric_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for inputs, input_lengths, targets in tqdm(val_loader, desc=f"Valid {epoch+1}/{epochs}"):
                    inputs = inputs.to(device)
                    input_lengths = input_lengths.to(device)
                    targets = targets.to(device)
                    logits = model(inputs, input_lengths)
                    logits = quantizer_module(logits)
                    loss = criterion(logits, targets)
                    batch_size = targets.size(0)
                    val_loss_sum += loss.item() * batch_size
                    val_metric_sum += metric_fn(logits, targets)
                    val_count += batch_size
            val_loss_avg = (val_loss_sum / max(1, val_count)) if val_count > 0 else 0.0
            val_metric_avg = (val_metric_sum / max(1, len(val_loader))) if len(val_loader) > 0 else 0.0

        train_loss_avg = (train_loss_sum / max(1, train_count)) if train_count > 0 else 0.0
        train_metric_avg = (train_metric_sum / max(1, len(train_loader))) if len(train_loader) > 0 else 0.0
        best_train_metric = max(best_train_metric, train_metric_avg)
        best_val_metric = max(best_val_metric, val_metric_avg)
        if use_mlflow:
            mlflow_client.log_metric("train_loss", train_loss_avg, step=epoch + 1)
            mlflow_client.log_metric(f"train_{metric_name}", train_metric_avg, step=epoch + 1)
            mlflow_client.log_metric("val_loss", val_loss_avg, step=epoch + 1)
            mlflow_client.log_metric(f"val_{metric_name}", val_metric_avg, step=epoch + 1)
            mlflow_client.log_metric(f"max_train_{metric_name}", best_train_metric, step=epoch + 1)
            mlflow_client.log_metric(f"max_val_{metric_name}", best_val_metric, step=epoch + 1)

        header = f"{'':12}{'Train':>12}{'Val':>12}"
        loss_row = f"{'Loss':12}{train_loss_avg:12.4f}{val_loss_avg:12.4f}"
        metric_row = f"{metric_name:12}{train_metric_avg:12.4f}{val_metric_avg:12.4f}"
        print(f"\nEpoch {epoch+1}/{epochs}\n{header}\n{loss_row}\n{metric_row}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["sasrec", "espcn", "lstm"])
    parser.add_argument(
        "--quantizer",
        required=True,
        choices=["base", "dummy", "lsq", "pact", "adaround", "apot", "qil"],
    )
    parser.add_argument("--model-config", dest="model_config", default=None)
    parser.add_argument("--quantizer-config", dest="quantizer_config", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--logging-backend",
        dest="logging_backend",
        choices=["none", "mlflow"],
        default="none",
    )
    args = parser.parse_args()

    model_cfg_path = args.model_config or resolve_default_model_config_path(args.model)
    q_cfg_path = args.quantizer_config or resolve_default_quantizer_config_path(args.quantizer)
    cfg_model_full = load_yaml_config(model_cfg_path)
    cfg_quantizer_full = load_yaml_config(q_cfg_path)

    model_cfg = cfg_model_full.get("model", {})
    dataset_cfg = cfg_model_full.get("dataset", {})
    train_cfg = cfg_model_full.get("training", {})
    quantizer_cfg = cfg_quantizer_full.get("quantizer", {})

    if args.device is None or args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available")
        device = torch.device("cuda")
    elif args.device == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise ValueError("MPS device requested but MPS is not available")
        device = torch.device("mps")
    else:
        raise ValueError(f"Unsupported device option: {args.device}")
    print(f"Training on device: {device}")

    mapping = model_name_to_model_dataset_class.get(args.model)
    if mapping is None:
        raise NotImplementedError(f"Model '{args.model}' not implemented")
    model_cls = mapping["model_class"]
    dataset_cls = mapping["dataset_class"]

    train_ds = dataset_cls(**{**dataset_cfg, "split": "train"})
    val_ds = dataset_cls(**{**dataset_cfg, "split": "val"})

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=dataset_cls.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=dataset_cls.collate_fn)

    inferred = getattr(train_ds, "inferred_params", None)
    if inferred is None:
        raise ValueError("Dataset does not expose inferred_params needed for model construction")
    if isinstance(inferred, dict):
        inferred_dict = inferred
    else:
        inferred_dict = getattr(inferred, "__dict__", {}) or {}
    num_items = int(inferred_dict.get("num_items", 0))
    if num_items <= 0:
        raise ValueError("Failed to infer a positive num_items from inferred_params")
    inferred_dict = {**inferred_dict, "num_items": num_items}
    model_obj = model_cls(**{**model_cfg, **inferred_dict})

    quantizer_cls = quantizer_name_to_quantizer_class.get(args.quantizer)
    if quantizer_cls is None:
        raise NotImplementedError(f"Quantizer '{args.quantizer}' not implemented")
    quantizer_obj = quantizer_cls(**quantizer_cfg)

    mlflow_client = None
    if args.logging_backend == "mlflow":
        mlflow_tracking_uri = get_mlflow_tracking_uri()
        import mlflow  # type: ignore[import]

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("QAT eval")
        mlflow_client = mlflow

    if args.model == "sasrec":
        metric_cfg = model_name_to_metric.get(args.model)
        if metric_cfg is None:
            raise ValueError(f"No metric mapping found for model '{args.model}'")
        metric_name = metric_cfg["metric_name"]
        metric_fn = metric_cfg["metric_fn"]
        if mlflow_client is not None:
            run_config = {
                "model": args.model,
                "quantizer": args.quantizer,
                "model_cfg": model_cfg,
                "dataset_cfg": dataset_cfg,
                "train_cfg": train_cfg,
                "quantizer_cfg": quantizer_cfg,
            }
            config_str = json.dumps(run_config, sort_keys=True, default=str)
            hash_int = int(hashlib.md5(config_str.encode("utf-8")).hexdigest(), 16) % 1_000_000
            run_name = f"{args.model}-{args.quantizer}-{hash_int:06d}"
            with mlflow_client.start_run(run_name=run_name):
                all_params: Dict[str, Any] = {
                    "model_name": args.model,
                    "quantizer_name": args.quantizer,
                    "batch_size": int(train_cfg.get("batch_size", 0)),
                }
                for section_name, section_cfg in [
                    ("model", model_cfg),
                    ("dataset", dataset_cfg),
                    ("training", train_cfg),
                    ("quantizer", quantizer_cfg),
                ]:
                    for k, v in section_cfg.items():
                        all_params[f"{section_name}.{k}"] = v
                mlflow_client.log_params(all_params)
                fit_sasrec(
                    model_obj,
                    quantizer_obj,
                    train_loader,
                    val_loader,
                    train_cfg,
                    metric_fn,
                    metric_name,
                    device,
                    logging_backend="mlflow",
                    mlflow_client=mlflow_client,
                )
        else:
            fit_sasrec(
                model_obj,
                quantizer_obj,
                train_loader,
                val_loader,
                train_cfg,
                metric_fn,
                metric_name,
                device,
            )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()


