#!/usr/bin/env python3

# Basic imports
import argparse
import warnings
from typing import Any, Dict, Tuple
import torch
import yaml
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm

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

model_name_to_metric_fn = {
    "sasrec": lambda train_cfg: (lambda preds, targets: ndcg_at_k(preds, targets, k=int(train_cfg.get("metric_k", 10)))),
    # "lstm": lambda train_cfg: rocauc(...),  # to be defined when LSTM is enabled
    # "espcn": lambda train_cfg: psnr(...),    # to be defined when ESPCN is enabled
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


def fit_sasrec(
    model: SASRecModel,
    quantizer_module: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: Dict[str, Any],
    metric_fn,
    device: torch.device,
) -> None:
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("epochs", 1))

    model = model.to(device)
    quantizer_module = quantizer_module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

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
        print(
            f"\nEpoch {epoch+1}/{epochs}\n"
            f"  Train\n"
            f"    Loss  : {train_loss_avg:.4f}\n"
            f"    Metric: {train_metric_avg:.4f}\n"
            f"  Val\n"
            f"    Loss  : {val_loss_avg:.4f}\n"
            f"    Metric: {val_metric_avg:.4f}"
        )


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

    if args.model == "sasrec":
        if args.model not in model_name_to_metric_fn:
            raise ValueError(f"No metric mapping found for model '{args.model}'")
        metric_fn = model_name_to_metric_fn[args.model](train_cfg)
        fit_sasrec(model_obj, quantizer_obj, train_loader, val_loader, train_cfg, metric_fn, device)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()


