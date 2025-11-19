#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, Tuple

ENABLE_MPS_FALLBACK = os.environ.get("ENABLE_MPS_FALLBACK")
if ENABLE_MPS_FALLBACK is not None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if ENABLE_MPS_FALLBACK.lower() in {"1", "true", "yes", "y"} else "0"

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.next_item_dataset import NextItemDataset
from models.sasrec import SASRec

# Quantizers
from quantizers.dummy import DummyQuantizer
try:
    from quantizers.lsq import LSQQuantizer
except Exception:
    LSQQuantizer = None  # optional


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


def build_quantizer(quantizer: str, q_cfg: Dict[str, Any]):
    bit_width = int(q_cfg.get("bit_width", 32))
    per_channel = bool(q_cfg.get("per_channel", False))
    if quantizer == "dummy":
        return DummyQuantizer(bit_width=bit_width, per_channel=per_channel)
    if quantizer == "lsq":
        if LSQQuantizer is None:
            raise RuntimeError("LSQQuantizer unavailable")
        return LSQQuantizer(bit_width=bit_width, per_channel=per_channel)
    if quantizer == "base":
        return DummyQuantizer(bit_width=32, per_channel=False)
    raise NotImplementedError(f"Quantizer '{quantizer}' not implemented")


def split_inputs_targets(padded: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = padded[:, :-1]
    last_index = torch.clamp(lengths - 1, min=0)
    batch_idx = torch.arange(padded.size(0), device=padded.device)
    targets = padded[batch_idx, last_index]
    return inputs, targets


def resolve_default_quantizer_config_path(quantizer: str) -> str:
    base_dir = os.path.join("configs", "quantizer_configs")
    candidates = [
        f"{quantizer}_default.yml",
        f"{quantizer}_quantizer_default.yml",
    ]
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    return os.path.join(base_dir, candidates[-1])


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def fit_sasrec(
    model: SASRec,
    quantizer_module: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("epochs", 1))

    model = model.to(device)
    quantizer_module = quantizer_module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for padded, lengths in train_loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            inputs, targets = split_inputs_targets(padded, lengths)
            logits = model(inputs, torch.clamp(lengths - 1, min=1))
            logits = quantizer_module(logits)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for padded, lengths in val_loader:
                    padded = padded.to(device)
                    lengths = lengths.to(device)
                    inputs, targets = split_inputs_targets(padded, lengths)
                    _ = model(inputs, torch.clamp(lengths - 1, min=1))
            model.train()


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
    args = parser.parse_args()

    model_cfg_path = args.model_config or resolve_default_model_config_path(args.model)
    q_cfg_path = args.quantizer_config or resolve_default_quantizer_config_path(args.quantizer)
    cfg_model_full = load_yaml_config(model_cfg_path)
    cfg_quantizer_full = load_yaml_config(q_cfg_path)

    model_cfg = cfg_model_full.get("model", {})
    dataset_cfg = cfg_model_full.get("dataset", {})
    train_cfg = cfg_model_full.get("training", {})
    quantizer_cfg = cfg_quantizer_full.get("quantizer", {})

    device = select_device()

    if args.model == "sasrec":
        root_dir = dataset_cfg.get("root_dir", os.path.join("external_repos", "TIFUKNN", "data"))
        dataset_name = dataset_cfg.get("dataset", "Dunnhumby")
        min_len = int(dataset_cfg.get("min_len", 2))

        train_ds = NextItemDataset(root_dir=root_dir, dataset=dataset_name, split="train", min_len=min_len)
        val_ds = NextItemDataset(root_dir=root_dir, dataset=dataset_name, split="val", min_len=min_len)

        batch_size = int(train_cfg.get("batch_size", 64))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=NextItemDataset.collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=NextItemDataset.collate_fn)

        if "num_items" not in model_cfg or int(model_cfg["num_items"]) <= 0:
            raise ValueError("model.num_items must be set to a positive integer in the model config")
        model_obj = SASRec(**model_cfg)
        quantizer_obj = build_quantizer(args.quantizer, quantizer_cfg)
        fit_sasrec(model_obj, quantizer_obj, train_loader, val_loader, train_cfg, device)
        return

    if args.model in {"espcn", "lstm"}:
        raise NotImplementedError(f"Training for model '{args.model}' not implemented")


if __name__ == "__main__":
    main()


