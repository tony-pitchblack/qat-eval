#!/usr/bin/env python3

# Basic imports
import argparse
import warnings
from typing import Any, Dict, Tuple, List, Optional
import hashlib
import json
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from itertools import product

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from models.sasrec import SASRecModel
from datasets.next_item_dataset import NextItemDataset
from models.simple_cnn import SimpleCNN
from datasets.mnist_dataset import MNISTDataset
from metrics import model_name_to_metrics
from datasets.masked_seq_dataset import MaskedSeqDataset, load_tifuknn_dfs, build_sequences_from_df, get_tifuknn_paths

from models.lstm import LSTMModel
from datasets.dummy_lstm_dataset import DummyLSTMDataset

from models.espcn import ESPCNModel
from datasets.dummy_espcn_dataset import DummyESPCNDataset

from quantizers._base import BaseQuantizerWrapper

# Uncomment models & datasets as they get implemented
model_name_to_model_dataset_class = {
    "sasrec": {
        "model_class": SASRecModel,
        "dataset_class": NextItemDataset,
    },
    "simple_cnn": {
        "model_class": SimpleCNN,
        "dataset_class": MNISTDataset,
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
from quantizers.no_quant import NoQuantizer
from quantizers.lsq import LSQQuantizer, LSQQuantizerWrapper
from quantizers.pact import PACTQuantizer
from quantizers.adaround import AdaRoundQuantizer
from quantizers.apot import APoTQuantizer
from quantizers.qil import QILQuantizer

quantizer_name_to_quantizer_class = {
    "no_quant": NoQuantizer,
    "lsq": LSQQuantizer,
    # "pact": PACTQuantizer,
    # "adaround": AdaRoundQuantizer,
    # "apot": APoTQuantizer,
    # "qil": QILQuantizer,
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


def resolve_bit_width_grid_quantizer_config_path(quantizer: str) -> str:
    return os.path.join("configs", "quantizer_configs", f"{quantizer}_bit_width_gridsearch.yml")


def _expand_config_grid(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    def _expand_node(node: Any):
        if isinstance(node, dict):
            keys = list(node.keys())
            expanded_children = [ _expand_node(node[k]) for k in keys ]
            combos = []
            for values in product(*expanded_children):
                combos.append({k: v for k, v in zip(keys, values)})
            return combos
        if isinstance(node, list):
            variants = []
            for v in node:
                variants.extend(_expand_node(v))
            return variants
        return [node]

    expanded = _expand_node(cfg)
    return tuple(expanded)


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


def _finalize_model_size(
    model: torch.nn.Module,
    quantizer_module: torch.nn.Module,
    mlflow_client=None,
) -> torch.nn.Module:
    if isinstance(quantizer_module, BaseQuantizerWrapper):
        quantized_model, size_before, size_after = quantizer_module.convert_model(model)
        if mlflow_client is not None:
            size_before = int(size_before)
            size_after = int(size_after)
            delta_pct = 0.0
            if size_before > 0:
                delta_pct = 100.0 * (size_before - size_after) / float(size_before)
            mlflow_client.log_metric("prequant_model_size", size_before)
            mlflow_client.log_metric("postquant_model_size", size_after)
            mlflow_client.log_metric("delta_model_size", float(delta_pct))
        else:
            size_before = int(size_before)
            size_after = int(size_after)
            delta_pct = 0.0
            if size_before > 0:
                delta_pct = 100.0 * (size_before - size_after) / float(size_before)
            print(
                f"Model size (bytes) before/after quantization: {size_before} -> {size_after} "
                f"(delta: {delta_pct:.2f}%)"
            )
        return quantized_model
    size_before = 0
    for param in model.parameters():
        if param is None:
            continue
        size_before += param.numel() * param.element_size()
    size_before = int(size_before)
    size_after = size_before
    delta_pct = 0.0
    if mlflow_client is not None:
        mlflow_client.log_metric("prequant_model_size", size_before)
        mlflow_client.log_metric("postquant_model_size", size_after)
        mlflow_client.log_metric("delta_model_size", float(delta_pct))
    else:
        print(
            f"Model size (bytes) before/after quantization: {size_before} -> {size_after} "
            f"(delta: {delta_pct:.2f}%)"
        )
    return model


def _log_mlflow_dataset(mlflow_client, train_path: str, dataset_name: str) -> None:
    try:
        import mlflow.data as mlfd  # type: ignore[import]
        abs_src = os.path.abspath(train_path)
        try:
            from mlflow.data.dataset_source import LocalArtifactDatasetSource  # type: ignore[import]

            src_obj = LocalArtifactDatasetSource(abs_src)
            ds_obj = mlfd.from_pandas(pd.DataFrame(), source=src_obj, name=str(dataset_name))
        except Exception:
            ds_obj = mlfd.from_pandas(pd.DataFrame(), source=abs_src, name=str(dataset_name))
        mlflow_client.log_input(ds_obj, context="training")
    except Exception:
        pass

def _sample_negatives(
    true_ids: torch.Tensor,
    max_item_id: int,
    k: int,
) -> torch.Tensor:
    m = true_ids.size(0)
    device = true_ids.device
    neg = torch.randint(low=1, high=max_item_id + 1, size=(m, k), device=device)
    true_exp = true_ids.unsqueeze(1).expand_as(neg)
    mask = neg.eq(true_exp)
    while mask.any():
        neg[mask] = torch.randint(low=1, high=max_item_id + 1, size=(mask.sum().item(),), device=device)
        mask = neg.eq(true_exp)
    return neg


def _generate_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)


def _encode_masked_sequences(
    model: SASRecModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positional_ids: torch.Tensor,
) -> torch.Tensor:
    device = input_ids.device
    x = model.item_embedding(input_ids)
    pos_emb = model.positional_embedding(positional_ids)
    x = x + pos_emb
    x = model.embedding_dropout(x)
    key_padding_mask = attention_mask == 0
    seq_len = x.size(1)
    attn_mask = _generate_subsequent_mask(seq_len, device=device)
    h = model.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
    h = model.final_norm(h)
    return h


def _evaluate_sampled_sasrec(
    model: SASRecModel,
    sequences: List[List[int]],
    max_seq_len: int,
    min_seq_len: int,
    pad_token_id: int,
    mask_token_id: int,
    max_item_id: int,
    device: torch.device,
    ks=(5, 10, 20),
    sampled_neg_k: int = 1000,
    batch_size: int = 512,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    ds = MaskedSeqDataset(
        sequences=sequences,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
    )
    if len(ds) == 0:
        return {f"HR_at_{k}": float("nan") for k in ks} | {
            f"NDCG_at_{k}": float("nan") for k in ks
        } | {f"MRR_at_{max(ks)}": float("nan")}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=MaskedSeqDataset.collate_fn)
    max_k = max(ks)
    metrics: Dict[str, float] = {f"HR_at_{k}": 0.0 for k in ks}
    metrics.update({f"NDCG_at_{k}": 0.0 for k in ks})
    mrr_total = 0.0
    n = 0
    with torch.no_grad():
        batch_idx = 0
        for batch in loader:
            inp = batch["masked_input"].to(device)
            attn = batch["attention_mask"].to(device)
            pos = batch["positional_ids"].to(device)
            targets = batch["target_id"].to(device)
            mask_index = batch["mask_index"].to(device)
            h = _encode_masked_sequences(model, inp, attn, pos)
            b, l, e = h.shape
            preds = h[torch.arange(b, device=device), mask_index]
            neg = _sample_negatives(targets, max_item_id=max_item_id, k=sampled_neg_k)
            neg = neg.to(device)
            cand = torch.cat([targets.unsqueeze(1), neg], dim=1)
            cand_emb = model.item_embedding(cand)
            scores = (cand_emb * preds.unsqueeze(1)).sum(-1)
            topk_idx = scores.topk(max_k, dim=1).indices
            hit_matrix = topk_idx == 0
            hit_any = hit_matrix.any(dim=1)
            hit_pos = torch.argmax(hit_matrix.int(), dim=1)
            mrr_total += (hit_any * (1.0 / (hit_pos + 1).float())).sum().item()
            for k in ks:
                hk = hit_matrix[:, :k].any(dim=1).float()
                metrics[f"HR_at_{k}"] += hk.sum().item()
                idx_or_neg1 = torch.where(
                    hit_matrix[:, :k],
                    torch.arange(k, device=device).unsqueeze(0).expand(b, k),
                    torch.full((b, k), -1, device=device),
                )
                pos_true = idx_or_neg1.max(dim=1).values
                mask_valid = pos_true >= 0
                ndcg = torch.zeros(b, device=device)
                ndcg[mask_valid] = 1.0 / torch.log2(pos_true[mask_valid].float() + 2.0)
                metrics[f"NDCG_at_{k}"] += ndcg.sum().item()
            n += b
            batch_idx += 1
            if max_batches is not None and batch_idx >= int(max_batches):
                break
    if n == 0:
        return {f"HR_at_{k}": float("nan") for k in ks} | {
            f"NDCG_at_{k}": float("nan") for k in ks
        } | {f"MRR_at_{max(ks)}": float("nan")}
    out: Dict[str, float] = {k: (metrics[k] / n) for k in metrics}
    out[f"MRR_at_{max(ks)}"] = mrr_total / n
    return out


def _train_one_epoch_sasrec_masked(
    model: SASRecModel,
    loader: DataLoader,
    max_item_id: int,
    neg_k: int,
    device: torch.device,
    grad_clip: Optional[float],
    quantizer_module: Optional[torch.nn.Module] = None,
) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters())
    running = 0.0
    steps = 0

    def apply_quantizer(x: torch.Tensor) -> torch.Tensor:
        if quantizer_module is None:
            return x
        return quantizer_module(x)

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        input_ids = batch["masked_input"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        positional_ids = batch["positional_ids"].to(device)
        true_ids = batch["target_id"].to(device)
        mask_index = batch["mask_index"].to(device)
        h = _encode_masked_sequences(model, input_ids, attention_mask, positional_ids)
        b, l, e = h.shape
        preds = h[torch.arange(b, device=device), mask_index]
        pos_emb = model.item_embedding(true_ids).unsqueeze(1)
        neg_ids = _sample_negatives(true_ids, max_item_id=max_item_id, k=neg_k)
        neg_ids = neg_ids.to(device)
        neg_emb = model.item_embedding(neg_ids)
        anchor = preds.unsqueeze(1)
        all_counterparty = torch.cat([pos_emb, neg_emb], dim=1)
        logits = (anchor * all_counterparty).sum(dim=-1)
        logits = apply_quantizer(logits)
        probas = torch.softmax(logits, dim=-1)
        loss = -torch.log(probas[:, 0] + 1e-12).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        running += loss.item()
        steps += 1
        if steps % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(running / steps):.4f}")
    avg = running / max(steps, 1)
    print(f"[train] steps={steps} avg_loss={avg:.4f}")
    return avg


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
    if isinstance(quantizer_module, LSQQuantizerWrapper):
        model = quantizer_module.prepare_model(model).to(device)

    def apply_quantizer(x: torch.Tensor) -> torch.Tensor:
        return quantizer_module(x)
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
            logits = apply_quantizer(logits)
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
                    logits = apply_quantizer(logits)
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
            mlflow_client.log_metric(f"train_max_{metric_name}", best_train_metric, step=epoch + 1)
            mlflow_client.log_metric(f"val_max_{metric_name}", best_val_metric, step=epoch + 1)

        header = f"{'':12}{'Train':>12}{'Val':>12}"
        loss_row = f"{'Loss':12}{train_loss_avg:12.4f}{val_loss_avg:12.4f}"
        metric_row = f"{metric_name:12}{train_metric_avg:12.4f}{val_metric_avg:12.4f}"
        print(f"\nEpoch {epoch+1}/{epochs}\n{header}\n{loss_row}\n{metric_row}")


def fit_simple_cnn(
    model: SimpleCNN,
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
    if isinstance(quantizer_module, LSQQuantizerWrapper):
        model = quantizer_module.prepare_model(model).to(device)

    def apply_quantizer(x: torch.Tensor) -> torch.Tensor:
        return quantizer_module(x)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_train_metric = float("-inf")
    best_val_metric = float("-inf")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0
        train_count = 0
        for inputs, targets in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            logits = apply_quantizer(logits)
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
                for inputs, targets in tqdm(val_loader, desc=f"Valid {epoch+1}/{epochs}"):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    logits = model(inputs)
                    logits = apply_quantizer(logits)
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
    parser.add_argument("--model", required=True, choices=["sasrec", "simple_cnn", "espcn", "lstm"])
    parser.add_argument(
        "--quantizer",
        required=True,
        choices=["no_quant", "lsq", "pact", "adaround", "apot", "qil"],
    )
    parser.add_argument("--model-config", dest="model_config", default=None)
    parser.add_argument(
        "--quantizer-config",
        dest="quantizer_config",
        default="default",
        help="Path to quantizer config YAML or one of: 'default', 'bit_width_gridsearch'",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--logging-backend",
        dest="logging_backend",
        choices=["none", "mlflow"],
        default="none",
    )
    args = parser.parse_args()

    if args.model_config:
        model_cfg_path = args.model_config
    else:
        model_cfg_path = resolve_default_model_config_path(args.model)

    if args.quantizer_config in (None, "default"):
        q_cfg_path = resolve_default_quantizer_config_path(args.quantizer)
    elif args.quantizer_config == "bit_width_gridsearch":
        q_cfg_path = resolve_bit_width_grid_quantizer_config_path(args.quantizer)
    else:
        q_cfg_path = args.quantizer_config

    cfg_model_full = load_yaml_config(model_cfg_path)
    cfg_quantizer_full = load_yaml_config(q_cfg_path)

    base_model_cfg = cfg_model_full.get("model", {})
    base_dataset_cfg = cfg_model_full.get("dataset", {})
    base_train_cfg = cfg_model_full.get("training", {})
    base_quantizer_cfg = cfg_quantizer_full.get("quantizer", {})

    combined_base_cfg: Dict[str, Any] = {
        "model": base_model_cfg,
        "dataset": base_dataset_cfg,
        "training": base_train_cfg,
        "quantizer": base_quantizer_cfg,
    }
    grid_cfgs = _expand_config_grid(combined_base_cfg)

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

    mlflow_client = None
    if args.logging_backend == "mlflow":
        mlflow_tracking_uri = get_mlflow_tracking_uri()
        import mlflow  # type: ignore[import]

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(f"QAT-eval-{args.model}")
        mlflow_client = mlflow

    if args.model == "sasrec":
        for cfg in grid_cfgs:
            model_cfg = cfg.get("model", {})
            dataset_cfg = cfg.get("dataset", {})
            train_cfg = cfg.get("training", {})
            quantizer_cfg = cfg.get("quantizer", {})

            root_dir = str(dataset_cfg.get("root_dir", os.path.join("external_repos", "TIFUKNN", "data")))
            dataset_name = str(dataset_cfg.get("dataset", "Dunnhumby"))
            min_len = int(dataset_cfg.get("min_len", 2))
            max_seq_len = int(model_cfg.get("max_seq_len", 10))
            neg_k = int(train_cfg.get("neg_k", 50))
            grad_clip = train_cfg.get("grad_clip", 1.0)

            df_train, df_val = load_tifuknn_dfs(root_dir, dataset_name)
            max_train = int(df_train["MATERIAL_NUMBER"].max()) if not df_train.empty else 0
            max_val = int(df_val["MATERIAL_NUMBER"].max()) if not df_val.empty else 0
            max_item_id = max(max_train, max_val)
            if max_item_id <= 0:
                raise ValueError("Failed to infer a positive max_item_id from data")
            mask_token_id = max_item_id + 1

            sequences_train = build_sequences_from_df(df_train, min_seq_len=min_len, max_seq_len=max_seq_len)
            sequences_val = build_sequences_from_df(df_val, min_seq_len=min_len, max_seq_len=max_seq_len)

            pad_token_id = int(model_cfg.get("pad_token_id", 0))
            model_obj = model_cls(
                num_items=mask_token_id,
                max_seq_len=max_seq_len,
                hidden_size=int(model_cfg.get("hidden_size", 64)),
                num_heads=int(model_cfg.get("num_heads", 1)),
                num_layers=int(model_cfg.get("num_layers", 1)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                pad_token_id=pad_token_id,
                device=device,
            )

            batch_size = int(train_cfg["batch_size"])
            train_ds = MaskedSeqDataset(
                sequences=sequences_train,
                max_seq_len=max_seq_len,
                min_seq_len=min_len,
                pad_token_id=pad_token_id,
                mask_token_id=mask_token_id,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=MaskedSeqDataset.collate_fn,
                drop_last=True,
            )

            quantizer_cls = quantizer_name_to_quantizer_class.get(args.quantizer)
            if quantizer_cls is None:
                raise NotImplementedError(f"Quantizer '{args.quantizer}' not implemented")
            quantizer_obj = quantizer_cls(**quantizer_cfg)
            if isinstance(quantizer_obj, LSQQuantizer):
                quantizer_obj = LSQQuantizerWrapper(quantizer_obj)

            bit_width = getattr(quantizer_obj, "bit_width", None)
            if bit_width is None and isinstance(quantizer_obj, BaseQuantizerWrapper):
                bit_width = getattr(quantizer_obj._quantizer, "bit_width", None)
            if bit_width is not None:
                print(f"Using quantization bit width: {bit_width}")
            else:
                print("No quantization is used")

            quantizer_module: torch.nn.Module = quantizer_obj
            if isinstance(quantizer_module, LSQQuantizerWrapper):
                model_obj = quantizer_module.prepare_model(model_obj).to(device)
            else:
                model_obj = model_obj.to(device)

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
                dataset_for_run = dataset_cfg.get("dataset")
                base_name = f"{args.model}-{args.quantizer}"
                if dataset_for_run:
                    base_name = f"{args.model}-{dataset_for_run}-{args.quantizer}"
                run_name = f"{base_name}-{hash_int:06d}"
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

                    if dataset_for_run:
                        train_path, _ = get_tifuknn_paths(root_dir, dataset_for_run)
                        try:
                            mlflow_client.set_tag("Dataset", str(dataset_for_run))
                        except Exception:
                            pass
                        _log_mlflow_dataset(mlflow_client, train_path, str(dataset_for_run))

                    sasrec_metrics_cfg = model_name_to_metrics.get("sasrec", [])
                    best_train_metrics: Dict[str, float] = {}
                    best_val_metrics: Dict[str, float] = {}

                    total_epochs = int(train_cfg.get("epochs", 1))
                    for ep in range(total_epochs):
                        avg_loss = _train_one_epoch_sasrec_masked(
                            model=model_obj,
                            loader=train_loader,
                            max_item_id=max_item_id,
                            neg_k=neg_k,
                            device=device,
                            grad_clip=float(grad_clip) if grad_clip is not None else None,
                            quantizer_module=quantizer_module,
                        )
                        log_payload: Dict[str, float] = {"train_loss": float(avg_loss)}

                        train_rank_metrics = _evaluate_sampled_sasrec(
                            model=model_obj,
                            sequences=sequences_train,
                            max_seq_len=max_seq_len,
                            min_seq_len=min_len,
                            pad_token_id=pad_token_id,
                            mask_token_id=mask_token_id,
                            max_item_id=max_item_id,
                            device=device,
                            ks=(5, 10, 20),
                            sampled_neg_k=1000,
                            batch_size=512,
                            max_batches=None,
                        )
                        val_rank_metrics: Dict[str, float] = {}
                        if sequences_val:
                            val_rank_metrics = _evaluate_sampled_sasrec(
                                model=model_obj,
                                sequences=sequences_val,
                                max_seq_len=max_seq_len,
                                min_seq_len=min_len,
                                pad_token_id=pad_token_id,
                                mask_token_id=mask_token_id,
                                max_item_id=max_item_id,
                                device=device,
                                ks=(5, 10, 20),
                                sampled_neg_k=1000,
                                batch_size=512,
                                max_batches=None,
                            )

                        for metric_name, _ in sasrec_metrics_cfg:
                            train_val = float(train_rank_metrics.get(metric_name, float("nan")))
                            val_val = float(val_rank_metrics.get(metric_name, float("nan"))) if sequences_val else float(
                                "nan"
                            )
                            prev_train_best = best_train_metrics.get(metric_name, float("-inf"))
                            prev_val_best = best_val_metrics.get(metric_name, float("-inf"))
                            best_train_metrics[metric_name] = max(prev_train_best, train_val)
                            best_val_metrics[metric_name] = max(prev_val_best, val_val)
                            log_payload[f"train_{metric_name}"] = train_val
                            log_payload[f"val_{metric_name}"] = val_val
                            log_payload[f"train_max_{metric_name}"] = best_train_metrics[metric_name]
                            log_payload[f"val_max_{metric_name}"] = best_val_metrics[metric_name]

                        if sequences_val:
                            metrics_str = " ".join(
                                [f"{k}={v:.4f}" for k, v in val_rank_metrics.items()]
                            )
                            print(f"[epoch {ep+1}] avg_loss={avg_loss:.4f}  VAL {metrics_str}")
                        else:
                            print(f"[epoch {ep+1}] avg_loss={avg_loss:.4f}  (no val set)")

                        mlflow_client.log_metrics(log_payload, step=ep + 1)

                    model_obj = _finalize_model_size(model_obj, quantizer_obj, mlflow_client=mlflow_client)
            else:
                sasrec_metrics_cfg = model_name_to_metrics.get("sasrec", [])
                best_train_metrics: Dict[str, float] = {}
                best_val_metrics: Dict[str, float] = {}

                total_epochs = int(train_cfg.get("epochs", 1))
                for ep in range(total_epochs):
                    avg_loss = _train_one_epoch_sasrec_masked(
                        model=model_obj,
                        loader=train_loader,
                        max_item_id=max_item_id,
                        neg_k=neg_k,
                        device=device,
                        grad_clip=float(grad_clip) if grad_clip is not None else None,
                        quantizer_module=quantizer_module,
                    )
                    train_rank_metrics = _evaluate_sampled_sasrec(
                        model=model_obj,
                        sequences=sequences_train,
                        max_seq_len=max_seq_len,
                        min_seq_len=min_len,
                        pad_token_id=pad_token_id,
                        mask_token_id=mask_token_id,
                        max_item_id=max_item_id,
                        device=device,
                        ks=(5, 10, 20),
                        sampled_neg_k=1000,
                        batch_size=512,
                        max_batches=None,
                    )
                    val_rank_metrics: Dict[str, float] = {}
                    if sequences_val:
                        val_rank_metrics = _evaluate_sampled_sasrec(
                            model=model_obj,
                            sequences=sequences_val,
                            max_seq_len=max_seq_len,
                            min_seq_len=min_len,
                            pad_token_id=pad_token_id,
                            mask_token_id=mask_token_id,
                            max_item_id=max_item_id,
                            device=device,
                            ks=(5, 10, 20),
                            sampled_neg_k=1000,
                            batch_size=512,
                            max_batches=None,
                        )

                    for metric_name, _ in sasrec_metrics_cfg:
                        train_val = float(train_rank_metrics.get(metric_name, float("nan")))
                        val_val = float(val_rank_metrics.get(metric_name, float("nan"))) if sequences_val else float(
                            "nan"
                        )
                        prev_train_best = best_train_metrics.get(metric_name, float("-inf"))
                        prev_val_best = best_val_metrics.get(metric_name, float("-inf"))
                        best_train_metrics[metric_name] = max(prev_train_best, train_val)
                        best_val_metrics[metric_name] = max(prev_val_best, val_val)

                    if sequences_val:
                        metrics_str = " ".join([f"{k}={v:.4f}" for k, v in val_rank_metrics.items()])
                        print(f"[epoch {ep+1}] avg_loss={avg_loss:.4f}  VAL {metrics_str}")
                    else:
                        print(f"[epoch {ep+1}] avg_loss={avg_loss:.4f}  (no val set)")

                model_obj = _finalize_model_size(model_obj, quantizer_obj, mlflow_client=None)
    elif args.model == "simple_cnn":
        metrics_cfg = model_name_to_metrics.get(args.model)
        if not metrics_cfg:
            raise ValueError(f"No metric mapping found for model '{args.model}'")
        metric_name, metric_fn = metrics_cfg[0]
        for cfg in grid_cfgs:
            model_cfg = cfg.get("model", {})
            dataset_cfg = cfg.get("dataset", {})
            train_cfg = cfg.get("training", {})
            quantizer_cfg = cfg.get("quantizer", {})

            train_ds = dataset_cls(**{**dataset_cfg, "split": "train"})
            val_ds = dataset_cls(**{**dataset_cfg, "split": "val"})

            model_obj = model_cls(**model_cfg)

            batch_size = int(train_cfg["batch_size"])
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=dataset_cls.collate_fn,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset_cls.collate_fn,
            )

            quantizer_cls = quantizer_name_to_quantizer_class.get(args.quantizer)
            if quantizer_cls is None:
                raise NotImplementedError(f"Quantizer '{args.quantizer}' not implemented")
            quantizer_obj = quantizer_cls(**quantizer_cfg)
            if isinstance(quantizer_obj, LSQQuantizer):
                quantizer_obj = LSQQuantizerWrapper(quantizer_obj)

            bit_width = getattr(quantizer_obj, "bit_width", None)
            if bit_width is None and isinstance(quantizer_obj, BaseQuantizerWrapper):
                bit_width = getattr(quantizer_obj._quantizer, "bit_width", None)
            if bit_width is not None:
                print(f"Using quantization bit width: {bit_width}")
            else:
                print("No quantization is used")

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
                dataset_for_run = dataset_cfg.get("dataset")
                base_name = f"{args.model}-{args.quantizer}"
                if dataset_for_run:
                    base_name = f"{args.model}-{dataset_for_run}-{args.quantizer}"
                run_name = f"{base_name}-{hash_int:06d}"
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
                    fit_simple_cnn(
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
                    model_obj = _finalize_model_size(model_obj, quantizer_obj, mlflow_client=mlflow_client)
            else:
                fit_simple_cnn(
                    model_obj,
                    quantizer_obj,
                    train_loader,
                    val_loader,
                    train_cfg,
                    metric_fn,
                    metric_name,
                    device,
                )
                model_obj = _finalize_model_size(model_obj, quantizer_obj, mlflow_client=None)


if __name__ == "__main__":
    main()


