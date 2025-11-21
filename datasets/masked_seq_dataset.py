import os
from typing import List, Tuple, Optional, Dict

import pandas as pd
import torch

from ._base import BaseDataset


def _tifuknn_name_map(dataset: str) -> Tuple[str, str]:
    mapping = {
        "Dunnhumby": ("Dunnhumby_history.csv", "Dunnhumby_future.csv"),
        "Instacart": ("Instacart_history.csv", "Instacart_future.csv"),
        "TaFang": ("TaFang_history_NB.csv", "TaFang_future_NB.csv"),
        "ValuedShopper": ("VS_history_order.csv", "VS_future_order.csv"),
    }
    if dataset not in mapping:
        raise ValueError(f"Unknown dataset: {dataset}")
    return mapping[dataset]


def load_tifuknn_dfs(root_dir: str, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_file, val_file = _tifuknn_name_map(dataset)
    dtypes = {"CUSTOMER_ID": "int64", "ORDER_NUMBER": "int64", "MATERIAL_NUMBER": "int64"}
    train_path = os.path.join(root_dir, train_file)
    val_path = os.path.join(root_dir, val_file)
    df_train = pd.read_csv(train_path, dtype=dtypes)
    df_val = pd.read_csv(val_path, dtype=dtypes)
    return df_train, df_val


def build_sequences_from_df(
    df: pd.DataFrame,
    min_seq_len: int,
    max_seq_len: int,
    max_sequences: Optional[int] = None,
) -> List[List[int]]:
    if df.empty:
        return []
    df = df.sort_values(["CUSTOMER_ID", "ORDER_NUMBER"], kind="mergesort")
    order_items = (
        df.groupby(["CUSTOMER_ID", "ORDER_NUMBER"], sort=False)["MATERIAL_NUMBER"]
        .apply(list)
        .reset_index(name="items")
    )
    cust_orders = order_items.groupby("CUSTOMER_ID", sort=False)["items"].apply(list)
    sequences: List[List[int]] = []
    for orders in cust_orders.tolist():
        flat = [it for order in orders for it in order]
        if len(flat) < min_seq_len:
            continue
        if len(flat) <= max_seq_len:
            sequences.append(flat)
        else:
            for i in range(0, len(flat), max_seq_len):
                chunk = flat[i : i + max_seq_len]
                if len(chunk) >= min_seq_len:
                    sequences.append(chunk)
    if max_sequences is not None and max_sequences > 0:
        sequences = sequences[: max_sequences]
    return sequences


class MaskedSeqDataset(BaseDataset):
    def __init__(
        self,
        sequences: List[List[int]],
        max_seq_len: int,
        min_seq_len: int,
        pad_token_id: int,
        mask_token_id: int,
    ):
        self.max_seq_len = int(max_seq_len)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)
        self.samples = []
        for seq in sequences:
            if len(seq) < min_seq_len:
                continue
            s = seq[-self.max_seq_len :]
            if len(s) < 2:
                continue
            ctx = s[:-1]
            target = s[-1]
            masked = ctx + [self.mask_token_id]
            attn = [1] * len(masked)
            pad = self.max_seq_len - len(masked)
            if pad > 0:
                masked += [self.pad_token_id] * pad
                attn += [0] * pad
            pos = list(range(self.max_seq_len))
            self.samples.append((masked, attn, pos, target, len(ctx)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        masked, attn, pos, target, mask_idx = self.samples[idx]
        return {
            "masked_input": torch.tensor(masked, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "positional_ids": torch.tensor(pos, dtype=torch.long),
            "target_id": torch.tensor(target, dtype=torch.long),
            "mask_index": torch.tensor(mask_idx, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out


