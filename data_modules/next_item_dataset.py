import os
from typing import List, Tuple, Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from data_modules._base import BaseDataset
from types import SimpleNamespace


class NextItemDataset(BaseDataset):
    def __init__(self, root_dir: Optional[str], dataset: str, split: str, min_len: int = 1):
        name_map = {
            "Dunnhumby": ("Dunnhumby_history.csv", "Dunnhumby_future.csv"),
            "Instacart": ("Instacart_history.csv", "Instacart_future.csv"),
            "TaFang": ("TaFang_history_NB.csv", "TaFang_future_NB.csv"),
            "ValuedShopper": ("VS_history_order.csv", "VS_future_order.csv"),
        }
        if dataset not in name_map:
            raise ValueError(f"Unknown dataset: {dataset}")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        filename = name_map[dataset][0 if split == "train" else 1]
        default_root = os.path.join("external_repos", "TIFUKNN", "data")
        base_root = root_dir or default_root
        path = os.path.join(base_root, filename)

        dtypes = {"CUSTOMER_ID": "int64", "ORDER_NUMBER": "int64", "MATERIAL_NUMBER": "int64"}
        df = pd.read_csv(path, dtype=dtypes)
        df = df.sort_values(["CUSTOMER_ID", "ORDER_NUMBER"])
        grouped = df.groupby("CUSTOMER_ID")["MATERIAL_NUMBER"].apply(list)
        self.sequences: List[torch.Tensor] = [
            torch.tensor(seq, dtype=torch.long) for seq in grouped.tolist() if len(seq) >= min_len
        ]

        # Infer vocabulary size (num_items) from BOTH train and val files
        train_file, val_file = name_map[dataset]
        train_path = os.path.join(base_root, train_file)
        val_path = os.path.join(base_root, val_file)
        df_train = pd.read_csv(train_path, dtype=dtypes)
        df_val = pd.read_csv(val_path, dtype=dtypes)
        # Assumes item ids are positive integers; use max id across splits as num_items
        max_train = int(df_train["MATERIAL_NUMBER"].max()) if not df_train.empty else 0
        max_val = int(df_val["MATERIAL_NUMBER"].max()) if not df_val.empty else 0
        self.num_items: int = max(max_train, max_val)
        self.inferred_params = SimpleNamespace(num_items=self.num_items)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sequences[index]

    @staticmethod
    def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
        padded = pad_sequence(batch, batch_first=True, padding_value=0)
        inputs = padded[:, :-1]
        input_lengths = torch.clamp(lengths - 1, min=1)
        batch_idx = torch.arange(padded.size(0), device=padded.device)
        targets = padded[batch_idx, lengths - 1]
        return inputs, input_lengths, targets


