import os
from typing import List, Tuple, Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets.base import BaseDataset


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

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sequences[index]

    @staticmethod
    def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
        padded = pad_sequence(batch, batch_first=True, padding_value=0)
        return padded, lengths
