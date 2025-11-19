from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        ...

    @staticmethod
    @abstractmethod
    def collate_fn(batch: List[Any]) -> Any:
        ...


