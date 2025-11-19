from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @property
    def inferred_params(self) -> Any:
        if not hasattr(self, "_inferred_params"):
            from types import SimpleNamespace
            self._inferred_params = SimpleNamespace()
        return self._inferred_params

    @inferred_params.setter
    def inferred_params(self, value: Any) -> None:
        self._inferred_params = value

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


