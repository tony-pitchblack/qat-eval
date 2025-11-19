# TODO: rename file & class

from typing import Any, List

from datasets._base import BaseDataset


class DummyLSTMDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DummyLSTMDataset init not implemented")

    def __len__(self) -> int:
        raise NotImplementedError("DummyLSTMDataset __len__ not implemented")

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("DummyLSTMDataset __getitem__ not implemented")

    @staticmethod
    def collate_fn(batch: List[Any]) -> Any:
        raise NotImplementedError("DummyLSTMDataset collate_fn not implemented")
