# TODO: rename file & class

from typing import Any, List

from datasets.base import BaseDataset


class DummyESPCNDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DummyESPCNDataset init not implemented")

    def __len__(self) -> int:
        raise NotImplementedError("DummyESPCNDataset __len__ not implemented")

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("DummyESPCNDataset __getitem__ not implemented")

    @staticmethod
    def collate_fn(batch: List[Any]) -> Any:
        raise NotImplementedError("DummyESPCNDataset collate_fn not implemented")
