from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from types import SimpleNamespace

from ._base import BaseDataset


class MNISTDataset(BaseDataset):
    def __init__(self, root_dir: str, split: str, download: bool = True):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        train = split == "train"
        transform = transforms.ToTensor()
        self.data: Dataset = datasets.MNIST(root=root_dir, train=train, transform=transform, download=download)
        self.inferred_params = SimpleNamespace(num_classes=10)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, label = self.data[index]
        return img, int(label)

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return images, labels



