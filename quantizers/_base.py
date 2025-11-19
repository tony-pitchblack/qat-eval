import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseQuantizer(nn.Module, ABC):
    def __init__(self, bit_width: int, per_channel: bool = False):
        super().__init__()
        self.bit_width = bit_width
        self.per_channel = per_channel

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def export_params(self):
        ...


class BaseQuantizerWrapper(nn.Module, ABC):
    def __init__(self, quantizer: BaseQuantizer):
        super().__init__()
        self._quantizer = quantizer

    @abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantizer(x)
