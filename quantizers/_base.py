import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseQuantizer(nn.Module, ABC):
    def __init__(self, bit_width: int = None, per_channel: bool = False):
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

    def measure_model_size(self, model: nn.Module, bit_width: int = 32) -> int:
        total_params = 0
        for param in model.parameters():
            if param is None:
                continue
            total_params += param.numel()
        total_bits = total_params * int(bit_width)
        return (total_bits + 7) // 8

    def convert_model(self, model: nn.Module) -> Tuple[nn.Module, int, int]:
        size_before = self.measure_model_size(model, bit_width=32)
        quant_params = self._quantizer.export_params()
        bit_width = quant_params.get("bit_width", 32)
        size_after = self.measure_model_size(model, bit_width=int(bit_width))
        return model, size_before, size_after

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantizer(x)
