import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseQuantizer(nn.Module, ABC):
    """
    Wraps one tensor (weights or activations).
    All QAT methods implement this interface.
    """
    def __init__(self, bit_width: int, per_channel: bool = False):
        super().__init__()
        self.bit_width = bit_width
        self.per_channel = per_channel

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize + dequantize during QAT."""
        ...

    @abstractmethod
    def export_params(self):
        """Return static quantization params for deployment (scales, zero-points, etc.)."""
        ...
