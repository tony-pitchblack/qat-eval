from typing import Any, Dict

import torch

from ._base import BaseQuantizer


class NoQuantizer(BaseQuantizer):
    def __init__(self, bit_width: int = None, per_channel: bool = False):
        super().__init__(bit_width=bit_width, per_channel=per_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def export_params(self) -> Dict[str, Any]:
        return {
            "type": "no_quant",
            "bit_width": self.bit_width,
            "per_channel": self.per_channel,
        }



