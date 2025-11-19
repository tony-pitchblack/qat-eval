from typing import Any, Dict

import torch

from ._base import BaseQuantizer


class NoQuantizer(BaseQuantizer):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def export_params(self) -> Dict[str, Any]:
        return {
            "type": "no_quant",
        }



