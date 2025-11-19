from typing import Any, Dict

import torch

from ._base import BaseQuantizer


class AdaRoundQuantizer(BaseQuantizer):
    def __init__(self, bit_width: int = 32, per_channel: bool = False):
        super().__init__(bit_width=bit_width, per_channel=per_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AdaRoundQuantizer forward not implemented")

    def export_params(self) -> Dict[str, Any]:
        raise NotImplementedError("AdaRoundQuantizer export_params not implemented")



