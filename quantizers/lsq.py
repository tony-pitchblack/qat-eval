from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseQuantizer


class LSQQuantizer(BaseQuantizer):
    def __init__(self, bit_width: int = 32, per_channel: bool = False):
        super().__init__(bit_width=bit_width, per_channel=per_channel)
        self.thd_pos = 2 ** (bit_width - 1)
        self.thd_neg = -2 ** (bit_width - 1) - 1
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            s = (x.max() - x.min()) / (self.thd_pos - self.thd_neg)
            self.s.data = s.to(self.s.data.dtype)

    def _skip_grad_scale(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    def _round_pass(self, x: torch.Tensor) -> torch.Tensor:
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bit_width >= 32:
            return x
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scaled = self._skip_grad_scale(self.s, s_grad_scale).to(x.device)
        x = torch.clamp(x / s_scaled, self.thd_neg, self.thd_pos)
        x = self._round_pass(x)
        x = x * s_scaled
        return x

    def export_params(self) -> Dict[str, Any]:
        return {
            "type": "lsq",
            "bit_width": self.bit_width,
            "per_channel": self.per_channel,
            "scale": self.s.detach().cpu().tolist(),
            "thd_neg": int(self.thd_neg),
            "thd_pos": int(self.thd_pos),
        }


class LSQLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bit_width: int):
        super().__init__()
        self.fc = linear
        self.act_quant = LSQQuantizer(bit_width=bit_width)
        self.weight_quant = LSQQuantizer(bit_width=bit_width)
        self.bias_quant = LSQQuantizer(bit_width=bit_width)
        with torch.no_grad():
            self.weight_quant.init_from(self.fc.weight)
            if self.fc.bias is not None:
                self.bias_quant.init_from(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_act = self.act_quant(x)
        q_weight = self.weight_quant(self.fc.weight)
        bias = self.fc.bias
        if bias is not None:
            bias = self.bias_quant(bias)
        return F.linear(q_act, q_weight, bias)

