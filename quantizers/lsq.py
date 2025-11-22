from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseQuantizer, BaseQuantizerWrapper


class LSQQuantizer(BaseQuantizer):
    def __init__(self, bit_width: int = 32, per_channel: bool = False):
        super().__init__(bit_width=bit_width, per_channel=per_channel)
        self.thd_pos = 2 ** (bit_width - 1)
        self.thd_neg = -2 ** (bit_width - 1) - 1
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            s = (x.max() - x.min()) / (self.thd_pos - self.thd_neg)
            s = torch.clamp(s, min=1e-4)
            self.s.data = s.to(self.s.data.dtype)

    def skip_grad_scale(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    def round_pass(self, x: torch.Tensor) -> torch.Tensor:
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bit_width >= 32:
            return x
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scaled = self.skip_grad_scale(self.s, s_grad_scale).to(x.device)
        x = torch.clamp(x / s_scaled, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)
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


class LSQConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, bit_width: int):
        super().__init__()
        self.conv = conv
        self.act_quant = LSQQuantizer(bit_width=bit_width)
        self.weight_quant = LSQQuantizer(bit_width=bit_width)
        self.bias_quant = LSQQuantizer(bit_width=bit_width)
        with torch.no_grad():
            self.weight_quant.init_from(self.conv.weight)
            if self.conv.bias is not None:
                self.bias_quant.init_from(self.conv.bias)
        # Expose weight and bias so external code that accesses
        # module.weight / module.bias continues to work.
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_act = self.act_quant(x)
        q_weight = self.weight_quant(self.conv.weight)
        bias = self.conv.bias
        if bias is not None:
            bias = self.bias_quant(bias)
        return F.conv2d(
            q_act,
            q_weight,
            bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


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
        # Expose weight and bias so modules like MultiheadAttention
        # that access out_proj.weight / .bias keep working.
        self.weight = self.fc.weight
        self.bias = self.fc.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_act = self.act_quant(x)
        q_weight = self.weight_quant(self.fc.weight)
        bias = self.fc.bias
        if bias is not None:
            bias = self.bias_quant(bias)
        return F.linear(q_act, q_weight, bias)


class LSQEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding, bit_width: int):
        super().__init__()
        self.emb = emb
        self.weight_quant = LSQQuantizer(bit_width=bit_width)
        with torch.no_grad():
            self.weight_quant.init_from(self.emb.weight)
        self.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quant(self.emb.weight)
        return F.embedding(
            x,
            q_weight,
            self.emb.padding_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )


class LSQLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm, bit_width: int):
        super().__init__()
        self.ln = ln
        self.weight_quant = LSQQuantizer(bit_width=bit_width)
        self.bias_quant = LSQQuantizer(bit_width=bit_width)
        with torch.no_grad():
            if self.ln.weight is not None:
                self.weight_quant.init_from(self.ln.weight)
            if self.ln.bias is not None:
                self.bias_quant.init_from(self.ln.bias)
        self.weight = self.ln.weight
        self.bias = self.ln.bias
        # Expose attributes expected from nn.LayerNorm
        self.eps = self.ln.eps
        self.normalized_shape = self.ln.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ln.weight
        bias = self.ln.bias
        if weight is not None:
            weight = self.weight_quant(weight)
        if bias is not None:
            bias = self.bias_quant(bias)
        return F.layer_norm(x, self.ln.normalized_shape, weight, bias, self.ln.eps)


class LSQMultiheadAttention(nn.Module):
    def __init__(self, attn: nn.MultiheadAttention, bit_width: int):
        super().__init__()
        self.attn = attn
        self.act_quant = LSQQuantizer(bit_width=bit_width)
        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.batch_first = getattr(attn, "batch_first", False)

        # Expose standard MultiheadAttention attributes so external
        # modules (e.g., TransformerEncoderLayer) can access them.
        self.in_proj_weight = attn.in_proj_weight
        self.in_proj_bias = attn.in_proj_bias
        self.q_proj_weight = getattr(attn, "q_proj_weight", None)
        self.k_proj_weight = getattr(attn, "k_proj_weight", None)
        self.v_proj_weight = getattr(attn, "v_proj_weight", None)
        self.out_proj = attn.out_proj
        self.bias_k = attn.bias_k
        self.bias_v = attn.bias_v
        self.add_zero_attn = attn.add_zero_attn
        self.dropout = attn.dropout
        self._qkv_same_embed_dim = attn._qkv_same_embed_dim

        self.in_proj_weight_quant = None
        self.in_proj_bias_quant = None
        if attn.in_proj_weight is not None:
            self.in_proj_weight_quant = LSQQuantizer(bit_width=bit_width)
            with torch.no_grad():
                self.in_proj_weight_quant.init_from(attn.in_proj_weight)
        if attn.in_proj_bias is not None:
            self.in_proj_bias_quant = LSQQuantizer(bit_width=bit_width)
            with torch.no_grad():
                self.in_proj_bias_quant.init_from(attn.in_proj_bias)

        self.q_proj_weight_quant = None
        self.k_proj_weight_quant = None
        self.v_proj_weight_quant = None
        if getattr(attn, "q_proj_weight", None) is not None:
            self.q_proj_weight_quant = LSQQuantizer(bit_width=bit_width)
            self.k_proj_weight_quant = LSQQuantizer(bit_width=bit_width)
            self.v_proj_weight_quant = LSQQuantizer(bit_width=bit_width)
            with torch.no_grad():
                self.q_proj_weight_quant.init_from(attn.q_proj_weight)
                self.k_proj_weight_quant.init_from(attn.k_proj_weight)
                self.v_proj_weight_quant.init_from(attn.v_proj_weight)

        self.out_proj_weight_quant = LSQQuantizer(bit_width=bit_width)
        self.out_proj_bias_quant = LSQQuantizer(bit_width=bit_width)
        with torch.no_grad():
            self.out_proj_weight_quant.init_from(attn.out_proj.weight)
            if attn.out_proj.bias is not None:
                self.out_proj_bias_quant.init_from(attn.out_proj.bias)

        self.bias_k_quant = None
        self.bias_v_quant = None
        if attn.bias_k is not None:
            self.bias_k_quant = LSQQuantizer(bit_width=bit_width)
            with torch.no_grad():
                self.bias_k_quant.init_from(attn.bias_k)
        if attn.bias_v is not None:
            self.bias_v_quant = LSQQuantizer(bit_width=bit_width)
            with torch.no_grad():
                self.bias_v_quant.init_from(attn.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        q = self.act_quant(query)
        k = self.act_quant(key)
        v = self.act_quant(value)

        if self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        attn = self.attn
        in_proj_weight = attn.in_proj_weight
        in_proj_bias = attn.in_proj_bias
        if in_proj_weight is not None and self.in_proj_weight_quant is not None:
            in_proj_weight = self.in_proj_weight_quant(in_proj_weight)
        if in_proj_bias is not None and self.in_proj_bias_quant is not None:
            in_proj_bias = self.in_proj_bias_quant(in_proj_bias)

        q_proj_weight = getattr(attn, "q_proj_weight", None)
        k_proj_weight = getattr(attn, "k_proj_weight", None)
        v_proj_weight = getattr(attn, "v_proj_weight", None)
        if q_proj_weight is not None and self.q_proj_weight_quant is not None:
            q_proj_weight = self.q_proj_weight_quant(q_proj_weight)
            k_proj_weight = self.k_proj_weight_quant(k_proj_weight)
            v_proj_weight = self.v_proj_weight_quant(v_proj_weight)

        out_proj_weight = attn.out_proj.weight
        out_proj_bias = attn.out_proj.bias
        out_proj_weight = self.out_proj_weight_quant(out_proj_weight)
        if out_proj_bias is not None:
            out_proj_bias = self.out_proj_bias_quant(out_proj_bias)

        bias_k = attn.bias_k
        bias_v = attn.bias_v
        if bias_k is not None and self.bias_k_quant is not None:
            bias_k = self.bias_k_quant(bias_k)
        if bias_v is not None and self.bias_v_quant is not None:
            bias_v = self.bias_v_quant(bias_v)

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            q,
            k,
            v,
            self.embed_dim,
            self.num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            attn.add_zero_attn,
            attn.dropout,
            out_proj_weight,
            out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=not attn._qkv_same_embed_dim,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=None,
            static_v=None,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


class LSQQuantizerWrapper(BaseQuantizerWrapper):
    def __init__(self, quantizer: LSQQuantizer, logging_backend: str = "none"):
        super().__init__(quantizer, logging_backend=logging_backend)

    def prepare_qat_model(self, model: nn.Module) -> nn.Module:
        for name, module in list(model.named_children()):
            if isinstance(module, nn.MultiheadAttention):
                setattr(model, name, LSQMultiheadAttention(module, bit_width=self._quantizer.bit_width))
            elif isinstance(module, nn.Linear):
                setattr(model, name, LSQLinear(module, bit_width=self._quantizer.bit_width))
            elif isinstance(module, nn.Conv2d):
                setattr(model, name, LSQConv2d(module, bit_width=self._quantizer.bit_width))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, LSQEmbedding(module, bit_width=self._quantizer.bit_width))
            elif isinstance(module, nn.LayerNorm):
                setattr(model, name, LSQLayerNorm(module, bit_width=self._quantizer.bit_width))
            else:
                self.prepare_qat_model(module)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def optimize_ptq(self, model: nn.Module, dataloader, device, **kwargs) -> nn.Module:
        return model