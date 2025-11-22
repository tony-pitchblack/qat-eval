from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseQuantizer, BaseQuantizerWrapper


class PACTQuantizer(BaseQuantizer):
    def __init__(self, bit_width: int = 8, per_channel: bool = False, init_alpha: float = 6.0):
        super().__init__(bit_width=bit_width, per_channel=per_channel)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    @staticmethod
    def _round_ste(x: torch.Tensor) -> torch.Tensor:
        y = x.round()
        return (y - x).detach() + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bit_width is None or self.bit_width >= 32:
            return x
        int_max = 2 ** (self.bit_width - 1) - 1
        if int_max <= 0:
            return x
        alpha = torch.clamp(self.alpha.to(x.device), min=1e-6)
        x_clipped = torch.clamp(x, -alpha, alpha)
        scale = alpha / float(int_max)
        x_scaled = x_clipped / scale
        x_q = self._round_ste(x_scaled)
        x_q = torch.clamp(x_q, -int_max, int_max)
        return x_q * scale

    def export_params(self) -> Dict[str, Any]:
        return {
            "type": "pact",
            "bit_width": self.bit_width,
            "per_channel": self.per_channel,
            "alpha": float(self.alpha.detach().cpu().item()),
        }


class SymmetricUniformQuantizer(nn.Module):
    def __init__(self, bit_width: int = 8, per_channel: bool = False):
        super().__init__()
        self.bit_width = bit_width
        self.per_channel = per_channel

    @staticmethod
    def _round_ste(x: torch.Tensor) -> torch.Tensor:
        y = x.round()
        return (y - x).detach() + x

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.bit_width is None or self.bit_width >= 32:
            return w
        int_max = 2 ** (self.bit_width - 1) - 1
        if int_max <= 0:
            return w
        if self.per_channel and w.dim() >= 2:
            reduce_dims = tuple(range(1, w.dim()))
            max_abs = w.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_abs = w.abs().max()
        if max_abs.numel() == 0:
            return w
        max_abs = torch.clamp(max_abs, min=1e-8)
        scale = max_abs / float(int_max)
        w_scaled = w / scale
        w_q = self._round_ste(w_scaled)
        w_q = torch.clamp(w_q, -int_max, int_max)
        return w_q * scale


class PACTLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bit_width: int):
        super().__init__()
        self.fc = linear
        self.act_quant = PACTQuantizer(bit_width=bit_width)
        self.weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=True)
        self.bias_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)
        self.weight = self.fc.weight
        self.bias = self.fc.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x = self.act_quant(x)
        q_w = self.weight_quant(self.fc.weight)
        bias = self.fc.bias
        if bias is not None:
            bias = self.bias_quant(bias)
        return F.linear(q_x, q_w, bias)


class PACTEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding, bit_width: int):
        super().__init__()
        self.emb = emb
        self.weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)
        self.act_quant = PACTQuantizer(bit_width=bit_width)
        self.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_w = self.weight_quant(self.emb.weight)
        out = F.embedding(
            x,
            q_w,
            self.emb.padding_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )
        return self.act_quant(out)


class PACTLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm, bit_width: int):
        super().__init__()
        self.ln = ln
        self.weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)
        self.bias_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)
        self.act_quant = PACTQuantizer(bit_width=bit_width)
        self.weight = self.ln.weight
        self.bias = self.ln.bias
        self.eps = self.ln.eps
        self.normalized_shape = self.ln.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ln.weight
        bias = self.ln.bias
        if weight is not None:
            weight = self.weight_quant(weight)
        if bias is not None:
            bias = self.bias_quant(bias)
        out = F.layer_norm(x, self.ln.normalized_shape, weight, bias, self.ln.eps)
        return self.act_quant(out)


class PACTMultiheadAttention(nn.Module):
    def __init__(self, attn: nn.MultiheadAttention, bit_width: int):
        super().__init__()
        self.attn = attn
        self.act_quant = PACTQuantizer(bit_width=bit_width)
        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.batch_first = getattr(attn, "batch_first", False)
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
            self.in_proj_weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=True)
        if attn.in_proj_bias is not None:
            self.in_proj_bias_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)

        self.q_proj_weight_quant = None
        self.k_proj_weight_quant = None
        self.v_proj_weight_quant = None
        if getattr(attn, "q_proj_weight", None) is not None:
            self.q_proj_weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=True)
            self.k_proj_weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=True)
            self.v_proj_weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=True)

        self.out_proj_weight_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=True)
        self.out_proj_bias_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)

        self.bias_k_quant = None
        self.bias_v_quant = None
        if attn.bias_k is not None:
            self.bias_k_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)
        if attn.bias_v is not None:
            self.bias_v_quant = SymmetricUniformQuantizer(bit_width=bit_width, per_channel=False)

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


class PACTQuantizerWrapper(BaseQuantizerWrapper):
    def __init__(self, quantizer: PACTQuantizer, logging_backend: str = "none"):
        super().__init__(quantizer, logging_backend=logging_backend)

    def prepare_qat_model(self, model: nn.Module) -> nn.Module:
        bit_width = self._quantizer.bit_width
        for name, module in list(model.named_children()):
            if isinstance(module, nn.MultiheadAttention):
                setattr(model, name, PACTMultiheadAttention(module, bit_width=bit_width))
            elif isinstance(module, nn.Linear):
                setattr(model, name, PACTLinear(module, bit_width=bit_width))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, PACTEmbedding(module, bit_width=bit_width))
            elif isinstance(module, nn.LayerNorm):
                setattr(model, name, PACTLayerNorm(module, bit_width=bit_width))
            else:
                self.prepare_qat_model(module)
        return model

    def prepare_for_inference(self, model: nn.Module, **kwargs) -> nn.Module:
        def convert_module(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, PACTLinear):
                    with torch.no_grad():
                        q_w = child.weight_quant(child.fc.weight)
                        q_b = None
                        if child.fc.bias is not None:
                            q_b = child.bias_quant(child.fc.bias)
                    new_layer = nn.Linear(
                        in_features=child.fc.in_features,
                        out_features=child.fc.out_features,
                        bias=child.fc.bias is not None,
                    )
                    new_layer.weight.data = q_w
                    if q_b is not None:
                        new_layer.bias.data = q_b
                    setattr(module, name, new_layer)
                elif isinstance(child, PACTEmbedding):
                    with torch.no_grad():
                        q_w = child.weight_quant(child.emb.weight)
                    new_layer = nn.Embedding(
                        num_embeddings=child.emb.num_embeddings,
                        embedding_dim=child.emb.embedding_dim,
                        padding_idx=child.emb.padding_idx,
                        max_norm=child.emb.max_norm,
                        norm_type=child.emb.norm_type,
                        scale_grad_by_freq=child.emb.scale_grad_by_freq,
                        sparse=child.emb.sparse,
                    )
                    new_layer.weight.data = q_w
                    setattr(module, name, new_layer)
                elif isinstance(child, PACTLayerNorm):
                    with torch.no_grad():
                        w = child.ln.weight
                        b = child.ln.bias
                        if w is not None:
                            w = child.weight_quant(w)
                        if b is not None:
                            b = child.bias_quant(b)
                    new_layer = nn.LayerNorm(
                        normalized_shape=child.ln.normalized_shape,
                        eps=child.ln.eps,
                    )
                    if w is not None:
                        new_layer.weight.data = w
                    if b is not None:
                        new_layer.bias.data = b
                    setattr(module, name, new_layer)
                elif isinstance(child, PACTMultiheadAttention):
                    continue
                else:
                    convert_module(child)

        convert_module(model)
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

