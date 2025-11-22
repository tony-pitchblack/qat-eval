from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseQuantizer, BaseQuantizerWrapper


class QILQuantizer(BaseQuantizer):
    """QIL quantizer from Jung et al. 2018."""
    
    def __init__(
        self, 
        bit_width: int = 32, 
        per_channel: bool = False,
        is_activation: bool = False,
        gamma: Optional[float] = None,
    ):
        super().__init__(bit_width=bit_width, per_channel=per_channel)
        self.is_activation = is_activation
        
        if is_activation:
            self.q = 2 ** bit_width - 1
        else:
            self.q = 2 ** (bit_width - 1) - 1
        
        self.c = nn.Parameter(torch.ones(1))
        self.d = nn.Parameter(torch.ones(1))
        
        if not is_activation:
            if gamma is None:
                self.gamma = nn.Parameter(torch.ones(1))
                self.trainable_gamma = True
            else:
                self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))
                self.trainable_gamma = False
        else:
            self.register_buffer('gamma', torch.tensor(1.0, dtype=torch.float32))
            self.trainable_gamma = False
    
    def init_from(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            if self.is_activation:
                x_mean = x.mean()
                x_std = x.std()
                self.c.data = x_mean.clamp(min=0).to(self.c.dtype)
                self.d.data = (2 * x_std).clamp(min=1e-5).to(self.d.dtype)
            else:
                x_std = x.std()
                self.c.data = torch.zeros_like(self.c.data)
                self.d.data = (2 * x_std).clamp(min=1e-5).to(self.d.dtype)
    
    def round_pass(self, x: torch.Tensor) -> torch.Tensor:
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bit_width >= 32:
            return x
        
        c = self.c.to(x.device)
        d = torch.clamp(self.d.to(x.device), min=1e-5)
        
        if self.is_activation:
            alpha_X = 0.5 / d
            beta_X = -0.5 * c / d + 0.5
            
            x_transformed = alpha_X * x + beta_X
            x_transformed = torch.clamp(x_transformed, 0, 1)
            x_discrete = self.round_pass(x_transformed * self.q) / self.q
            x_output = (x_discrete - beta_X) / alpha_X
            
        else:
            alpha_W = 0.5 / d
            beta_W = -0.5 * c / d
            gamma = self.gamma.to(x.device)
            
            sign = torch.sign(x)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            
            x_abs = torch.abs(x)
            x_transformed = alpha_W * x_abs + beta_W
            x_transformed = torch.clamp(x_transformed, 0, 1)
            x_transformed = torch.pow(x_transformed + 1e-8, gamma)
            
            x_discrete = self.round_pass(x_transformed * self.q)
            x_discrete = x_discrete / self.q
            
            x_inv_transformed = torch.pow(x_discrete + 1e-8, 1.0 / (gamma + 1e-8))
            x_abs_reconstructed = (x_inv_transformed - beta_W) / alpha_W
            
            x_output = sign * x_abs_reconstructed
            
            pruning_threshold = c - d
            mask = (torch.abs(x) >= pruning_threshold).float()
            x_output = x_output * mask
        
        return x_output
    
    def export_params(self) -> Dict[str, Any]:
        c = self.c.item()
        d = self.d.item()
        gamma_val = self.gamma.item() if isinstance(self.gamma, nn.Parameter) else self.gamma.item()
        
        return {
            "type": "qil",
            "bit_width": self.bit_width,
            "per_channel": self.per_channel,
            "is_activation": self.is_activation,
            "center": c,
            "distance": d,
            "gamma": gamma_val,
            "pruning_threshold": c - d,
            "clipping_threshold": c + d,
            "q_levels": int(self.q),
            "trainable_gamma": self.trainable_gamma,
        }


class QILConv2d(nn.Module):
    
    def __init__(self, conv: nn.Conv2d, bit_width: int, gamma_weight: Optional[float] = None):
        super().__init__()
        self.conv = conv
        
        self.act_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=True,
            gamma=1.0
        )
        
        if gamma_weight is None:
            gamma_weight = None if bit_width == 2 else 1.0
        
        self.weight_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=False,
            gamma=gamma_weight
        )
        
        self.bias_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=False,
            gamma=1.0
        )
        
        with torch.no_grad():
            self.weight_quant.init_from(self.conv.weight)
            if self.conv.bias is not None:
                self.bias_quant.init_from(self.conv.bias)
        
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


class QILLinear(nn.Module):
    
    def __init__(self, linear: nn.Linear, bit_width: int, gamma_weight: Optional[float] = None):
        super().__init__()
        self.fc = linear
        
        self.act_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=True,
            gamma=1.0
        )
        
        if gamma_weight is None:
            gamma_weight = None if bit_width == 2 else 1.0
        
        self.weight_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=False,
            gamma=gamma_weight
        )
        
        self.bias_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=False,
            gamma=1.0
        )
        
        with torch.no_grad():
            self.weight_quant.init_from(self.fc.weight)
            if self.fc.bias is not None:
                self.bias_quant.init_from(self.fc.bias)
        
        self.weight = self.fc.weight
        self.bias = self.fc.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_act = self.act_quant(x)
        q_weight = self.weight_quant(self.fc.weight)
        bias = self.fc.bias
        if bias is not None:
            bias = self.bias_quant(bias)
        
        return F.linear(q_act, q_weight, bias)


class QILEmbedding(nn.Module):
    
    def __init__(self, emb: nn.Embedding, bit_width: int, gamma_weight: Optional[float] = None):
        super().__init__()
        self.emb = emb
        
        if gamma_weight is None:
            gamma_weight = None if bit_width == 2 else 1.0
        
        self.weight_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=False,
            gamma=gamma_weight
        )
        
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


class QILLayerNorm(nn.Module):
    
    def __init__(self, ln: nn.LayerNorm, bit_width: int):
        super().__init__()
        self.ln = ln
        
        self.weight_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=False,
            gamma=1.0
        )
        
        self.bias_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=False,
            gamma=1.0
        )
        
        with torch.no_grad():
            if self.ln.weight is not None:
                self.weight_quant.init_from(self.ln.weight)
            if self.ln.bias is not None:
                self.bias_quant.init_from(self.ln.bias)
        
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
        
        return F.layer_norm(x, self.ln.normalized_shape, weight, bias, self.ln.eps)


class QILMultiheadAttention(nn.Module):
    
    def __init__(self, attn: nn.MultiheadAttention, bit_width: int, gamma_weight: Optional[float] = None):
        super().__init__()
        self.attn = attn
        
        if gamma_weight is None:
            gamma_weight = None if bit_width == 2 else 1.0
        
        self.act_quant = QILQuantizer(
            bit_width=bit_width, 
            is_activation=True,
            gamma=1.0
        )
        
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
            self.in_proj_weight_quant = QILQuantizer(
                bit_width=bit_width, 
                is_activation=False,
                gamma=gamma_weight
            )
            with torch.no_grad():
                self.in_proj_weight_quant.init_from(attn.in_proj_weight)
        
        if attn.in_proj_bias is not None:
            self.in_proj_bias_quant = QILQuantizer(
                bit_width=bit_width, 
                is_activation=False,
                gamma=1.0
            )
            with torch.no_grad():
                self.in_proj_bias_quant.init_from(attn.in_proj_bias)
        
        self.q_proj_weight_quant = None
        self.k_proj_weight_quant = None
        self.v_proj_weight_quant = None
        
        if getattr(attn, "q_proj_weight", None) is not None:
            self.q_proj_weight_quant = QILQuantizer(bit_width=bit_width, is_activation=False, gamma=gamma_weight)
            self.k_proj_weight_quant = QILQuantizer(bit_width=bit_width, is_activation=False, gamma=gamma_weight)
            self.v_proj_weight_quant = QILQuantizer(bit_width=bit_width, is_activation=False, gamma=gamma_weight)
            
            with torch.no_grad():
                self.q_proj_weight_quant.init_from(attn.q_proj_weight)
                self.k_proj_weight_quant.init_from(attn.k_proj_weight)
                self.v_proj_weight_quant.init_from(attn.v_proj_weight)
        
        self.out_proj_weight_quant = QILQuantizer(bit_width=bit_width, is_activation=False, gamma=gamma_weight)
        self.out_proj_bias_quant = QILQuantizer(bit_width=bit_width, is_activation=False, gamma=1.0)
        
        with torch.no_grad():
            self.out_proj_weight_quant.init_from(attn.out_proj.weight)
            if attn.out_proj.bias is not None:
                self.out_proj_bias_quant.init_from(attn.out_proj.bias)
        
        self.bias_k_quant = None
        self.bias_v_quant = None
        
        if attn.bias_k is not None:
            self.bias_k_quant = QILQuantizer(bit_width=bit_width, is_activation=False, gamma=1.0)
            with torch.no_grad():
                self.bias_k_quant.init_from(attn.bias_k)
        
        if attn.bias_v is not None:
            self.bias_v_quant = QILQuantizer(bit_width=bit_width, is_activation=False, gamma=1.0)
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
            q, k, v,
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


class QILQuantizerWrapper(BaseQuantizerWrapper):
    
    def __init__(
        self, 
        quantizer: QILQuantizer, 
        gamma_weight: Optional[float] = None,
        skip_first_last: bool = True,
        logging_backend: str = "none",
    ):
        super().__init__(quantizer, logging_backend=logging_backend)
        self.gamma_weight = gamma_weight
        self.skip_first_last = skip_first_last
        self._layer_count = 0
    
    def prepare_qat_model(self, model: nn.Module, is_first_call: bool = True) -> nn.Module:
        if is_first_call:
            self._layer_count = self._count_quantizable_layers(model)
        
        current_idx = [0]
        
        def replace_layers(module):
            for child_name, child_module in list(module.named_children()):
                is_quantizable = isinstance(child_module, (
                    nn.Linear, nn.Conv2d, nn.Embedding, 
                    nn.LayerNorm, nn.MultiheadAttention
                ))
                
                if is_quantizable:
                    current_idx[0] += 1
                    
                    if self.skip_first_last:
                        if current_idx[0] == 1 or current_idx[0] == self._layer_count:
                            continue
                
                if isinstance(child_module, nn.MultiheadAttention):
                    setattr(
                        module, 
                        child_name, 
                        QILMultiheadAttention(
                            child_module, 
                            bit_width=self._quantizer.bit_width,
                            gamma_weight=self.gamma_weight
                        )
                    )
                elif isinstance(child_module, nn.Linear):
                    setattr(
                        module, 
                        child_name, 
                        QILLinear(
                            child_module, 
                            bit_width=self._quantizer.bit_width, 
                            gamma_weight=self.gamma_weight
                        )
                    )
                elif isinstance(child_module, nn.Conv2d):
                    setattr(
                        module, 
                        child_name, 
                        QILConv2d(
                            child_module, 
                            bit_width=self._quantizer.bit_width, 
                            gamma_weight=self.gamma_weight
                        )
                    )
                elif isinstance(child_module, nn.Embedding):
                    setattr(
                        module, 
                        child_name, 
                        QILEmbedding(
                            child_module, 
                            bit_width=self._quantizer.bit_width,
                            gamma_weight=self.gamma_weight
                        )
                    )
                elif isinstance(child_module, nn.LayerNorm):
                    setattr(
                        module, 
                        child_name, 
                        QILLayerNorm(
                            child_module, 
                            bit_width=self._quantizer.bit_width
                        )
                    )
                else:
                    replace_layers(child_module)
        
        replace_layers(model)
        return model
    
    def _count_quantizable_layers(self, model: nn.Module) -> int:
        count = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding, 
                                  nn.LayerNorm, nn.MultiheadAttention)):
                count += 1
        return count
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
