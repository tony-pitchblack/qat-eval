from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseQuantizer, BaseQuantizerWrapper


class RCFFunction(torch.autograd.Function):
    """Reparameterized Clipping Function with exact gradients from Equation 8"""
    
    @staticmethod
    def forward(ctx, x, alpha, levels, gamma, eps):
        x_normalized = x / (alpha + eps)
        x_clipped = torch.clamp(x_normalized, -1.0, 1.0)
        
        levels_normalized = (levels * gamma).to(x.device)
        
        # Memory-efficient projection
        original_shape = x_clipped.shape
        x_flat = x_clipped.reshape(-1)
        
        batch_size = 10000
        num_elements = x_flat.numel()
        indices_list = []
        
        for i in range(0, num_elements, batch_size):
            end_idx = min(i + batch_size, num_elements)
            x_batch = x_flat[i:end_idx]
            x_exp = x_batch.unsqueeze(-1)
            levels_exp = levels_normalized.unsqueeze(0)
            distances = torch.abs(x_exp - levels_exp)
            batch_indices = torch.argmin(distances, dim=-1)
            indices_list.append(batch_indices)
        
        indices = torch.cat(indices_list, dim=0).reshape(original_shape)
        x_proj_normalized = levels_normalized[indices]
        
        output = x_proj_normalized * alpha
        
        ctx.save_for_backward(x, alpha, x_normalized, x_proj_normalized)
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, x_normalized, x_proj_normalized = ctx.saved_tensors
        
        grad_x = grad_output.clone()
        
        outlier_mask = torch.abs(x) > alpha
        grad_alpha_outlier = torch.sign(x) * outlier_mask.float()
        grad_alpha_inside = (x_proj_normalized - x_normalized) * (~outlier_mask).float()
        
        grad_alpha = (grad_alpha_outlier + grad_alpha_inside) * grad_output
        grad_alpha = grad_alpha.sum().view_as(alpha)
        
        return grad_x, grad_alpha, None, None, None


class APoTQuantizer(BaseQuantizer):
    """APoT Quantizer with running statistics for stable inference"""
    
    def __init__(self, bit_width: int = 32, per_channel: bool = False, k: int = 2):
        super().__init__(bit_width=bit_width, per_channel=per_channel)
        
        self.k = k
        self.eps = 1e-5
        
        if bit_width < 32:
            if bit_width % k == 1:
                self.n = (bit_width - 1) // k + 1
                self.odd_bit = True
            else:
                self.n = bit_width // k
                self.odd_bit = False
            
            self.levels = self._generate_levels()
            self.gamma = self._compute_gamma()
        else:
            self.n = 0
            self.odd_bit = False
            self.levels = None
            self.gamma = 1.0
        
        self.alpha = nn.Parameter(torch.ones(1))
        
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)
        self.momentum = 0.1
    
    def _compute_gamma(self) -> float:
        max_level = 0.0
        
        if self.odd_bit:
            for i in range(self.n - 1):
                max_level += 2.0 ** (-i)
            max_level += 2.0 ** (-(self.n - 1))
        else:
            for i in range(self.n):
                max_level += 2.0 ** (-i)
        
        return 1.0 / max_level
    
    def _generate_levels(self) -> torch.Tensor:
        levels_set = set()
        num_values = 2 ** self.k
        
        def generate_recursive(term_idx, current_sum):
            if term_idx == self.n:
                levels_set.add(current_sum)
                return
            
            if self.odd_bit and term_idx == self.n - 1:
                n_values = 2
            else:
                n_values = num_values
            
            for j in range(n_values):
                if j == 0:
                    value = 0.0
                else:
                    exponent = -(term_idx + (j - 1) * self.n)
                    value = 2.0 ** exponent
                
                generate_recursive(term_idx + 1, current_sum + value)
        
        generate_recursive(0, 0.0)
        
        positive_levels = sorted(levels_set)
        if positive_levels[0] == 0.0:
            positive_levels = positive_levels[1:]
        
        negative_levels = [-x for x in reversed(positive_levels)]
        all_levels = negative_levels + [0.0] + positive_levels
        
        return torch.tensor(all_levels, dtype=torch.float32)
    
    def weight_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Weight normalization with running statistics (like BatchNorm)"""
        
        if self.per_channel and x.dim() >= 2:
            dims = list(range(1, x.dim()))
            
            if not self.training and self.running_mean is not None:
                # Inference: use running stats
                mu = self.running_mean
                var = self.running_var
            else:
                # Training: compute stats
                mu = x.mean(dim=dims, keepdim=True)
                var = x.var(dim=dims, keepdim=True, unbiased=False)
                
                # Update running stats
                if self.training:
                    if self.running_mean is None:
                        self.running_mean = mu.detach().clone()
                        self.running_var = var.detach().clone()
                    else:
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.detach()
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            if not self.training and self.running_mean is not None:
                mu = self.running_mean
                var = self.running_var
            else:
                mu = x.mean()
                var = x.var(unbiased=False)
                
                if self.training:
                    if self.running_mean is None:
                        self.running_mean = mu.detach().clone()
                        self.running_var = var.detach().clone()
                    else:
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.detach()
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        
        sigma = torch.sqrt(var + self.eps)
        return (x - mu) / sigma
    
    def skip_grad_scale(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad
    
    def init_from(self, x: torch.Tensor, is_weight: bool = True) -> None:
        with torch.no_grad():
            if is_weight:
                alpha_init = 3.0
            else:
                max_val = torch.max(torch.abs(x)).item()
                alpha_init = min(8.0, max_val * 1.2)
            
            alpha_init = max(alpha_init, 1e-3)
            self.alpha.data = torch.tensor([alpha_init], dtype=self.alpha.dtype)
    
    def forward(self, x: torch.Tensor, is_weight: bool = False) -> torch.Tensor:
        if self.bit_width >= 32:
            return x
        
        if is_weight:
            x_input = self.weight_normalization(x)
        else:
            x_input = x
        
        grad_scale = 1.0 / ((2 ** self.bit_width * x.numel()) ** 0.5)
        alpha_scaled = self.skip_grad_scale(self.alpha, grad_scale).to(x.device)
        
        if self.levels is None:
            return x_input
        
        x_output = RCFFunction.apply(
            x_input, 
            alpha_scaled, 
            self.levels, 
            self.gamma, 
            self.eps
        )
        
        return x_output
    
    def export_params(self) -> Dict[str, Any]:
        return {
            "type": "apot",
            "bit_width": self.bit_width,
            "per_channel": self.per_channel,
            "k": self.k,
            "n": self.n,
            "odd_bit": self.odd_bit,
            "alpha": self.alpha.detach().cpu().item(),
            "gamma": self.gamma,
            "num_levels": len(self.levels) if self.levels is not None else 0,
        }


class APoTConv2d(nn.Module):
    """Quantized Conv2d with APoT"""
    
    def __init__(self, conv: nn.Conv2d, bit_width: int, k: int = 2):
        super().__init__()
        self.conv = conv
        
        self.act_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        self.weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=True, k=k)
        self.bias_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        
        with torch.no_grad():
            self.weight_quant.init_from(self.conv.weight, is_weight=True)
            if self.conv.bias is not None:
                self.bias_quant.init_from(self.conv.bias, is_weight=False)
        
        self.weight = self.conv.weight
        self.bias = self.conv.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_act = self.act_quant(x, is_weight=False)
        q_weight = self.weight_quant(self.conv.weight, is_weight=True)
        
        bias = self.conv.bias
        if bias is not None:
            bias = self.bias_quant(bias, is_weight=False)
        
        return F.conv2d(
            q_act,
            q_weight,
            bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class APoTLinear(nn.Module):
    """Quantized Linear with APoT"""
    
    def __init__(self, linear: nn.Linear, bit_width: int, k: int = 2):
        super().__init__()
        self.fc = linear
        
        self.act_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        self.weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=True, k=k)
        self.bias_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        
        with torch.no_grad():
            self.weight_quant.init_from(self.fc.weight, is_weight=True)
            if self.fc.bias is not None:
                self.bias_quant.init_from(self.fc.bias, is_weight=False)
        
        self.weight = self.fc.weight
        self.bias = self.fc.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_act = self.act_quant(x, is_weight=False)
        q_weight = self.weight_quant(self.fc.weight, is_weight=True)
        
        bias = self.fc.bias
        if bias is not None:
            bias = self.bias_quant(bias, is_weight=False)
        
        return F.linear(q_act, q_weight, bias)


class APoTEmbedding(nn.Module):
    """Quantized Embedding with APoT"""
    
    def __init__(self, emb: nn.Embedding, bit_width: int, k: int = 2):
        super().__init__()
        self.emb = emb
        self.weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        
        with torch.no_grad():
            self.weight_quant.init_from(self.emb.weight, is_weight=True)
        
        self.weight = self.emb.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quant(self.emb.weight, is_weight=True)
        
        return F.embedding(
            x,
            q_weight,
            self.emb.padding_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )


class APoTLayerNorm(nn.Module):
    """Quantized LayerNorm with APoT"""
    
    def __init__(self, ln: nn.LayerNorm, bit_width: int, k: int = 2):
        super().__init__()
        self.ln = ln
        
        self.weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        self.bias_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        
        with torch.no_grad():
            if self.ln.weight is not None:
                self.weight_quant.init_from(self.ln.weight, is_weight=True)
            if self.ln.bias is not None:
                self.bias_quant.init_from(self.ln.bias, is_weight=False)
        
        self.weight = self.ln.weight
        self.bias = self.ln.bias
        self.eps = self.ln.eps
        self.normalized_shape = self.ln.normalized_shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ln.weight
        bias = self.ln.bias
        
        if weight is not None:
            weight = self.weight_quant(weight, is_weight=True)
        if bias is not None:
            bias = self.bias_quant(bias, is_weight=False)
        
        return F.layer_norm(x, self.ln.normalized_shape, weight, bias, self.ln.eps)


class APoTLSTM(nn.Module):
    """Quantized LSTM with APoT"""
    
    def __init__(self, lstm: nn.LSTM, bit_width: int, k: int = 2):
        super().__init__()
        self.lstm = lstm
        
        # Store LSTM attributes
        self.input_size = lstm.input_size
        self.hidden_size = lstm.hidden_size
        self.num_layers = lstm.num_layers
        self.bias = lstm.bias
        self.batch_first = lstm.batch_first
        self.dropout = lstm.dropout
        self.bidirectional = lstm.bidirectional
        
        # Create quantizers for each layer's weights
        self.weight_quantizers = nn.ModuleDict()
        
        num_directions = 2 if self.bidirectional else 1
        
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = f"_reverse" if direction == 1 else ""
                prefix = f"l{layer}{suffix}"
                
                # Quantizers for input-hidden and hidden-hidden weights
                self.weight_quantizers[f"{prefix}_ih"] = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
                self.weight_quantizers[f"{prefix}_hh"] = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
                
                # Initialize quantizers from actual weights
                with torch.no_grad():
                    weight_ih = getattr(lstm, f'weight_ih_l{layer}{suffix}')
                    weight_hh = getattr(lstm, f'weight_hh_l{layer}{suffix}')
                    self.weight_quantizers[f"{prefix}_ih"].init_from(weight_ih, is_weight=True)
                    self.weight_quantizers[f"{prefix}_hh"].init_from(weight_hh, is_weight=True)
                
                # Bias quantizers if bias is used
                if self.bias:
                    self.weight_quantizers[f"{prefix}_ih_bias"] = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
                    self.weight_quantizers[f"{prefix}_hh_bias"] = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
                    with torch.no_grad():
                        bias_ih = getattr(lstm, f'bias_ih_l{layer}{suffix}')
                        bias_hh = getattr(lstm, f'bias_hh_l{layer}{suffix}')
                        self.weight_quantizers[f"{prefix}_ih_bias"].init_from(bias_ih, is_weight=False)
                        self.weight_quantizers[f"{prefix}_hh_bias"].init_from(bias_hh, is_weight=False)
    
    def forward(self, input, hx=None):
        # Quantize all LSTM weights
        num_directions = 2 if self.bidirectional else 1
        
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = f"_reverse" if direction == 1 else ""
                prefix = f"l{layer}{suffix}"
                
                # Get original weights
                weight_ih = getattr(self.lstm, f'weight_ih_l{layer}{suffix}')
                weight_hh = getattr(self.lstm, f'weight_hh_l{layer}{suffix}')
                
                # Quantize and replace
                q_weight_ih = self.weight_quantizers[f"{prefix}_ih"](weight_ih, is_weight=True)
                q_weight_hh = self.weight_quantizers[f"{prefix}_hh"](weight_hh, is_weight=True)
                
                # Temporarily replace weights (for forward pass)
                setattr(self.lstm, f'weight_ih_l{layer}{suffix}', nn.Parameter(q_weight_ih))
                setattr(self.lstm, f'weight_hh_l{layer}{suffix}', nn.Parameter(q_weight_hh))
                
                if self.bias:
                    bias_ih = getattr(self.lstm, f'bias_ih_l{layer}{suffix}')
                    bias_hh = getattr(self.lstm, f'bias_hh_l{layer}{suffix}')
                    q_bias_ih = self.weight_quantizers[f"{prefix}_ih_bias"](bias_ih, is_weight=False)
                    q_bias_hh = self.weight_quantizers[f"{prefix}_hh_bias"](bias_hh, is_weight=False)
                    setattr(self.lstm, f'bias_ih_l{layer}{suffix}', nn.Parameter(q_bias_ih))
                    setattr(self.lstm, f'bias_hh_l{layer}{suffix}', nn.Parameter(q_bias_hh))
        
        # Forward through LSTM with quantized weights
        output, hx = self.lstm(input, hx)
        
        # Restore original weights (to maintain gradient flow)
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = f"_reverse" if direction == 1 else ""
                weight_ih = getattr(self.lstm, f'weight_ih_l{layer}{suffix}')
                weight_hh = getattr(self.lstm, f'weight_hh_l{layer}{suffix}')
                setattr(self.lstm, f'weight_ih_l{layer}{suffix}', nn.Parameter(weight_ih.data))
                setattr(self.lstm, f'weight_hh_l{layer}{suffix}', nn.Parameter(weight_hh.data))
                
                if self.bias:
                    bias_ih = getattr(self.lstm, f'bias_ih_l{layer}{suffix}')
                    bias_hh = getattr(self.lstm, f'bias_hh_l{layer}{suffix}')
                    setattr(self.lstm, f'bias_ih_l{layer}{suffix}', nn.Parameter(bias_ih.data))
                    setattr(self.lstm, f'bias_hh_l{layer}{suffix}', nn.Parameter(bias_hh.data))
        
        return output, hx


class APoTMultiheadAttention(nn.Module):
    """Quantized MultiheadAttention with APoT"""
    
    def __init__(self, attn: nn.MultiheadAttention, bit_width: int, k: int = 2):
        super().__init__()
        self.attn = attn
        self.act_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        
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
            self.in_proj_weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=True, k=k)
            with torch.no_grad():
                self.in_proj_weight_quant.init_from(attn.in_proj_weight, is_weight=True)
        if attn.in_proj_bias is not None:
            self.in_proj_bias_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
            with torch.no_grad():
                self.in_proj_bias_quant.init_from(attn.in_proj_bias, is_weight=False)
        
        self.q_proj_weight_quant = None
        self.k_proj_weight_quant = None
        self.v_proj_weight_quant = None
        if getattr(attn, "q_proj_weight", None) is not None:
            self.q_proj_weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=True, k=k)
            self.k_proj_weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=True, k=k)
            self.v_proj_weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=True, k=k)
            with torch.no_grad():
                self.q_proj_weight_quant.init_from(attn.q_proj_weight, is_weight=True)
                self.k_proj_weight_quant.init_from(attn.k_proj_weight, is_weight=True)
                self.v_proj_weight_quant.init_from(attn.v_proj_weight, is_weight=True)
        
        self.out_proj_weight_quant = APoTQuantizer(bit_width=bit_width, per_channel=True, k=k)
        self.out_proj_bias_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
        with torch.no_grad():
            self.out_proj_weight_quant.init_from(attn.out_proj.weight, is_weight=True)
            if attn.out_proj.bias is not None:
                self.out_proj_bias_quant.init_from(attn.out_proj.bias, is_weight=False)
        
        self.bias_k_quant = None
        self.bias_v_quant = None
        if attn.bias_k is not None:
            self.bias_k_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
            with torch.no_grad():
                self.bias_k_quant.init_from(attn.bias_k, is_weight=False)
        if attn.bias_v is not None:
            self.bias_v_quant = APoTQuantizer(bit_width=bit_width, per_channel=False, k=k)
            with torch.no_grad():
                self.bias_v_quant.init_from(attn.bias_v, is_weight=False)
    
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
        q = self.act_quant(query, is_weight=False)
        k = self.act_quant(key, is_weight=False)
        v = self.act_quant(value, is_weight=False)
        
        if self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        
        attn = self.attn
        
        in_proj_weight = attn.in_proj_weight
        in_proj_bias = attn.in_proj_bias
        if in_proj_weight is not None and self.in_proj_weight_quant is not None:
            in_proj_weight = self.in_proj_weight_quant(in_proj_weight, is_weight=True)
        if in_proj_bias is not None and self.in_proj_bias_quant is not None:
            in_proj_bias = self.in_proj_bias_quant(in_proj_bias, is_weight=False)
        
        q_proj_weight = getattr(attn, "q_proj_weight", None)
        k_proj_weight = getattr(attn, "k_proj_weight", None)
        v_proj_weight = getattr(attn, "v_proj_weight", None)
        if q_proj_weight is not None and self.q_proj_weight_quant is not None:
            q_proj_weight = self.q_proj_weight_quant(q_proj_weight, is_weight=True)
            k_proj_weight = self.k_proj_weight_quant(k_proj_weight, is_weight=True)
            v_proj_weight = self.v_proj_weight_quant(v_proj_weight, is_weight=True)
        
        out_proj_weight = attn.out_proj.weight
        out_proj_bias = attn.out_proj.bias
        out_proj_weight = self.out_proj_weight_quant(out_proj_weight, is_weight=True)
        if out_proj_bias is not None:
            out_proj_bias = self.out_proj_bias_quant(out_proj_bias, is_weight=False)
        
        bias_k = attn.bias_k
        bias_v = attn.bias_v
        if bias_k is not None and self.bias_k_quant is not None:
            bias_k = self.bias_k_quant(bias_k, is_weight=False)
        if bias_v is not None and self.bias_v_quant is not None:
            bias_v = self.bias_v_quant(bias_v, is_weight=False)
        
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


class APoTLinearInference(nn.Module):
    """Inference Linear with frozen quantized weights"""
    
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class APoTConv2dInference(nn.Module):
    """Inference Conv2d with frozen quantized weights"""
    
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class APoTQuantizerWrapper(BaseQuantizerWrapper):
    """Wrapper to apply APoT quantization to entire model"""
    
    def __init__(self, quantizer: APoTQuantizer, k: int = 2, logging_backend: str = "none"):
        super().__init__(quantizer, logging_backend=logging_backend)
        self.k = k
    
    def prepare_qat_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for QAT training"""
        for name, module in list(model.named_children()):
            if isinstance(module, nn.MultiheadAttention):
                setattr(model, name, APoTMultiheadAttention(
                    module, bit_width=self._quantizer.bit_width, k=self.k
                ))
            elif isinstance(module, nn.LSTM):
                setattr(model, name, APoTLSTM(
                    module, bit_width=self._quantizer.bit_width, k=self.k
                ))
            elif isinstance(module, nn.Linear):
                setattr(model, name, APoTLinear(
                    module, bit_width=self._quantizer.bit_width, k=self.k
                ))
            elif isinstance(module, nn.Conv2d):
                setattr(model, name, APoTConv2d(
                    module, bit_width=self._quantizer.bit_width, k=self.k
                ))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, APoTEmbedding(
                    module, bit_width=self._quantizer.bit_width, k=self.k
                ))
            elif isinstance(module, nn.LayerNorm):
                setattr(model, name, APoTLayerNorm(
                    module, bit_width=self._quantizer.bit_width, k=self.k
                ))
            else:
                self.prepare_qat_model(module)
        return model
    
    def prepare_for_inference(self, model: nn.Module, **kwargs) -> nn.Module:
        """Convert APoT model to inference mode with frozen quantized weights"""
        print("Converting APoT model to inference mode...")
        
        def convert_module(module):
            for name, child in list(module.named_children()):
                if isinstance(child, APoTConv2d):
                    # Set to eval mode for running stats
                    child.weight_quant.eval()
                    child.bias_quant.eval()
                    
                    # Quantize weights once using running stats
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.conv.weight, is_weight=True)
                        q_bias = None
                        if child.conv.bias is not None:
                            q_bias = child.bias_quant(child.conv.bias, is_weight=False)
                    
                    # Create simple inference layer
                    new_layer = APoTConv2dInference(
                        weight=q_weight,
                        bias=q_bias,
                        stride=child.conv.stride,
                        padding=child.conv.padding,
                        dilation=child.conv.dilation,
                        groups=child.conv.groups
                    )
                    
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, APoTLinear):
                    child.weight_quant.eval()
                    child.bias_quant.eval()
                    
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.fc.weight, is_weight=True)
                        q_bias = None
                        if child.fc.bias is not None:
                            q_bias = child.bias_quant(child.fc.bias, is_weight=False)
                    
                    new_layer = APoTLinearInference(
                        weight=q_weight,
                        bias=q_bias
                    )
                    
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, APoTEmbedding):
                    child.weight_quant.eval()
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.emb.weight, is_weight=True)
                    
                    new_layer = nn.Embedding(
                        num_embeddings=child.emb.num_embeddings,
                        embedding_dim=child.emb.embedding_dim,
                        padding_idx=child.emb.padding_idx,
                        max_norm=child.emb.max_norm,
                        norm_type=child.emb.norm_type,
                        scale_grad_by_freq=child.emb.scale_grad_by_freq,
                        sparse=child.emb.sparse
                    )
                    new_layer.weight.data = q_weight
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, APoTLayerNorm):
                    child.weight_quant.eval()
                    child.bias_quant.eval()
                    
                    with torch.no_grad():
                        q_weight = child.ln.weight
                        q_bias = child.ln.bias
                        if q_weight is not None:
                            q_weight = child.weight_quant(q_weight, is_weight=True)
                        if q_bias is not None:
                            q_bias = child.bias_quant(q_bias, is_weight=False)
                    
                    new_layer = nn.LayerNorm(
                        normalized_shape=child.ln.normalized_shape,
                        eps=child.ln.eps
                    )
                    if q_weight is not None:
                        new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, APoTLSTM):
                    # Set quantizers to eval mode and quantize LSTM weights once
                    for quantizer in child.weight_quantizers.values():
                        quantizer.eval()
                    
                    with torch.no_grad():
                        num_directions = 2 if child.bidirectional else 1
                        
                        # Create a new LSTM with the same parameters
                        new_layer = nn.LSTM(
                            input_size=child.input_size,
                            hidden_size=child.hidden_size,
                            num_layers=child.num_layers,
                            bias=child.bias,
                            batch_first=child.batch_first,
                            dropout=child.dropout,
                            bidirectional=child.bidirectional
                        )
                        
                        # Copy quantized weights
                        for layer in range(child.num_layers):
                            for direction in range(num_directions):
                                suffix = f"_reverse" if direction == 1 else ""
                                prefix = f"l{layer}{suffix}"
                                
                                # Get and quantize weights
                                weight_ih = getattr(child.lstm, f'weight_ih_l{layer}{suffix}')
                                weight_hh = getattr(child.lstm, f'weight_hh_l{layer}{suffix}')
                                
                                q_weight_ih = child.weight_quantizers[f"{prefix}_ih"](weight_ih, is_weight=True)
                                q_weight_hh = child.weight_quantizers[f"{prefix}_hh"](weight_hh, is_weight=True)
                                
                                # Set quantized weights
                                getattr(new_layer, f'weight_ih_l{layer}{suffix}').data = q_weight_ih
                                getattr(new_layer, f'weight_hh_l{layer}{suffix}').data = q_weight_hh
                                
                                if child.bias:
                                    bias_ih = getattr(child.lstm, f'bias_ih_l{layer}{suffix}')
                                    bias_hh = getattr(child.lstm, f'bias_hh_l{layer}{suffix}')
                                    q_bias_ih = child.weight_quantizers[f"{prefix}_ih_bias"](bias_ih, is_weight=False)
                                    q_bias_hh = child.weight_quantizers[f"{prefix}_hh_bias"](bias_hh, is_weight=False)
                                    getattr(new_layer, f'bias_ih_l{layer}{suffix}').data = q_bias_ih
                                    getattr(new_layer, f'bias_hh_l{layer}{suffix}').data = q_bias_hh
                    
                    setattr(module, name, new_layer)
                    
                else:
                    convert_module(child)
        
        convert_module(model)
        model.eval()
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
