import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, List, Callable
from tqdm.auto import tqdm
from ._base import BaseQuantizer, BaseQuantizerWrapper


class AdaRoundQuantizer(BaseQuantizer):
    """AdaRound: Adaptive Rounding for Post-Training Quantization
    
    Paper: https://arxiv.org/abs/2004.10568
    """
    
    def __init__(
        self, 
        bit_width: int = 8, 
        per_channel: bool = False,
        symmetric: bool = True
    ):
        super().__init__(bit_width=bit_width, per_channel=per_channel)
        self.symmetric = symmetric
        self.thd_pos = 2 ** (bit_width - 1) - 1
        self.thd_neg = -2 ** (bit_width - 1) if symmetric else 0
        
        self.register_buffer('s', torch.ones(1))
        self.register_buffer('rounding', None)
        self.is_calibrated = False
        
    def init_scale_from_weights(self, weights: torch.Tensor) -> None:
        """Initialize scale to minimize MSE with round-to-nearest"""
        with torch.no_grad():
            if self.per_channel:
                if weights.dim() == 4:  # Conv2d
                    w_flat = weights.reshape(weights.size(0), -1)
                    w_max = w_flat.max(dim=1, keepdim=True)[0]
                    w_min = w_flat.min(dim=1, keepdim=True)[0]
                    s = (w_max - w_min) / (self.thd_pos - self.thd_neg)
                    s = s.reshape(-1, 1, 1, 1)
                elif weights.dim() == 2:  # Linear
                    w_max = weights.max(dim=1, keepdim=True)[0]
                    w_min = weights.min(dim=1, keepdim=True)[0]
                    s = (w_max - w_min) / (self.thd_pos - self.thd_neg)
                else:
                    s = (weights.max() - weights.min()) / (self.thd_pos - self.thd_neg)
            else:
                s = (weights.max() - weights.min()) / (self.thd_pos - self.thd_neg)
            
            s = torch.clamp(s, min=1e-8)
            self.s.copy_(s.to(self.s.dtype))
    
    def get_quantized_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply quantization with learned rounding"""
        if self.bit_width >= 32:
            return weight
            
        w_scaled = weight / self.s
        w_floor = torch.floor(w_scaled)
        
        if self.rounding is None:
            w_quant = torch.round(w_scaled)
        else:
            w_quant = w_floor + self.rounding
        
        w_quant = torch.clamp(w_quant, self.thd_neg, self.thd_pos)
        return w_quant * self.s
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_quantized_weight(x)
    
    def export_params(self) -> Dict[str, Any]:
        return {
            "type": "adaround",
            "bit_width": self.bit_width,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
            "scale": self.s.detach().cpu().tolist(),
            "thd_neg": int(self.thd_neg),
            "thd_pos": int(self.thd_pos),
            "rounding": self.rounding.detach().cpu().tolist() if self.rounding is not None else None,
            "is_calibrated": self.is_calibrated,
        }


class AdaRoundOptimizer:
    """AdaRound optimizer implementing the paper's algorithm"""
    
    def __init__(
        self,
        num_iterations: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-3,
        lambda_reg: float = 0.01,
        beta_range: Tuple[float, float] = (20.0, 2.0),
        zeta: float = 1.1,
        gamma: float = -0.1,
    ):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.beta_start, self.beta_end = beta_range
        self.zeta = zeta
        self.gamma = gamma
    
    def rectified_sigmoid(self, V: torch.Tensor) -> torch.Tensor:
        """h(V) = clip(σ(V(ζ - γ) + γ), 0, 1)"""
        return torch.clamp(
            torch.sigmoid(V * (self.zeta - self.gamma) + self.gamma),
            0, 1
        )
    
    def regularization_loss(self, h_V: torch.Tensor, beta: float) -> torch.Tensor:
        """f_reg(V) = Σ(1 - |2h(V) - 1|^β)"""
        return torch.sum(1 - torch.abs(2 * h_V - 1).pow(beta))
    
    def _create_batches(self, tensors: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """Create batches from list of tensors"""
        num_samples = len(tensors)
        indices = torch.randperm(num_samples)
        
        batches = []
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [tensors[idx] for idx in batch_indices]
            batches.append(batch)
        
        return batches
    
    def optimize_linear_layer(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        thd_neg: float,
        thd_pos: float,
        input_activations: List[torch.Tensor],
        target_outputs: List[torch.Tensor],
        activation_fn: Optional[Callable] = None,
        log_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ) -> torch.Tensor:
        """Optimize rounding for Linear layer using local MSE loss"""
        device = weight.device
        
        # Initialize V from fractional part
        with torch.no_grad():
            w_scaled = weight / scale
            w_floor = torch.floor(w_scaled)
            frac = w_scaled - w_floor
            frac = torch.clamp(frac, 0.01, 0.99)
            V = torch.log(frac / (1 - frac)) / (self.zeta - self.gamma) - \
                self.gamma / (self.zeta - self.gamma)
        
        V = nn.Parameter(V.to(device))
        optimizer = torch.optim.Adam([V], lr=self.lr)
        
        # Beta annealing schedule
        beta_decay = (self.beta_start / self.beta_end) ** (1 / self.num_iterations)
        current_beta = self.beta_start
        
        # Create batches
        input_batches = self._create_batches(input_activations)
        output_batches = self._create_batches(target_outputs)
        
        # Optimization loop
        pbar = tqdm(range(self.num_iterations), desc="adaround_linear", leave=False)
        for iteration in pbar:
            for input_batch, output_batch in zip(input_batches, output_batches):
                optimizer.zero_grad()
                
                # Stack batch
                x_batch = torch.stack([x.to(device) for x in input_batch])
                target_batch = torch.stack([y.to(device) for y in output_batch])
                
                # Reshape for batch processing
                if x_batch.dim() == 3:  # [B, seq, features]
                    x_batch = x_batch.reshape(-1, x_batch.size(-1))
                    target_batch = target_batch.reshape(-1, target_batch.size(-1))
                
                # Soft quantization
                h_V = self.rectified_sigmoid(V)
                w_soft = w_floor + h_V
                w_quant = torch.clamp(w_soft, thd_neg, thd_pos) * scale
                
                # Forward with quantized weights
                pred_batch = F.linear(x_batch, w_quant)
                if activation_fn is not None:
                    pred_batch = activation_fn(pred_batch)
                
                # Asymmetric reconstruction loss
                recon_loss = F.mse_loss(pred_batch, target_batch)
                
                # Regularization loss
                reg_loss = self.regularization_loss(h_V, current_beta)
                
                # Total loss
                loss = recon_loss + self.lambda_reg * reg_loss
                
                loss.backward()
                optimizer.step()

            if log_callback is not None:
                log_callback(
                    iteration,
                    float(recon_loss.detach().cpu()),
                    float(reg_loss.detach().cpu()),
                    float(loss.detach().cpu()),
                )
            pbar.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                recon=f"{float(recon_loss.detach().cpu()):.4f}",
                reg=f"{float(reg_loss.detach().cpu()):.4f}",
            )
            
            # Anneal beta
            current_beta = max(self.beta_end, current_beta / beta_decay)
        
        # Extract binary rounding
        with torch.no_grad():
            h_V_final = self.rectified_sigmoid(V)
            rounding = (h_V_final > 0.5).float()
        
        return rounding
    
    def optimize_conv2d_layer(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        thd_neg: float,
        thd_pos: float,
        input_activations: List[torch.Tensor],
        target_outputs: List[torch.Tensor],
        conv_params: Dict[str, Any],
        activation_fn: Optional[Callable] = None,
        log_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ) -> torch.Tensor:
        """Optimize rounding for Conv2d layer using local MSE loss"""
        device = weight.device
        
        # Initialize V
        with torch.no_grad():
            w_scaled = weight / scale
            w_floor = torch.floor(w_scaled)
            frac = w_scaled - w_floor
            frac = torch.clamp(frac, 0.01, 0.99)
            V = torch.log(frac / (1 - frac)) / (self.zeta - self.gamma) - \
                self.gamma / (self.zeta - self.gamma)
        
        V = nn.Parameter(V.to(device))
        optimizer = torch.optim.Adam([V], lr=self.lr)
        
        # Beta annealing
        beta_decay = (self.beta_start / self.beta_end) ** (1 / self.num_iterations)
        current_beta = self.beta_start
        
        # Create batches
        input_batches = self._create_batches(input_activations)
        output_batches = self._create_batches(target_outputs)
        
        # Optimization loop
        pbar = tqdm(range(self.num_iterations), desc="adaround_conv2d", leave=False)
        for iteration in pbar:
            for input_batch, output_batch in zip(input_batches, output_batches):
                optimizer.zero_grad()
                
                # Stack batch
                x_batch = torch.stack([x.to(device) for x in input_batch])
                target_batch = torch.stack([y.to(device) for y in output_batch])
                
                # Soft quantization
                h_V = self.rectified_sigmoid(V)
                w_soft = w_floor + h_V
                w_quant = torch.clamp(w_soft, thd_neg, thd_pos) * scale
                
                # Forward pass
                pred_batch = F.conv2d(
                    x_batch, w_quant,
                    stride=conv_params['stride'],
                    padding=conv_params['padding'],
                    dilation=conv_params['dilation'],
                    groups=conv_params['groups']
                )
                if activation_fn is not None:
                    pred_batch = activation_fn(pred_batch)
                
                # Losses
                recon_loss = F.mse_loss(pred_batch, target_batch)
                reg_loss = self.regularization_loss(h_V, current_beta)
                loss = recon_loss + self.lambda_reg * reg_loss
                
                loss.backward()
                optimizer.step()

            if log_callback is not None:
                log_callback(
                    iteration,
                    float(recon_loss.detach().cpu()),
                    float(reg_loss.detach().cpu()),
                    float(loss.detach().cpu()),
                )
            pbar.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                recon=f"{float(recon_loss.detach().cpu()):.4f}",
                reg=f"{float(reg_loss.detach().cpu()):.4f}",
            )
            
            current_beta = max(self.beta_end, current_beta / beta_decay)
        
        # Extract binary rounding
        with torch.no_grad():
            h_V_final = self.rectified_sigmoid(V)
            rounding = (h_V_final > 0.5).float()
        
        return rounding
    
    def optimize_embedding_layer(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        thd_neg: float,
        thd_pos: float,
        input_ids: List[torch.Tensor],
        log_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ) -> torch.Tensor:
        """Optimize rounding for Embedding layer"""
        device = weight.device
        
        # Initialize V
        with torch.no_grad():
            w_scaled = weight / scale
            w_floor = torch.floor(w_scaled)
            frac = w_scaled - w_floor
            frac = torch.clamp(frac, 0.01, 0.99)
            V = torch.log(frac / (1 - frac)) / (self.zeta - self.gamma) - \
                self.gamma / (self.zeta - self.gamma)
        
        V = nn.Parameter(V.to(device))
        opt = torch.optim.Adam([V], lr=self.lr)
        
        # Beta annealing
        beta_decay = (self.beta_start / self.beta_end) ** (1 / self.num_iterations)
        current_beta = self.beta_start
        
        # Optimization
        pbar = tqdm(range(self.num_iterations), desc="adaround_embedding", leave=False)
        for iteration in pbar:
            for ids in input_ids:
                ids = ids.to(device)
                opt.zero_grad()
                
                # Soft quantization
                h_V = self.rectified_sigmoid(V)
                w_soft = w_floor + h_V
                w_quant = torch.clamp(w_soft, thd_neg, thd_pos) * scale
                
                # Reconstruction loss
                original_emb = F.embedding(ids, weight)
                quant_emb = F.embedding(ids, w_quant)
                recon_loss = F.mse_loss(quant_emb, original_emb)
                
                # Regularization
                reg_loss = self.regularization_loss(h_V, current_beta)
                
                loss = recon_loss + self.lambda_reg * reg_loss
                loss.backward()
                opt.step()

            if log_callback is not None:
                log_callback(
                    iteration,
                    float(recon_loss.detach().cpu()),
                    float(reg_loss.detach().cpu()),
                    float(loss.detach().cpu()),
                )
            pbar.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                recon=f"{float(recon_loss.detach().cpu()):.4f}",
                reg=f"{float(reg_loss.detach().cpu()):.4f}",
            )
            
            current_beta = max(self.beta_end, current_beta / beta_decay)
        
        # Extract binary rounding
        with torch.no_grad():
            h_V_final = self.rectified_sigmoid(V)
            rounding = (h_V_final > 0.5).float()
        
        return rounding


class AdaRoundConv2d(nn.Module):
    """Conv2d with AdaRound PTQ"""
    
    def __init__(self, conv: nn.Conv2d, bit_width: int):
        super().__init__()
        self.conv = conv
        self.weight_quant = AdaRoundQuantizer(bit_width=bit_width, per_channel=False)
        
        with torch.no_grad():
            self.weight_quant.init_scale_from_weights(conv.weight)
        
        self.conv_params = {
            'stride': conv.stride,
            'padding': conv.padding,
            'dilation': conv.dilation,
            'groups': conv.groups,
        }
        
        self.weight = conv.weight
        self.bias = conv.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quant(self.conv.weight)
        
        return F.conv2d(
            x, q_weight, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
    
    def calibrate_rounding(
        self,
        input_activations: List[torch.Tensor],
        target_outputs: List[torch.Tensor],
        optimizer: AdaRoundOptimizer,
        activation_fn: Optional[Callable] = None,
        log_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ):
        rounding = optimizer.optimize_conv2d_layer(
            weight=self.conv.weight,
            scale=self.weight_quant.s,
            thd_neg=self.weight_quant.thd_neg,
            thd_pos=self.weight_quant.thd_pos,
            input_activations=input_activations,
            target_outputs=target_outputs,
            conv_params=self.conv_params,
            activation_fn=activation_fn,
            log_callback=log_callback,
        )
        
        self.weight_quant.rounding = rounding
        self.weight_quant.is_calibrated = True


class AdaRoundLinear(nn.Module):
    """Linear with AdaRound PTQ"""
    
    def __init__(self, linear: nn.Linear, bit_width: int):
        super().__init__()
        self.fc = linear
        self.weight_quant = AdaRoundQuantizer(bit_width=bit_width, per_channel=False)
        
        with torch.no_grad():
            self.weight_quant.init_scale_from_weights(linear.weight)
        
        self.weight = linear.weight
        self.bias = linear.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quant(self.fc.weight)
        return F.linear(x, q_weight, self.fc.bias)
    
    def calibrate_rounding(
        self,
        input_activations: List[torch.Tensor],
        target_outputs: List[torch.Tensor],
        optimizer: AdaRoundOptimizer,
        activation_fn: Optional[Callable] = None,
        log_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ):
        rounding = optimizer.optimize_linear_layer(
            weight=self.fc.weight,
            scale=self.weight_quant.s,
            thd_neg=self.weight_quant.thd_neg,
            thd_pos=self.weight_quant.thd_pos,
            input_activations=input_activations,
            target_outputs=target_outputs,
            activation_fn=activation_fn,
            log_callback=log_callback,
        )
        
        self.weight_quant.rounding = rounding
        self.weight_quant.is_calibrated = True


class AdaRoundEmbedding(nn.Module):
    """Embedding with AdaRound PTQ"""
    
    def __init__(self, emb: nn.Embedding, bit_width: int):
        super().__init__()
        self.emb = emb
        self.weight_quant = AdaRoundQuantizer(bit_width=bit_width)
        
        with torch.no_grad():
            self.weight_quant.init_scale_from_weights(emb.weight)
        
        self.weight = emb.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quant(self.emb.weight)
        return F.embedding(
            x, q_weight,
            self.emb.padding_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )
    
    def calibrate_rounding(
        self,
        input_ids: List[torch.Tensor],
        optimizer: AdaRoundOptimizer,
        log_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ):
        rounding = optimizer.optimize_embedding_layer(
            weight=self.emb.weight,
            scale=self.weight_quant.s,
            thd_neg=self.weight_quant.thd_neg,
            thd_pos=self.weight_quant.thd_pos,
            input_ids=input_ids,
            log_callback=log_callback,
        )
        
        self.weight_quant.rounding = rounding
        self.weight_quant.is_calibrated = True


class AdaRoundLayerNorm(nn.Module):
    """LayerNorm - kept in FP32"""
    
    def __init__(self, ln: nn.LayerNorm, bit_width: int):
        super().__init__()
        self.ln = ln
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps
        self.normalized_shape = ln.normalized_shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.ln.normalized_shape,
                          self.ln.weight, self.ln.bias, self.ln.eps)


class AdaRoundQuantizerWrapper(BaseQuantizerWrapper):
    """AdaRound Post-Training Quantization Wrapper"""
    
    def __init__(self, quantizer: AdaRoundQuantizer, bit_width: int = 4, logging_backend: str = "none"):
        super().__init__(quantizer, logging_backend=logging_backend)
        self.bit_width = bit_width
        self._layer_sequence = []
        self.logging_backend = logging_backend
    
    def prepare_qat_model(self, model: nn.Module) -> nn.Module:
        return model

    def prepare_ptq_model(self, model: nn.Module) -> nn.Module:
        self._layer_sequence = []
        self._prepare_module(model, prefix='')
        return model
    
    def _prepare_module(self, module: nn.Module, prefix: str = ''):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                new_layer = AdaRoundLinear(child, bit_width=self.bit_width)
                setattr(module, name, new_layer)
                self._layer_sequence.append((full_name, new_layer))
                
            elif isinstance(child, nn.Conv2d):
                new_layer = AdaRoundConv2d(child, bit_width=self.bit_width)
                setattr(module, name, new_layer)
                self._layer_sequence.append((full_name, new_layer))
                
            elif isinstance(child, nn.Embedding):
                new_layer = AdaRoundEmbedding(child, bit_width=self.bit_width)
                setattr(module, name, new_layer)
                self._layer_sequence.append((full_name, new_layer))
                
            elif isinstance(child, nn.LayerNorm):
                new_layer = AdaRoundLayerNorm(child, bit_width=self.bit_width)
                setattr(module, name, new_layer)
                
            else:
                self._prepare_module(child, full_name)
    
    def _get_activation_fn(self, model: nn.Module, current_layer_name: str) -> Optional[Callable]:
        """Detect activation function after current layer"""
        activation_map = {
            'ReLU': F.relu,
            'GELU': F.gelu,
            'Sigmoid': torch.sigmoid,
            'Tanh': torch.tanh,
        }
        
        found_current = False
        for name, module in model.named_modules():
            if found_current:
                module_type = type(module).__name__
                if module_type in activation_map:
                    return activation_map[module_type]
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    return None
            
            if name == current_layer_name:
                found_current = True
        
        return None
    
    def _collect_single_layer_io(
        self,
        model: nn.Module,
        layer_module: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int,
        device,
        batch_to_model_inputs: Optional[Callable[[Any, torch.device], Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect input/output activations for a single specific layer"""
        device_obj = device if isinstance(device, torch.device) else torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        layer_inputs = []
        layer_outputs = []
        
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            
            if len(layer_inputs) * inp.size(0) < max_samples:
                layer_inputs.append(inp.detach().cpu())
                layer_outputs.append(output.detach().cpu())
        
        # Register hook only for this specific layer
        hook = layer_module.register_forward_hook(hook_fn)
        
        samples_seen = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_to_model_inputs is not None:
                    inputs = batch_to_model_inputs(batch, device_obj)
                else:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device_obj)
                    else:
                        inputs = batch.to(device_obj)

                if isinstance(inputs, (list, tuple)):
                    model(*inputs)
                    batch_size = inputs[0].size(0)
                else:
                    model(inputs)
                    batch_size = inputs.size(0)

                samples_seen += batch_size
                if samples_seen >= max_samples:
                    break
        
        hook.remove()
        
        return layer_inputs, layer_outputs
    
    def calibrate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_iterations: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-3,
        lambda_reg: float = 0.01,
        max_samples: int = 1024,
        device: str = 'cuda',
        batch_to_model_inputs: Optional[Callable[[Any, torch.device], Any]] = None,
        metrics_eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
        mlflow_client=None,
    ) -> nn.Module:
        """Calibrate AdaRound layer-by-layer
        
        Key insight: After calibrating layer i, we collect activations for layer i+1
        through the partially quantized network (layers 0...i quantized, i+1...N in FP32)
        """
        device_obj = device if isinstance(device, torch.device) else torch.device(device)

        optimizer = AdaRoundOptimizer(
            num_iterations=num_iterations,
            batch_size=batch_size,
            lr=lr,
            lambda_reg=lambda_reg,
        )
        layer_type_counters = {"linear": 0, "conv2d": 0, "embedding": 0, "other": 0}
        best_per_layer_metrics: Dict[str, float] = {}

        layer_pbar = tqdm(self._layer_sequence, desc="adaround_model_layers", leave=True)
        for idx, (layer_name, layer_module) in enumerate(layer_pbar):
            layer_inputs, layer_outputs = self._collect_single_layer_io(
                model=model,
                layer_module=layer_module,
                dataloader=dataloader,
                max_samples=max_samples,
                device=device_obj,
                batch_to_model_inputs=batch_to_model_inputs,
            )
            
            if len(layer_inputs) == 0:
                continue
            
            activation_fn = self._get_activation_fn(model, layer_name)

            layer_pbar.set_postfix_str(layer_name)

            log_callback = None
            if mlflow_client is not None and getattr(self, "logging_backend", "none") == "mlflow":
                if isinstance(layer_module, AdaRoundLinear):
                    layer_type = "linear"
                elif isinstance(layer_module, AdaRoundConv2d):
                    layer_type = "conv2d"
                elif isinstance(layer_module, AdaRoundEmbedding):
                    layer_type = "embedding"
                else:
                    layer_type = "other"

                layer_idx_by_type = layer_type_counters[layer_type]
                layer_type_counters[layer_type] += 1
                layer_id = str(layer_idx_by_type)
                last_metrics = {"loss": None, "recon_loss": None, "reg_loss": None}

                def _log_cb(
                    iter_idx: int,
                    recon: float,
                    reg: float,
                    total: float,
                    lname: str = layer_name,
                    ltype: str = layer_type,
                    lid: str = layer_id,
                    lm: dict = last_metrics,
                ):
                    lm["loss"] = total
                    lm["recon_loss"] = recon
                    lm["reg_loss"] = reg

                    metrics = {
                        f"adaround_{ltype}-{lid}_loss": total,
                        f"adaround_{ltype}-{lid}_recon_loss": recon,
                        f"adaround_{ltype}-{lid}_reg_loss": reg,
                    }
                    step = int(iter_idx) + 1
                    for mk, mv in metrics.items():
                        mlflow_client.log_metric(mk, float(mv), step=step)
                log_callback = _log_cb
            
            if isinstance(layer_module, (AdaRoundConv2d, AdaRoundLinear)):
                layer_module.calibrate_rounding(
                    input_activations=layer_inputs,
                    target_outputs=layer_outputs,
                    optimizer=optimizer,
                    activation_fn=activation_fn,
                    log_callback=log_callback,
                )
            elif isinstance(layer_module, AdaRoundEmbedding):
                input_ids = [act.long() for act in layer_inputs]
                layer_module.calibrate_rounding(input_ids, optimizer, log_callback=log_callback)
            if mlflow_client is not None and getattr(self, "logging_backend", "none") == "mlflow":
                if "last_metrics" in locals() and last_metrics["loss"] is not None:
                    for mk, mv in last_metrics.items():
                        cur_val = float(mv)
                        mlflow_client.log_metric(
                            f"adaround_per-layer_{mk}", cur_val, step=idx + 1
                        )
                        prev_best = best_per_layer_metrics.get(mk, float("-inf"))
                        if cur_val > prev_best:
                            best_per_layer_metrics[mk] = cur_val
                        mlflow_client.log_metric(
                            f"max_adaround_per-layer_{mk}",
                            best_per_layer_metrics[mk],
                            step=idx + 1,
                        )
                if metrics_eval_fn is not None:
                    try:
                        metric_values = metrics_eval_fn(model)
                    except Exception:
                        metric_values = {}
                    for mname, mval in metric_values.items():
                        cur_val = float(mval)
                        mlflow_client.log_metric(
                            f"adaround_per-layer_{mname}", cur_val, step=idx + 1
                        )
                        prev_best = best_per_layer_metrics.get(mname, float("-inf"))
                        if cur_val > prev_best:
                            best_per_layer_metrics[mname] = cur_val
                        mlflow_client.log_metric(
                            f"max_adaround_per-layer_{mname}",
                            best_per_layer_metrics[mname],
                            step=idx + 1,
                        )
        
        model.eval()
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def optimize_ptq(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device,
        **kwargs,
    ) -> nn.Module:
        return self.calibrate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            **kwargs,
        )
