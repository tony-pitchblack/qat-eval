"""
ONNX INT8 Converter for Quantized Models
Supports: QIL, AdaRound, APoT, LSQ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantType, QuantizationMode
import numpy as np
import os
from typing import Optional, Dict, Any, Tuple


class ONNXCalibrationDataReader(CalibrationDataReader):
    """Universal calibration data reader for ONNX quantization"""
    
    def __init__(self, dataloader, input_name='input', max_samples=100):
        self.dataloader = dataloader
        self.input_name = input_name
        self.max_samples = max_samples
        self.iterator = iter(dataloader)
        self.samples_processed = 0
    
    def get_next(self):
        if self.samples_processed >= self.max_samples:
            return None
        
        try:
            batch = next(self.iterator)
            self.samples_processed += 1
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            
            # Convert to numpy
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            
            return {self.input_name: data}
        except StopIteration:
            return None
    
    def rewind(self):
        self.iterator = iter(self.dataloader)
        self.samples_processed = 0


class ONNXConverter:
    """Base class for ONNX conversion"""
    
    def __init__(self, model_name: str, quantizer_name: str):
        self.model_name = model_name
        self.quantizer_name = quantizer_name
    
    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Convert quantized layers to standard layers with quantized weights"""
        raise NotImplementedError
    
    def export_to_onnx(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        opset_version: int = 13
    ) -> str:
        """Export model to ONNX format"""
        model = self.prepare_model_for_export(model)
        model.eval()
        
        # Dynamic axes for LSTM models
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
        # Use legacy exporter (dynamo=False) for better compatibility with LSTM/RNN
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            dynamo=False  # Use legacy exporter, not torch.export
        )
        
        print(f"[{self.quantizer_name}] Model exported to {output_path}")
        return output_path
    
    def quantize_to_int8(
        self,
        onnx_fp32_path: str,
        output_path: str,
        calibration_loader: Optional[Any] = None,
        quantization_type: str = 'dynamic'
    ) -> str:
        """Quantize ONNX model to INT8"""
        
        if quantization_type == 'dynamic':
            return self._quantize_dynamic(onnx_fp32_path, output_path)
        elif quantization_type == 'static':
            if calibration_loader is None:
                raise ValueError("calibration_loader required for static quantization")
            return self._quantize_static(onnx_fp32_path, output_path, calibration_loader)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    def _quantize_dynamic(self, model_path: str, output_path: str) -> str:
        """Dynamic quantization - simple, no calibration needed"""
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )
        print(f"[{self.quantizer_name}] Dynamic INT8 model saved to {output_path}")
        return output_path
    
    def _quantize_static(
        self,
        model_path: str,
        output_path: str,
        calibration_loader: Any
    ) -> str:
        """Static quantization - best quality, requires calibration"""
        calibration_reader = ONNXCalibrationDataReader(calibration_loader, max_samples=100)
        
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8
        )
        print(f"[{self.quantizer_name}] Static INT8 model saved to {output_path}")
        return output_path
    
    def convert_to_onnx_int8(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_dir: str,
        calibration_loader: Optional[Any] = None,
        quantization_type: str = 'dynamic'
    ) -> Tuple[str, str]:
        """Full pipeline: PyTorch -> ONNX FP32 -> ONNX INT8"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Export to ONNX FP32
        fp32_path = os.path.join(output_dir, f'{self.model_name}_{self.quantizer_name}_fp32.onnx')
        self.export_to_onnx(model, dummy_input, fp32_path)
        
        # Step 2: Quantize to INT8
        int8_path = os.path.join(output_dir, f'{self.model_name}_{self.quantizer_name}_int8.onnx')
        self.quantize_to_int8(fp32_path, int8_path, calibration_loader, quantization_type)
        
        # Step 3: Validate
        self._validate_onnx(int8_path)
        
        return fp32_path, int8_path
    
    def _validate_onnx(self, model_path: str):
        """Validate ONNX model"""
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            print(f"[{self.quantizer_name}] ONNX model is valid")
        except Exception as e:
            print(f"[{self.quantizer_name}] Warning: ONNX validation failed: {e}")


class QILONNXConverter(ONNXConverter):
    """ONNX converter for QIL quantized models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name, 'QIL')
    
    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Convert QIL layers to standard layers with quantized weights"""
        from quantizers.qil import QILConv2d, QILLinear, QILEmbedding, QILLayerNorm
        
        def convert_module(module):
            for name, child in list(module.named_children()):
                if isinstance(child, QILConv2d):
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.conv.weight)
                        q_bias = None
                        if child.conv.bias is not None:
                            q_bias = child.bias_quant(child.conv.bias)
                    
                    new_layer = nn.Conv2d(
                        in_channels=child.conv.in_channels,
                        out_channels=child.conv.out_channels,
                        kernel_size=child.conv.kernel_size,
                        stride=child.conv.stride,
                        padding=child.conv.padding,
                        dilation=child.conv.dilation,
                        groups=child.conv.groups,
                        bias=child.conv.bias is not None
                    )
                    new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, QILLinear):
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.fc.weight)
                        q_bias = None
                        if child.fc.bias is not None:
                            q_bias = child.bias_quant(child.fc.bias)
                    
                    new_layer = nn.Linear(
                        in_features=child.fc.in_features,
                        out_features=child.fc.out_features,
                        bias=child.fc.bias is not None
                    )
                    new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, QILEmbedding):
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.emb.weight)
                    
                    new_layer = nn.Embedding(
                        num_embeddings=child.emb.num_embeddings,
                        embedding_dim=child.emb.embedding_dim,
                        padding_idx=child.emb.padding_idx
                    )
                    new_layer.weight.data = q_weight
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, QILLayerNorm):
                    new_layer = nn.LayerNorm(
                        normalized_shape=child.ln.normalized_shape,
                        eps=child.ln.eps
                    )
                    new_layer.weight.data = child.ln.weight.data
                    if child.ln.bias is not None:
                        new_layer.bias.data = child.ln.bias.data
                    setattr(module, name, new_layer)
                else:
                    convert_module(child)
        
        convert_module(model)
        return model


class AdaRoundONNXConverter(ONNXConverter):
    """ONNX converter for AdaRound quantized models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name, 'AdaRound')
        self.quantization_params = {}
    
    def export_adaround_weights_to_int8(self, model: nn.Module) -> Dict[str, Any]:
        """Extract quantized weights and parameters from AdaRound model"""
        from quantizers.adaround import AdaRoundConv2d, AdaRoundLinear, AdaRoundEmbedding
        
        int8_state_dict = {}
        quant_params = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (AdaRoundConv2d, AdaRoundLinear)):
                with torch.no_grad():
                    if isinstance(module, AdaRoundConv2d):
                        weight_fp32 = module.conv.weight
                    else:
                        weight_fp32 = module.fc.weight
                    
                    # Apply AdaRound quantization with rounding
                    scale = module.weight_quant.s
                    w_scaled = weight_fp32 / scale
                    w_floor = torch.floor(w_scaled)
                    
                    # Use optimized rounding if calibrated
                    if module.weight_quant.rounding is not None:
                        w_quant = w_floor + module.weight_quant.rounding
                    else:
                        w_quant = torch.round(w_scaled)
                    
                    # Clipping
                    w_quant = torch.clamp(
                        w_quant,
                        module.weight_quant.thd_neg,
                        module.weight_quant.thd_pos
                    )
                    
                    # Save quantization parameters
                    quant_params[f"{name}.weight"] = {
                        'scale': scale.cpu().numpy(),
                        'zero_point': 0,  # Symmetric quantization
                        'dtype': 'int8',
                        'symmetric': True
                    }
                    
                    # Store quantized weights for export
                    int8_state_dict[f"{name}.weight"] = w_quant
        
        self.quantization_params = quant_params
        return int8_state_dict
    
    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Convert AdaRound layers preserving quantization info"""
        from quantizers.adaround import AdaRoundConv2d, AdaRoundLinear, AdaRoundEmbedding, AdaRoundLayerNorm
        
        # Extract INT8 weights first
        int8_weights = self.export_adaround_weights_to_int8(model)
        
        def convert_module(module):
            for name, child in list(module.named_children()):
                if isinstance(child, AdaRoundConv2d):
                    with torch.no_grad():
                        q_weight = child.weight_quant.get_quantized_weight(child.conv.weight)
                        q_bias = child.conv.bias
                    
                    new_layer = nn.Conv2d(
                        in_channels=child.conv.in_channels,
                        out_channels=child.conv.out_channels,
                        kernel_size=child.conv.kernel_size,
                        stride=child.conv.stride,
                        padding=child.conv.padding,
                        dilation=child.conv.dilation,
                        groups=child.conv.groups,
                        bias=child.conv.bias is not None
                    )
                    new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, AdaRoundLinear):
                    with torch.no_grad():
                        q_weight = child.weight_quant.get_quantized_weight(child.fc.weight)
                        q_bias = child.fc.bias
                    
                    new_layer = nn.Linear(
                        in_features=child.fc.in_features,
                        out_features=child.fc.out_features,
                        bias=child.fc.bias is not None
                    )
                    new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, AdaRoundEmbedding):
                    with torch.no_grad():
                        q_weight = child.weight_quant.get_quantized_weight(child.emb.weight)
                    
                    new_layer = nn.Embedding(
                        num_embeddings=child.emb.num_embeddings,
                        embedding_dim=child.emb.embedding_dim,
                        padding_idx=child.emb.padding_idx
                    )
                    new_layer.weight.data = q_weight
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, AdaRoundLayerNorm):
                    new_layer = nn.LayerNorm(
                        normalized_shape=child.ln.normalized_shape,
                        eps=child.ln.eps
                    )
                    new_layer.weight.data = child.ln.weight.data
                    if child.ln.bias is not None:
                        new_layer.bias.data = child.ln.bias.data
                    setattr(module, name, new_layer)
                else:
                    convert_module(child)
        
        convert_module(model)
        return model
    
    def _quantize_static(
        self,
        model_path: str,
        output_path: str,
        calibration_loader: Any
    ) -> str:
        """Static quantization with AdaRound-specific parameters"""
        calibration_reader = ONNXCalibrationDataReader(calibration_loader, max_samples=100)
        
        # Use symmetric quantization for weights (AdaRound uses symmetric)
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            extra_options={
                'WeightSymmetric': True,  # AdaRound uses symmetric quantization
                'ActivationSymmetric': False,
            }
        )
        print(f"[{self.quantizer_name}] Static INT8 model saved to {output_path}")
        return output_path


class APoTONNXConverter(ONNXConverter):
    """ONNX converter for APoT quantized models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name, 'APoT')
    
    def quantize_to_indices(self, weight: torch.Tensor, levels: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
        """Quantize weights to indices in levels array for lookup table"""
        levels_normalized = levels * gamma
        weight_normalized = weight.cpu().numpy() / alpha
        weight_clipped = np.clip(weight_normalized, -1.0, 1.0)
        
        # Find nearest level index
        indices = np.zeros(weight_clipped.shape, dtype=np.int64)
        weight_flat = weight_clipped.flatten()
        
        for i, w in enumerate(weight_flat):
            distances = np.abs(levels_normalized - w)
            indices.flat[i] = np.argmin(distances)
        
        return indices
    
    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Convert APoT layers to lookup table representation for ONNX"""
        from quantizers.apot import APoTConv2d, APoTLinear, APoTEmbedding, APoTLayerNorm
        
        class APoTLinearONNX(nn.Module):
            """ONNX-compatible APoT Linear using Gather operation"""
            
            def __init__(self, indices, levels, alpha, gamma, bias, in_features, out_features):
                super().__init__()
                
                # Register as buffers for ONNX export
                self.register_buffer('indices', torch.from_numpy(indices).long())
                self.register_buffer('levels', torch.from_numpy(levels).float())
                self.register_buffer('alpha', torch.tensor([alpha]).float())
                self.register_buffer('gamma', torch.tensor([gamma]).float())
                
                if bias is not None:
                    self.register_buffer('bias', bias)
                else:
                    self.bias = None
                
                self.in_features = in_features
                self.out_features = out_features
            
            def forward(self, x):
                # Reconstruct weights using Gather (ONNX-efficient)
                levels_normalized = self.levels * self.gamma
                
                # Gather operation - supported natively by ONNX
                flat_indices = self.indices.flatten()
                weight_flat = torch.gather(
                    levels_normalized.unsqueeze(0).expand(len(flat_indices), -1),
                    1,
                    flat_indices.unsqueeze(1)
                ).squeeze(1)
                
                weight = weight_flat.reshape(self.indices.shape) * self.alpha
                
                return F.linear(x, weight, self.bias)
        
        class APoTConv2dONNX(nn.Module):
            """ONNX-compatible APoT Conv2d using Gather operation"""
            
            def __init__(self, indices, levels, alpha, gamma, bias, conv_params):
                super().__init__()
                
                self.register_buffer('indices', torch.from_numpy(indices).long())
                self.register_buffer('levels', torch.from_numpy(levels).float())
                self.register_buffer('alpha', torch.tensor([alpha]).float())
                self.register_buffer('gamma', torch.tensor([gamma]).float())
                
                if bias is not None:
                    self.register_buffer('bias', bias)
                else:
                    self.bias = None
                
                self.conv_params = conv_params
            
            def forward(self, x):
                # Reconstruct weights using Gather
                levels_normalized = self.levels * self.gamma
                flat_indices = self.indices.flatten()
                weight_flat = torch.gather(
                    levels_normalized.unsqueeze(0).expand(len(flat_indices), -1),
                    1,
                    flat_indices.unsqueeze(1)
                ).squeeze(1)
                
                weight = weight_flat.reshape(self.indices.shape) * self.alpha
                
                return F.conv2d(
                    x, weight, self.bias,
                    stride=self.conv_params['stride'],
                    padding=self.conv_params['padding'],
                    dilation=self.conv_params['dilation'],
                    groups=self.conv_params['groups']
                )
        
        def convert_module(module):
            for name, child in list(module.named_children()):
                if isinstance(child, APoTLinear):
                    # Extract APoT parameters
                    quantizer = child.weight_quant
                    weight = child.fc.weight.data
                    
                    # Quantize to indices
                    indices = self.quantize_to_indices(
                        weight,
                        quantizer.levels.cpu().numpy() if hasattr(quantizer, 'levels') and quantizer.levels is not None else np.linspace(-1, 1, 256),
                        quantizer.alpha.item() if hasattr(quantizer, 'alpha') else 1.0,
                        quantizer.gamma if hasattr(quantizer, 'gamma') else 1.0
                    )
                    
                    # Create ONNX-compatible layer with lookup table
                    new_layer = APoTLinearONNX(
                        indices=indices,
                        levels=quantizer.levels.cpu().numpy() if hasattr(quantizer, 'levels') and quantizer.levels is not None else np.linspace(-1, 1, 256),
                        alpha=quantizer.alpha.item() if hasattr(quantizer, 'alpha') else 1.0,
                        gamma=quantizer.gamma if hasattr(quantizer, 'gamma') else 1.0,
                        bias=child.fc.bias.data if child.fc.bias is not None else None,
                        in_features=child.fc.in_features,
                        out_features=child.fc.out_features
                    )
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, APoTConv2d):
                    # Extract APoT parameters
                    quantizer = child.weight_quant
                    weight = child.conv.weight.data
                    
                    # Quantize to indices
                    indices = self.quantize_to_indices(
                        weight,
                        quantizer.levels.cpu().numpy() if hasattr(quantizer, 'levels') and quantizer.levels is not None else np.linspace(-1, 1, 256),
                        quantizer.alpha.item() if hasattr(quantizer, 'alpha') else 1.0,
                        quantizer.gamma if hasattr(quantizer, 'gamma') else 1.0
                    )
                    
                    # Create ONNX-compatible layer
                    conv_params = {
                        'stride': child.conv.stride,
                        'padding': child.conv.padding,
                        'dilation': child.conv.dilation,
                        'groups': child.conv.groups
                    }
                    
                    new_layer = APoTConv2dONNX(
                        indices=indices,
                        levels=quantizer.levels.cpu().numpy() if hasattr(quantizer, 'levels') and quantizer.levels is not None else np.linspace(-1, 1, 256),
                        alpha=quantizer.alpha.item() if hasattr(quantizer, 'alpha') else 1.0,
                        gamma=quantizer.gamma if hasattr(quantizer, 'gamma') else 1.0,
                        bias=child.conv.bias.data if child.conv.bias is not None else None,
                        conv_params=conv_params
                    )
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, APoTEmbedding):
                    # Embeddings: use quantized weights directly
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.emb.weight, is_weight=True)
                    
                    new_layer = nn.Embedding(
                        num_embeddings=child.emb.num_embeddings,
                        embedding_dim=child.emb.embedding_dim,
                        padding_idx=child.emb.padding_idx
                    )
                    new_layer.weight.data = q_weight
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, APoTLayerNorm):
                    # LayerNorm: keep in FP32
                    new_layer = nn.LayerNorm(
                        normalized_shape=child.ln.normalized_shape,
                        eps=child.ln.eps
                    )
                    new_layer.weight.data = child.ln.weight.data
                    if child.ln.bias is not None:
                        new_layer.bias.data = child.ln.bias.data
                    setattr(module, name, new_layer)
                else:
                    convert_module(child)
        
        convert_module(model)
        return model


class LSQONNXConverter(ONNXConverter):
    """ONNX converter for LSQ quantized models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name, 'LSQ')
    
    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Convert LSQ layers to standard layers with quantized weights"""
        from quantizers.lsq import LSQConv2d, LSQLinear, LSQEmbedding, LSQLayerNorm
        
        def convert_module(module):
            for name, child in list(module.named_children()):
                if isinstance(child, LSQConv2d):
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.conv.weight)
                        q_bias = None
                        if child.conv.bias is not None:
                            q_bias = child.bias_quant(child.conv.bias)
                    
                    new_layer = nn.Conv2d(
                        in_channels=child.conv.in_channels,
                        out_channels=child.conv.out_channels,
                        kernel_size=child.conv.kernel_size,
                        stride=child.conv.stride,
                        padding=child.conv.padding,
                        dilation=child.conv.dilation,
                        groups=child.conv.groups,
                        bias=child.conv.bias is not None
                    )
                    new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, LSQLinear):
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.fc.weight)
                        q_bias = None
                        if child.fc.bias is not None:
                            q_bias = child.bias_quant(child.fc.bias)
                    
                    new_layer = nn.Linear(
                        in_features=child.fc.in_features,
                        out_features=child.fc.out_features,
                        bias=child.fc.bias is not None
                    )
                    new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, LSQEmbedding):
                    with torch.no_grad():
                        q_weight = child.weight_quant(child.emb.weight)
                    
                    new_layer = nn.Embedding(
                        num_embeddings=child.emb.num_embeddings,
                        embedding_dim=child.emb.embedding_dim,
                        padding_idx=child.emb.padding_idx
                    )
                    new_layer.weight.data = q_weight
                    setattr(module, name, new_layer)
                    
                elif isinstance(child, LSQLayerNorm):
                    with torch.no_grad():
                        q_weight = child.ln.weight
                        q_bias = child.ln.bias
                        if q_weight is not None:
                            q_weight = child.weight_quant(q_weight)
                        if q_bias is not None:
                            q_bias = child.bias_quant(q_bias)
                    
                    new_layer = nn.LayerNorm(
                        normalized_shape=child.ln.normalized_shape,
                        eps=child.ln.eps
                    )
                    if q_weight is not None:
                        new_layer.weight.data = q_weight
                    if q_bias is not None:
                        new_layer.bias.data = q_bias
                    setattr(module, name, new_layer)
                else:
                    convert_module(child)
        
        convert_module(model)
        return model


def get_onnx_converter(model_name: str, quantizer_name: str) -> Optional[ONNXConverter]:
    """Factory function to get appropriate ONNX converter"""
    converters = {
        'qil': QILONNXConverter,
        'adaround': AdaRoundONNXConverter,
        'apot': APoTONNXConverter,
        'lsq': LSQONNXConverter,
    }
    
    converter_class = converters.get(quantizer_name.lower())
    if converter_class is None:
        return None
    
    return converter_class(model_name)


def create_dummy_input(model_name: str, model_cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Create dummy input for ONNX export based on model type"""
    if model_name == "lstm":
        batch_size = 1
        seq_len = model_cfg.get('max_seq_len', 512)
        return torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    elif model_name == "simple_cnn":
        return torch.randn(1, 1, 28, 28, device=device)
    
    elif model_name == "sasrec":
        batch_size = 1
        seq_len = model_cfg.get('max_seq_len', 10)
        return torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

