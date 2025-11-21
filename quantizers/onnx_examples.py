import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantType
import numpy as np


# Шаг 1: Подготовка модели для экспорта
def prepare_qil_model_for_onnx(model):
    """
    Заменяет QIL слои на обычные с квантованными весами.
    ONNX не понимает custom quantizers, поэтому квантуем веса заранее.
    """
    model.eval()
    
    for name, module in list(model.named_children()):
        if isinstance(module, (QILConv2d, QILLinear)):
            with torch.no_grad():
                # Квантуем веса
                if hasattr(module, 'conv'):
                    original = module.conv
                    q_weight = module.weight_quant(original.weight)
                    q_bias = module.bias_quant(original.bias) if original.bias is not None else None
                    
                    new_conv = nn.Conv2d(
                        original.in_channels,
                        original.out_channels,
                        original.kernel_size,
                        original.stride,
                        original.padding,
                        original.dilation,
                        original.groups,
                        bias=original.bias is not None
                    )
                    new_conv.weight.data = q_weight
                    if q_bias is not None:
                        new_conv.bias.data = q_bias
                    
                    setattr(model, name, new_conv)
                    
                else:  # Linear
                    original = module.fc
                    q_weight = module.weight_quant(original.weight)
                    q_bias = module.bias_quant(original.bias) if original.bias is not None else None
                    
                    new_linear = nn.Linear(
                        original.in_features,
                        original.out_features,
                        bias=original.bias is not None
                    )
                    new_linear.weight.data = q_weight
                    if q_bias is not None:
                        new_linear.bias.data = q_bias
                    
                    setattr(model, name, new_linear)
        else:
            prepare_qil_model_for_onnx(module)
    
    return model


# Шаг 2: Экспорт в ONNX
def export_to_onnx(model, dummy_input, output_path="model.onnx"):
    """Экспорт PyTorch модели в ONNX."""
    model = prepare_qil_model_for_onnx(model)
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # Минимум 10 для quantization
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    return output_path


# Шаг 3: Квантизация ONNX модели

# Вариант 1: Dynamic Quantization (простой, без калибровки)
def quantize_onnx_dynamic(model_path, output_path="model_int8.onnx"):
    """
    Динамическая квантизация - веса в int8, активации квантуются динамически.
    Подходит для CPU inference.
    """
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,  # Веса в int8
        per_channel=False,
        reduce_range=False,
        optimize_model=True,
    )
    print(f"Dynamic quantized model saved to {output_path}")
    return output_path


# Вариант 2: Static Quantization (лучшее качество, нужна калибровка)
class QILCalibrationDataReader(CalibrationDataReader):
    """DataReader для калибровки ONNX модели."""
    
    def __init__(self, calibration_dataset, input_name='input'):
        self.data = calibration_dataset
        self.input_name = input_name
        self.datasize = len(calibration_dataset)
        self.enum_data = iter(calibration_dataset)
    
    def get_next(self):
        try:
            batch = next(self.enum_data)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Если есть labels
            return {self.input_name: batch.cpu().numpy()}
        except StopIteration:
            return None
    
    def rewind(self):
        self.enum_data = iter(self.data)


def quantize_onnx_static(
    model_path, 
    calibration_dataset,
    output_path="model_int8_static.onnx"
):
    """
    Статическая квантизация - веса и активации в int8.
    Требует калибровочные данные.
    """
    calibration_reader = QILCalibrationDataReader(calibration_dataset)
    
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    print(f"Static quantized model saved to {output_path}")
    return output_path


# Вариант 3: QDQ (Quantize-Dequantize) формат
def quantize_onnx_qdq(
    model_path,
    calibration_dataset, 
    output_path="model_int8_qdq.onnx"
):
    """
    QDQ формат - совместим с TensorRT и другими runtime.
    Рекомендуется для GPU inference.
    """
    from onnxruntime.quantization import QuantFormat
    
    calibration_reader = QILCalibrationDataReader(calibration_dataset)
    
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,  # QDQ формат
        per_channel=True,  # Per-channel для лучшего качества
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    print(f"QDQ quantized model saved to {output_path}")
    return output_path


# Шаг 4: Inference с ONNX Runtime
def inference_onnx(model_path, input_data):
    """Запуск inference с квантованной ONNX моделью."""
    
    # Создаем сессию
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']  # Или 'CUDAExecutionProvider'
    )
    
    # Получаем имена входов/выходов
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Inference
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.cpu().numpy()
    
    outputs = session.run(
        [output_name],
        {input_name: input_data}
    )
    
    return outputs[0]


# Полный пайплайн
def qil_to_onnx_int8_pipeline(
    qil_model,
    dummy_input,
    calibration_dataset=None,
    quantization_type='static'  # 'dynamic', 'static', 'qdq'
):
    """
    Полный пайплайн: QIL → ONNX → Int8 ONNX
    
    Args:
        qil_model: обученная QIL модель
        dummy_input: пример входа для экспорта
        calibration_dataset: датасет для калибровки (для static/qdq)
        quantization_type: тип квантизации
    """
    
    # 1. Экспорт в ONNX (float32)
    print("Step 1: Exporting to ONNX...")
    onnx_path = export_to_onnx(qil_model, dummy_input, "model_fp32.onnx")
    
    # 2. Квантизация
    print(f"Step 2: Quantizing with {quantization_type} method...")
    
    if quantization_type == 'dynamic':
        int8_path = quantize_onnx_dynamic(onnx_path)
    
    elif quantization_type == 'static':
        if calibration_dataset is None:
            raise ValueError("calibration_dataset required for static quantization")
        int8_path = quantize_onnx_static(onnx_path, calibration_dataset)
    
    elif quantization_type == 'qdq':
        if calibration_dataset is None:
            raise ValueError("calibration_dataset required for QDQ quantization")
        int8_path = quantize_onnx_qdq(onnx_path, calibration_dataset)
    
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    # 3. Проверка модели
    print("Step 3: Validating ONNX model...")
    onnx_model = onnx.load(int8_path)
    onnx.checker.check_model(onnx_model)
    print("Model is valid!")
    
    # 4. Тест inference
    print("Step 4: Testing inference...")
    if isinstance(dummy_input, torch.Tensor):
        test_input = dummy_input.cpu().numpy()
    else:
        test_input = dummy_input
    
    output = inference_onnx(int8_path, test_input)
    print(f"Output shape: {output.shape}")
    
    return int8_path


# Пример использования
if __name__ == "__main__":
    # Загружаем обученную QIL модель
    model = ...  # Ваша обученная QIL модель
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Калибровочный датасет (100-1000 сэмплов)
    calibration_data = []
    for i in range(100):
        calibration_data.append(torch.randn(1, 3, 224, 224))
    
    # Экспорт и квантизация
    int8_model_path = qil_to_onnx_int8_pipeline(
        qil_model=model,
        dummy_input=dummy_input,
        calibration_dataset=calibration_data,
        quantization_type='static'  # Рекомендуется для лучшего качества
    )
    
    # Inference
    test_input = torch.randn(1, 3, 224, 224)
    output = inference_onnx(int8_model_path, test_input)
    print(f"Final output: {output}")


def export_apot_with_lookup_to_onnx(model_apot, input_shape, output_path):
    """
    Export APoT model preserving non-uniform levels
    Uses Gather operator as lookup table
    """
    
    # Step 1: Конвертируем APoT в lookup table representation
    print("Step 1: Converting APoT to lookup representation...")
    model_lookup = convert_apot_to_lookup_representation(model_apot)
    
    # Step 2: Export to ONNX
    print("Step 2: Exporting to ONNX with custom ops...")
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model_lookup,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model with lookup tables exported to {output_path}")
    return output_path


def convert_apot_to_lookup_representation(model_apot):
    """
    Convert APoT layers to use ONNX-compatible lookup tables
    Uses Gather operator which is efficient in ONNX runtime
    """
    
    class APoTLinearONNX(nn.Module):
        """ONNX-compatible APoT Linear using Gather"""
        
        def __init__(self, indices, levels, alpha, gamma, bias, in_features, out_features):
            super().__init__()
            
            # Регистрируем как buffers для ONNX export
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
            # Reconstruct weights using Gather (эффективно в ONNX)
            levels_normalized = self.levels * self.gamma
            
            # Gather operation - ONNX поддерживает нативно
            flat_indices = self.indices.flatten()
            weight_flat = torch.gather(
                levels_normalized.unsqueeze(0).expand(len(flat_indices), -1),
                1,
                flat_indices.unsqueeze(1)
            ).squeeze(1)
            
            weight = weight_flat.reshape(self.indices.shape) * self.alpha
            
            return F.linear(x, weight, self.bias)
    
    model_lookup = nn.Module()
    
    # Заменяем APoT слои на ONNX-compatible версии
    for name, module in model_apot.named_modules():
        if isinstance(module, APoTLinear):
            # Извлекаем параметры
            quantizer = module.weight_quant
            weight = module.fc.weight.data
            
            # Квантуем к индексам
            indices = quantize_to_indices(
                weight,
                quantizer.levels.numpy(),
                quantizer.alpha.item(),
                quantizer.gamma
            )
            
            # Создаем ONNX-compatible слой
            new_module = APoTLinearONNX(
                indices=indices,
                levels=quantizer.levels.cpu().numpy(),
                alpha=quantizer.alpha.item(),
                gamma=quantizer.gamma,
                bias=module.fc.bias.data if module.fc.bias is not None else None,
                in_features=module.fc.in_features,
                out_features=module.fc.out_features
            )
            
            # Заменяем
            # ... (replace logic)
    
    return model_lookup


def quantize_to_indices(weight, levels, alpha, gamma):
    """Quantize weights to indices in levels array"""
    
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

import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class AdaRoundCalibrationReader(CalibrationDataReader):
    """Calibration reader с параметрами из AdaRound"""
    def __init__(self, model, dataloader, adaround_params):
        self.model = model
        self.dataloader = dataloader
        self.adaround_params = adaround_params
        self.iterator = iter(dataloader)
    
    def get_next(self):
        try:
            batch = next(self.iterator)
            if isinstance(batch, (list, tuple)):
                return {"input": batch[0].numpy()}
            return {"input": batch.numpy()}
        except StopIteration:
            return None

def convert_adaround_to_onnx_int8(
    model, 
    dummy_input, 
    calibration_loader,
    output_path="model_int8.onnx"
):
    """Конвертируем AdaRound в ONNX INT8"""
    
    # 1. Экспортируем параметры квантизации
    int8_weights, quant_params = export_adaround_weights_to_int8(model)
    
    # 2. Экспортируем в ONNX FP32
    torch.onnx.export(
        model,
        dummy_input,
        "model_fp32.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13
    )
    
    # 3. Создаем custom quantization config с параметрами AdaRound
    from onnxruntime.quantization import QuantizationMode, QuantType
    
    def create_adaround_quant_config():
        # Используем параметры из AdaRound
        return {
            'weight_type': QuantType.QInt8,
            'activation_type': QuantType.QUInt8,
            'per_channel': False,
            'reduce_range': False,
            'use_external_data_format': False,
            # Передаем scale/zero_point из AdaRound
            'extra_options': {
                'WeightSymmetric': True,  # AdaRound использует symmetric
                'ActivationSymmetric': False,
            }
        }
    
    # 4. Квантизация с использованием AdaRound параметров
    quantize_static(
        model_input="model_fp32.onnx",
        model_output=output_path,
        calibration_data_reader=AdaRoundCalibrationReader(
            model, calibration_loader, quant_params
        ),
        quant_format=QuantizationMode.IntegerOps,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        extra_options={
            'WeightSymmetric': True,
            # Инжектим наши scale параметры
            'CalibrationDataReader': quant_params
        }
    )
    
    print(f"INT8 model saved to {output_path}")
    return output_path



import torch.nn.quantized as nnq
import torch.quantization as tq

class Int8Conv2d(nn.Module):
    """INT8 Conv2d с параметрами из AdaRound"""
    def __init__(self, conv_module, scale, zero_point=0):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point
        
        # Создаем quantized conv
        self.qconv = nnq.Conv2d(
            conv_module.conv.in_channels,
            conv_module.conv.out_channels,
            conv_module.conv.kernel_size,
            conv_module.conv.stride,
            conv_module.conv.padding,
            conv_module.conv.dilation,
            conv_module.conv.groups,
        )
        
    def forward(self, x):
        # Quantize input
        x_q = torch.quantize_per_tensor(x, scale=self.scale, zero_point=0, dtype=torch.qint8)
        # INT8 computation
        out_q = self.qconv(x_q)
        # Dequantize output
        return out_q.dequantize()

def convert_adaround_to_pytorch_int8(model):
    """Конвертируем AdaRound модель в PyTorch Quantized API"""
    int8_weights, quant_params = export_adaround_weights_to_int8(model)
    
    # Создаем новую INT8 модель
    model_int8 = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if isinstance(module, AdaRoundConv2d):
            # Получаем параметры квантизации
            scale = quant_params[f"{name}.weight"]['scale']
            w_int8 = int8_weights[f"{name}.weight"]
            
            # Создаем quantized layer
            qconv = nnq.Conv2d(
                module.conv.in_channels,
                module.conv.out_channels,
                module.conv.kernel_size,
                stride=module.conv.stride,
                padding=module.conv.padding,
                dilation=module.conv.dilation,
                groups=module.conv.groups,
            )
            
            # Устанавливаем веса
            qconv.set_weight_bias(
                torch.quantize_per_tensor(
                    w_int8.float() * scale, 
                    scale=scale.item(), 
                    zero_point=0, 
                    dtype=torch.qint8
                ),
                module.conv.bias
            )
            
            # Заменяем модуль
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model_int8.named_modules())[parent_name]
            setattr(parent, child_name, qconv)
    
    return model_int8


def export_adaround_weights_to_int8(model):
    """Извлекаем квантованные веса из AdaRound модели"""
    int8_state_dict = {}
    quantization_params = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (AdaRoundConv2d, AdaRoundLinear)):
            # Получаем квантованные веса
            with torch.no_grad():
                weight_fp32 = module.weight
                
                # Применяем AdaRound квантизацию
                scale = module.weight_quant.s
                w_scaled = weight_fp32 / scale
                w_floor = torch.floor(w_scaled)
                
                # Используем найденные решения округления
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
                
                # Конвертируем в INT8
                w_int8 = w_quant.to(torch.int8)
                
                # Сохраняем
                int8_state_dict[f"{name}.weight"] = w_int8
                quantization_params[f"{name}.weight"] = {
                    'scale': scale.cpu().numpy(),
                    'zero_point': 0,  # Symmetric quantization
                    'dtype': 'int8'
                }
                
                # Bias остается в FP32
                if module.bias is not None:
                    int8_state_dict[f"{name}.bias"] = module.bias
    
    return int8_state_dict, quantization_params
