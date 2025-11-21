#!/usr/bin/env python3

import argparse
import os
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import warnings
import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Import models
from models.sasrec import SASRecModel
from models.simple_cnn import SimpleCNN
from models.lstm import LSTMModel

# Import datasets
from datasets.next_item_dataset import NextItemDataset
from datasets.mnist_dataset import MNISTDataset
from datasets.imdb_dataset import IMDBDataset

# Import quantizers for PTQ
from quantizers.no_quant import NoQuantizer
from quantizers.lsq import LSQQuantizer, LSQQuantizerWrapper
from quantizers.pact import PACTQuantizer
from quantizers.adaround import AdaRoundQuantizer, AdaRoundQuantizerWrapper
from quantizers.apot import APoTQuantizer, APoTQuantizerWrapper
from quantizers.qil import QILQuantizer, QILQuantizerWrapper


model_name_to_model_class = {
    "sasrec": SASRecModel,
    "simple_cnn": SimpleCNN,
    "lstm": LSTMModel,
}

model_name_to_dataset_class = {
    "sasrec": NextItemDataset,
    "simple_cnn": MNISTDataset,
    "lstm": IMDBDataset,
}

quantizer_name_to_quantizer_class = {
    "no_quant": NoQuantizer,
    "lsq": LSQQuantizer,
    "adaround": AdaRoundQuantizer,
    "apot": APoTQuantizer,
    "qil": QILQuantizer,
}


def create_calibration_dataloader(
    model_name: str,
    dataset_cfg: Dict[str, Any],
    batch_size: int = 32,
    num_samples: int = 512
) -> Optional[DataLoader]:
    """Create calibration dataloader for PTQ methods."""
    try:
        dataset_class = model_name_to_dataset_class.get(model_name)
        if dataset_class is None:
            return None
        
        # Load dataset
        if model_name == "sasrec":
            root_dir = dataset_cfg.get('root_dir', os.path.join('external_repos', 'TIFUKNN', 'data'))
            dataset_name = dataset_cfg.get('dataset', 'Dunnhumby')
            min_len = dataset_cfg.get('min_len', 1)
            dataset = dataset_class(root_dir=root_dir, dataset=dataset_name, split='train', min_len=min_len)
        elif model_name == "simple_cnn":
            root_dir = dataset_cfg.get('root_dir', './data')
            dataset = dataset_class(root_dir=root_dir, split='train', download=True)
        elif model_name == "lstm":
            dataset = dataset_class(split='train')
        else:
            return None
        
        # Create subset
        from torch.utils.data import Subset
        indices = list(range(min(num_samples, len(dataset))))
        subset = Subset(dataset, indices)
        
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset_class.collate_fn if hasattr(dataset_class, 'collate_fn') else None,
            num_workers=0
        )
        return loader
    except Exception as e:
        print(f"Warning: Could not create calibration dataloader: {e}")
        return None


def load_model_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a saved model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_name = checkpoint['model_name']
    model_cfg = checkpoint.get('model_cfg', {})
    dataset_cfg = checkpoint.get('dataset_cfg', {})
    quantizer_name = checkpoint['quantizer_name']
    quantizer_cfg = checkpoint.get('quantizer_cfg', {})
    
    # Reconstruct model
    model_class = model_name_to_model_class.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Handle model-specific initialization
    if model_name == "sasrec":
        # For SASRec, we need num_items and max_seq_len
        num_items = model_cfg.get('num_items')
        if num_items is None:
            # Try to infer from dataset
            dataset_name = dataset_cfg.get('dataset', 'Dunnhumby')
            root_dir = dataset_cfg.get('root_dir', os.path.join('external_repos', 'TIFUKNN', 'data'))
            try:
                ds = NextItemDataset(root_dir=root_dir, dataset=dataset_name, split='train', min_len=1)
                num_items = ds.num_items
                model_cfg['num_items'] = num_items + 1  # +1 for mask token
            except Exception as e:
                print(f"Warning: Could not infer num_items: {e}")
                num_items = model_cfg.get('num_items', 1000)
        
        max_seq_len = model_cfg.get('max_seq_len', 10)
        pad_token_id = model_cfg.get('pad_token_id', 0)
        
        model = model_class(
            num_items=num_items,
            max_seq_len=max_seq_len,
            hidden_size=model_cfg.get('hidden_size', 64),
            num_heads=model_cfg.get('num_heads', 1),
            num_layers=model_cfg.get('num_layers', 1),
            dropout=model_cfg.get('dropout', 0.1),
            pad_token_id=pad_token_id,
            device=device,
        )
    elif model_name == "lstm":
        # For LSTM, we need vocab_size and num_classes
        vocab_size = model_cfg.get('vocab_size')
        num_classes = model_cfg.get('num_classes')
        
        if vocab_size is None or num_classes is None:
            # Try to infer from dataset
            try:
                ds = IMDBDataset(split='train')
                vocab_size = ds.inferred_params.vocab_size
                num_classes = ds.inferred_params.num_classes
            except Exception as e:
                print(f"Warning: Could not infer vocab_size/num_classes: {e}")
                vocab_size = model_cfg.get('vocab_size', 50000)
                num_classes = model_cfg.get('num_classes', 2)
        
        model = model_class(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=model_cfg.get('embedding_dim', 256),
            hidden_size=model_cfg.get('hidden_size', 256),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.2),
            bidirectional=model_cfg.get('bidirectional', True),
            pad_idx=model_cfg.get('pad_idx', 0),
            max_seq_len=model_cfg.get('max_seq_len'),
            embedding_dropout=model_cfg.get('embedding_dropout', 0.1),
        )
    elif model_name == "simple_cnn":
        model = model_class()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Apply quantization structure if needed (to match saved state_dict)
    if quantizer_name != "no_quant":
        try:
            quantizer_class = quantizer_name_to_quantizer_class.get(quantizer_name)
            if quantizer_class is not None:
                quantizer_obj = quantizer_class(**quantizer_cfg)
                
                # Wrap quantizer and prepare model structure
                if quantizer_name == "lsq":
                    from quantizers.lsq import LSQQuantizerWrapper
                    wrapper = LSQQuantizerWrapper(quantizer_obj)
                    model = wrapper.prepare_model(model)
                elif quantizer_name == "adaround":
                    from quantizers.adaround import AdaRoundQuantizerWrapper
                    bit_width = quantizer_cfg.get('bit_width', 4)
                    wrapper = AdaRoundQuantizerWrapper(quantizer_obj, bit_width=bit_width)
                    model = wrapper.prepare_model(model)
                elif quantizer_name == "apot":
                    from quantizers.apot import APoTQuantizerWrapper
                    k = quantizer_cfg.get('k', 2)
                    wrapper = APoTQuantizerWrapper(quantizer_obj, k=k)
                    model = wrapper.prepare_model(model)
                elif quantizer_name == "qil":
                    from quantizers.qil import QILQuantizerWrapper
                    gamma_weight = quantizer_cfg.get('gamma_weight', None)
                    skip_first_last = quantizer_cfg.get('skip_first_last', True)
                    wrapper = QILQuantizerWrapper(quantizer_obj, gamma_weight=gamma_weight, skip_first_last=skip_first_last)
                    model = wrapper.prepare_model(model)
        except Exception as e:
            print(f"Warning: Could not prepare quantized model structure: {e}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    metadata = {
        'model_name': model_name,
        'quantizer_name': quantizer_name,
        'config_hash': checkpoint.get('config_hash', 'unknown'),
        'model_cfg': model_cfg,
        'dataset_cfg': dataset_cfg,
        'quantizer_cfg': quantizer_cfg,
    }
    
    return model, metadata


def extract_weights(model: nn.Module, model_name: str) -> Dict[str, np.ndarray]:
    """Extract relevant weights from model based on model type."""
    weights = {}
    
    if model_name == "lstm":
        # Extract LSTM layer weights
        if hasattr(model, 'lstm'):
            lstm = model.lstm
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    weights[f'lstm.{name}'] = param.detach().cpu().numpy().flatten()
    
    elif model_name == "simple_cnn":
        # Extract Conv layer weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.weight is not None:
                    weights[f'{name}.weight'] = module.weight.detach().cpu().numpy().flatten()
    
    elif model_name == "sasrec":
        # Extract Q/K/V weights from transformer encoder
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Extract Q, K, V projection weights
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    w = module.in_proj_weight.detach().cpu().numpy()
                    d = w.shape[0] // 3
                    weights[f'{name}.q_proj'] = w[:d].flatten()
                    weights[f'{name}.k_proj'] = w[d:2*d].flatten()
                    weights[f'{name}.v_proj'] = w[2*d:].flatten()
                elif hasattr(module, 'q_proj_weight'):
                    # Separate Q, K, V weights
                    if module.q_proj_weight is not None:
                        weights[f'{name}.q_proj'] = module.q_proj_weight.detach().cpu().numpy().flatten()
                    if hasattr(module, 'k_proj_weight') and module.k_proj_weight is not None:
                        weights[f'{name}.k_proj'] = module.k_proj_weight.detach().cpu().numpy().flatten()
                    if hasattr(module, 'v_proj_weight') and module.v_proj_weight is not None:
                        weights[f'{name}.v_proj'] = module.v_proj_weight.detach().cpu().numpy().flatten()
    
    return weights


def extract_onnx_weights(onnx_path: str, model_name: str) -> Dict[str, np.ndarray]:
    """Extract weights from ONNX model."""
    import onnx
    
    try:
        model = onnx.load(onnx_path)
        weights = {}
        
        # Map model types to layer names we care about
        if model_name == "lstm":
            target_prefixes = ['lstm', 'embedding', 'classifier']
        elif model_name == "simple_cnn":
            target_prefixes = ['conv', 'fc']
        elif model_name == "sasrec":
            target_prefixes = ['q_proj', 'k_proj', 'v_proj', 'embedding']
        else:
            target_prefixes = []
        
        # Extract weights from initializers
        for initializer in model.graph.initializer:
            name = initializer.name
            
            # Check if this is a weight we're interested in
            if any(prefix in name.lower() for prefix in target_prefixes):
                # Convert to numpy array
                weight_data = onnx.numpy_helper.to_array(initializer)
                
                # Only include actual weight matrices (not biases or scalars)
                if weight_data.size > 100:  # Skip small tensors like biases
                    weights[name] = weight_data.flatten()
        
        return weights
    except Exception as e:
        print(f"Warning: Could not extract weights from ONNX model: {e}")
        return {}


def plot_weight_distribution(weights_dict: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """Plot weight distributions for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_id, weights in weights_dict.items():
        if not weights:
            continue
        
        # Create subplots for each weight tensor
        n_weights = len(weights)
        if n_weights == 0:
            continue
        
        n_cols = min(3, n_weights)
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_weights == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (weight_name, weight_values) in enumerate(weights.items()):
            ax = axes[idx]
            ax.hist(weight_values, bins=100, alpha=0.7, edgecolor='black')
            ax.set_title(f'{weight_name}')
            ax.set_xlabel('Weight value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(weight_values)
            std_val = np.std(weight_values)
            ax.text(0.02, 0.98, f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(n_weights, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{model_id}_weight_distribution.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved weight distribution plot: {output_path}")


def plot_onnx_weight_distributions(results: List[Dict[str, Any]], output_dir: str):
    """Plot weight distributions for ONNX INT8 models and compare with PyTorch."""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        # Skip if no ONNX results
        if 'onnx_results' not in result or not result['onnx_results']:
            continue
        
        model_id = result['model_id']
        model_name = result['metadata']['model_name']
        
        # Get ONNX model path
        onnx_int8_path = result['onnx_results'].get('onnx_int8_path')
        if not onnx_int8_path or not os.path.exists(onnx_int8_path):
            continue
        
        # Extract ONNX weights
        onnx_weights = extract_onnx_weights(onnx_int8_path, model_name)
        if not onnx_weights:
            continue
        
        # Get PyTorch weights for comparison
        pytorch_weights = result.get('weights', {})
        
        # Create comparison plots
        n_weights = len(onnx_weights)
        n_cols = min(3, n_weights)
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_weights == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (onnx_name, onnx_values) in enumerate(onnx_weights.items()):
            ax = axes[idx]
            
            # Plot ONNX INT8 weights
            ax.hist(onnx_values, bins=100, alpha=0.6, label='ONNX INT8', 
                   color='coral', edgecolor='black')
            
            # Try to find corresponding PyTorch weights
            pytorch_found = False
            for pt_name, pt_values in pytorch_weights.items():
                # Simple matching - check if names are similar
                if any(keyword in onnx_name.lower() and keyword in pt_name.lower() 
                       for keyword in ['lstm', 'embedding', 'conv', 'fc', 'classifier', 'q_proj', 'k_proj', 'v_proj']):
                    ax.hist(pt_values, bins=100, alpha=0.5, label='PyTorch', 
                           color='steelblue', edgecolor='black')
                    pytorch_found = True
                    break
            
            # Shorten ONNX name for display
            display_name = onnx_name.split('/')[-1] if '/' in onnx_name else onnx_name
            ax.set_title(f'{display_name}')
            ax.set_xlabel('Weight value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics for ONNX INT8
            mean_val = np.mean(onnx_values)
            std_val = np.std(onnx_values)
            min_val = np.min(onnx_values)
            max_val = np.max(onnx_values)
            unique_vals = len(np.unique(onnx_values))
            
            stats_text = f'ONNX INT8:\nμ={mean_val:.4f}\nσ={std_val:.4f}\n'
            stats_text += f'range=[{min_val:.2f}, {max_val:.2f}]\nunique={unique_vals}'
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, va='top', ha='left', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(n_weights, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{model_id} - Weight Distribution (PyTorch vs ONNX INT8)', fontsize=14, y=1.0)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{model_id}_onnx_int8_weight_distribution.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ONNX INT8 weight distribution plot: {output_path}")


def _convert_adaround_to_standard_layers(model: nn.Module) -> nn.Module:
    """Convert AdaRound quantized layers to standard layers with quantized weights"""
    from quantizers.adaround import AdaRoundConv2d, AdaRoundLinear, AdaRoundEmbedding, AdaRoundLayerNorm
    
    def convert_module(module):
        for name, child in list(module.named_children()):
            if isinstance(child, AdaRoundConv2d):
                # Quantize weights once
                with torch.no_grad():
                    q_weight = child.weight_quant.get_quantized_weight(child.conv.weight)
                    q_bias = None
                    if child.conv.bias is not None:
                        q_bias = child.conv.bias  # Bias not quantized in AdaRound typically
                
                # Create standard Conv2d with quantized weights
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
                # Quantize weights once
                with torch.no_grad():
                    q_weight = child.weight_quant.get_quantized_weight(child.fc.weight)
                    q_bias = None
                    if child.fc.bias is not None:
                        q_bias = child.fc.bias
                
                # Create standard Linear with quantized weights
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
                # Quantize embedding weights once
                with torch.no_grad():
                    q_weight = child.weight_quant.get_quantized_weight(child.emb.weight)
                
                # Create standard Embedding with quantized weights
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
                
            elif isinstance(child, AdaRoundLayerNorm):
                # LayerNorm stays in FP32, just restore original
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


def prepare_model_for_inference(
    model: nn.Module,
    quantizer_name: str,
    quantizer_cfg: Dict[str, Any],
    model_name: str,
    dataset_cfg: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """Prepare model for inference (apply PTQ if needed)"""
    
    if quantizer_name == "no_quant":
        return model
    
    # Create quantizer wrapper
    quantizer_class = quantizer_name_to_quantizer_class.get(quantizer_name)
    if quantizer_class is None:
        print(f"Warning: Unknown quantizer {quantizer_name}, skipping preparation")
        return model
    
    try:
        quantizer_obj = quantizer_class(**quantizer_cfg)
        
        # Wrap quantizer
        if quantizer_name == "lsq":
            from quantizers.lsq import LSQQuantizerWrapper
            wrapper = LSQQuantizerWrapper(quantizer_obj)
        elif quantizer_name == "adaround":
            from quantizers.adaround import AdaRoundQuantizerWrapper
            bit_width = quantizer_cfg.get('bit_width', 4)
            wrapper = AdaRoundQuantizerWrapper(quantizer_obj, bit_width=bit_width)
        elif quantizer_name == "apot":
            from quantizers.apot import APoTQuantizerWrapper
            k = quantizer_cfg.get('k', 2)
            wrapper = APoTQuantizerWrapper(quantizer_obj, k=k)
        elif quantizer_name == "qil":
            from quantizers.qil import QILQuantizerWrapper
            gamma_weight = quantizer_cfg.get('gamma_weight', None)
            skip_first_last = quantizer_cfg.get('skip_first_last', True)
            wrapper = QILQuantizerWrapper(quantizer_obj, gamma_weight=gamma_weight, skip_first_last=skip_first_last)
        else:
            print(f"Warning: No wrapper for quantizer {quantizer_name}")
            return model
        
        # Apply quantizer-specific preparation for inference
        if quantizer_name == "adaround":
            # AdaRound: Skip calibration for benchmark (model already trained with it)
            # Just convert quantized layers to regular layers with quantized weights
            print("Converting AdaRound layers to standard layers with quantized weights...")
            model = _convert_adaround_to_standard_layers(model)
            model.eval()
        elif quantizer_name == "qil":
            # QIL converts quantized layers to regular layers with quantized weights
            model = wrapper.prepare_for_inference(model=model)
        elif quantizer_name == "lsq":
            # LSQ converts quantized layers to regular layers with quantized weights
            model = wrapper.prepare_for_inference(model=model)
        elif quantizer_name == "apot":
            # APoT converts quantized layers to regular layers with quantized weights
            model = wrapper.prepare_for_inference(model=model)
        else:
            # Other quantizers might have their own preparation
            if hasattr(wrapper, 'prepare_for_inference'):
                model = wrapper.prepare_for_inference(model=model)
        
        print(f"Model prepared for inference with {quantizer_name}")
        
    except Exception as e:
        print(f"Warning: Could not prepare model for inference: {e}")
        import traceback
        traceback.print_exc()
    
    return model


def evaluate_model_quality(
    model: nn.Module,
    model_name: str,
    dataset_cfg: Dict[str, Any],
    n_samples: int,
    batch_size: int,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model quality using appropriate metrics"""
    from metrics import model_name_to_metrics
    
    metrics_cfg = model_name_to_metrics.get(model_name)
    if not metrics_cfg:
        print(f"No metrics configured for model: {model_name}")
        return {}
    
    dataset_class = model_name_to_dataset_class.get(model_name)
    if dataset_class is None:
        print(f"Unknown dataset for model: {model_name}")
        return {}
    
    try:
        # Load dataset
        if model_name == "sasrec":
            root_dir = dataset_cfg.get('root_dir', os.path.join('external_repos', 'TIFUKNN', 'data'))
            dataset_name = dataset_cfg.get('dataset', 'Dunnhumby')
            min_len = dataset_cfg.get('min_len', 1)
            dataset = dataset_class(root_dir=root_dir, dataset=dataset_name, split='val', min_len=min_len)
        elif model_name == "simple_cnn":
            root_dir = dataset_cfg.get('root_dir', './data')
            dataset = dataset_class(root_dir=root_dir, split='val', download=True)
        elif model_name == "lstm":
            dataset = dataset_class(split='val')
        else:
            return {}
    except Exception as e:
        print(f"Warning: Could not load dataset for evaluation: {e}")
        return {}
    
    # Create dataloader with limited samples
    from torch.utils.data import Subset
    import random
    
    actual_n_samples = min(n_samples, len(dataset))
    
    # For classification tasks, ensure balanced sampling
    if model_name in ["lstm", "simple_cnn"]:
        # Try to get balanced samples
        try:
            # Get labels
            if model_name == "lstm":
                labels = [dataset[i][1] for i in range(len(dataset))]
            else:
                labels = [dataset[i][1] for i in range(len(dataset))]
            
            # Sample from each class
            class_indices = {}
            for idx, label in enumerate(labels):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
            
            # Sample equally from each class
            indices = []
            samples_per_class = actual_n_samples // len(class_indices)
            for class_idx_list in class_indices.values():
                if len(class_idx_list) >= samples_per_class:
                    sampled = random.sample(class_idx_list, samples_per_class)
                else:
                    sampled = class_idx_list
                indices.extend(sampled)
            
            # If we need more samples, add randomly
            if len(indices) < actual_n_samples:
                remaining = list(set(range(len(dataset))) - set(indices))
                indices.extend(random.sample(remaining, min(actual_n_samples - len(indices), len(remaining))))
            
            indices = indices[:actual_n_samples]
            print(f"Balanced sampling: {len(indices)} samples from {len(class_indices)} classes")
        except Exception as e:
            print(f"Could not do balanced sampling: {e}, using random sampling")
            indices = random.sample(range(len(dataset)), actual_n_samples)
    else:
        indices = list(range(actual_n_samples))
    
    subset = Subset(dataset, indices)
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_class.collate_fn if hasattr(dataset_class, 'collate_fn') else None,
        num_workers=0
    )
    
    model.eval()
    
    # Collect predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating model quality"):
            if model_name == "sasrec":
                inputs, lengths, targets = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                targets = targets.to(device)
                outputs = model(inputs, lengths)
            elif model_name == "simple_cnn":
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
            elif model_name == "lstm":
                inputs, lengths, targets = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                targets = targets.to(device)
                outputs = model(inputs, lengths)
            else:
                continue
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    if not all_preds:
        print("Warning: No predictions collected")
        return {}
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"Collected {len(all_targets)} samples for evaluation")
    print(f"Predictions shape: {all_preds.shape}, Targets shape: {all_targets.shape}")
    print(f"Target distribution: {torch.bincount(all_targets.long()).tolist()}")
    
    # Compute metrics
    results = {}
    for metric_name, metric_fn in metrics_cfg:
        try:
            score = metric_fn(all_preds, all_targets)
            results[metric_name] = float(score)
            print(f"{metric_name}: {score:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute {metric_name}: {e}")
            import traceback
            traceback.print_exc()
            results[metric_name] = float('nan')
    
    return results


def export_and_benchmark_onnx(
    model: nn.Module,
    model_name: str,
    quantizer_name: str,
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    output_dir: str,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    quantization_type: str = 'dynamic',
    num_warmup: int = 10,
    num_iterations: int = 100
) -> Dict[str, Any]:
    """Export model to ONNX INT8 and benchmark it"""
    from quantizers.onnx_converter import get_onnx_converter, create_dummy_input
    import onnxruntime as ort
    
    # Get converter
    converter = get_onnx_converter(model_name, quantizer_name)
    if converter is None:
        return {}
    
    # Create dummy input
    dummy_input = create_dummy_input(model_name, model_cfg, device)
    
    # Create calibration loader if needed
    calibration_loader = None
    if quantization_type == 'static':
        calibration_loader = create_calibration_dataloader(
            model_name=model_name,
            dataset_cfg=dataset_cfg,
            batch_size=batch_size,
            num_samples=100
        )
    
    # Export to ONNX
    onnx_dir = os.path.join(output_dir, 'onnx_models')
    fp32_path, int8_path = converter.convert_to_onnx_int8(
        model=model,
        dummy_input=dummy_input,
        output_dir=onnx_dir,
        calibration_loader=calibration_loader,
        quantization_type=quantization_type
    )
    
    # Measure ONNX model sizes
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)  # MB
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)  # MB
    
    # Benchmark ONNX INT8 model
    inference_speed_int8 = measure_onnx_inference_speed(
        onnx_path=int8_path,
        model_name=model_name,
        dataset_cfg=dataset_cfg,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        num_warmup=num_warmup,
        num_iterations=num_iterations
    )
    
    # Measure quality of ONNX INT8 model
    print("Evaluating ONNX INT8 model quality...")
    onnx_quality = evaluate_onnx_model_quality(
        onnx_path=int8_path,
        model_name=model_name,
        dataset_cfg=dataset_cfg,
        device=device,
        eval_n_samples=n_samples
    )
    
    return {
        'onnx_fp32_path': fp32_path,
        'onnx_int8_path': int8_path,
        'onnx_fp32_size_mb': fp32_size,
        'onnx_int8_size_mb': int8_size,
        'onnx_size_mb': int8_size,
        'inference_speed_int8': inference_speed_int8,
        'onnx_quality_metrics': onnx_quality,
    }


def evaluate_onnx_model_quality(
    onnx_path: str,
    model_name: str,
    dataset_cfg: Dict[str, Any],
    device: torch.device,
    eval_n_samples: int = 2000
) -> Dict[str, float]:
    """Evaluate ONNX model quality using appropriate metrics"""
    import onnxruntime as ort
    from torch.utils.data import Subset
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Get expected input shape
    expected_shape = session.get_inputs()[0].shape
    if isinstance(expected_shape[1], str):
        expected_seq_len = None
    else:
        expected_seq_len = expected_shape[1]
    
    # Load dataset
    dataset_class = model_name_to_dataset_class.get(model_name)
    if dataset_class is None:
        return {}
    
    try:
        if model_name == "sasrec":
            root_dir = dataset_cfg.get('root_dir', os.path.join('external_repos', 'TIFUKNN', 'data'))
            dataset_name = dataset_cfg.get('dataset', 'Dunnhumby')
            min_len = dataset_cfg.get('min_len', 1)
            dataset = dataset_class(root_dir=root_dir, dataset=dataset_name, split='val', min_len=min_len)
        elif model_name == "simple_cnn":
            root_dir = dataset_cfg.get('root_dir', './data')
            dataset = dataset_class(root_dir=root_dir, split='val', download=True)
        elif model_name == "lstm":
            dataset = dataset_class(split='val')
        else:
            return {}
    except Exception as e:
        print(f"Warning: Could not load dataset for quality evaluation: {e}")
        return {}
    
    # Balanced sampling for binary classification
    if model_name in ["lstm", "simple_cnn"]:
        eval_n_samples = max(eval_n_samples, 2000)
        target_per_class = eval_n_samples // 2
        
        indices_by_class = {0: [], 1: []}
        for idx in range(len(dataset)):
            try:
                batch = dataset[idx]
                # Unpack depending on length: (input, target) or (input, lengths, target)
                if len(batch) == 3:
                    _, _, target = batch
                else:
                    _, target = batch
            except Exception as e:
                print(f"Warning: Failed to get sample {idx}: {e}")
                continue
            
            if target in indices_by_class:
                indices_by_class[target].append(idx)
                if len(indices_by_class[0]) >= target_per_class and len(indices_by_class[1]) >= target_per_class:
                    break
        
        selected_indices = indices_by_class[0][:target_per_class] + indices_by_class[1][:target_per_class]
        print(f"Balanced sampling: {len(selected_indices)} samples from {len(indices_by_class)} classes")
    else:
        actual_n_samples = min(eval_n_samples, len(dataset))
        selected_indices = list(range(actual_n_samples))
    
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        collate_fn=dataset_class.collate_fn if hasattr(dataset_class, 'collate_fn') else None,
        num_workers=0
    )
    
    # Collect predictions and targets
    all_preds = []
    all_targets = []
    
    for batch in tqdm(loader, desc="Evaluating ONNX model quality"):
        if model_name in ["sasrec", "lstm"]:
            inputs, lengths, targets = batch
        else:
            inputs, targets = batch
        
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        
        # Pad or truncate to expected length
        if expected_seq_len is not None and model_name in ["sasrec", "lstm"]:
            if inputs.shape[1] > expected_seq_len:
                inputs = inputs[:, :expected_seq_len]
            elif inputs.shape[1] < expected_seq_len:
                pad_width = ((0, 0), (0, expected_seq_len - inputs.shape[1]))
                inputs = np.pad(inputs, pad_width, mode='constant', constant_values=0)
        
        # Run inference
        outputs = session.run([output_name], {input_name: inputs})[0]
        
        all_preds.append(torch.from_numpy(outputs))
        all_targets.append(targets)
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print(f"Collected {len(targets)} samples for ONNX evaluation")
    print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
    
    # Compute metrics
    from metrics import rocauc, ndcg_at_k, hr_at_k
    
    metrics_dict = {}
    
    if model_name in ["lstm", "simple_cnn"]:
        # Binary classification
        auc = rocauc(predictions, targets)
        metrics_dict['ROCAUC'] = auc
        print(f"ONNX ROCAUC: {auc:.4f}")
    
    elif model_name == "sasrec":
        # Recommendation metrics
        k = 10
        ndcg = ndcg_at_k(predictions, targets, k=k)
        hr = hr_at_k(predictions, targets, k=k)
        metrics_dict[f'NDCG@{k}'] = ndcg
        metrics_dict[f'HR@{k}'] = hr
        print(f"ONNX NDCG@{k}: {ndcg:.4f}, HR@{k}: {hr:.4f}")
    
    return metrics_dict


def measure_onnx_inference_speed(
    onnx_path: str,
    model_name: str,
    dataset_cfg: Dict[str, Any],
    n_samples: int,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> Dict[str, float]:
    """Measure ONNX model inference speed"""
    import onnxruntime as ort
    
    # Create ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['CPUExecutionProvider']
    if device.type == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Load dataset
    dataset_class = model_name_to_dataset_class.get(model_name)
    if dataset_class is None:
        return {'avg_time_ms': float('nan'), 'throughput_samples_per_sec': float('nan')}
    
    try:
        if model_name == "sasrec":
            root_dir = dataset_cfg.get('root_dir', os.path.join('external_repos', 'TIFUKNN', 'data'))
            dataset_name = dataset_cfg.get('dataset', 'Dunnhumby')
            min_len = dataset_cfg.get('min_len', 1)
            dataset = dataset_class(root_dir=root_dir, dataset=dataset_name, split='val', min_len=min_len)
        elif model_name == "simple_cnn":
            root_dir = dataset_cfg.get('root_dir', './data')
            dataset = dataset_class(root_dir=root_dir, split='val', download=True)
        elif model_name == "lstm":
            dataset = dataset_class(split='val')
        else:
            return {'avg_time_ms': float('nan'), 'throughput_samples_per_sec': float('nan')}
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        return {'avg_time_ms': float('nan'), 'throughput_samples_per_sec': float('nan')}
    
    from torch.utils.data import Subset
    actual_n_samples = min(n_samples, len(dataset))
    indices = list(range(actual_n_samples))
    subset = Subset(dataset, indices)
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_class.collate_fn if hasattr(dataset_class, 'collate_fn') else None,
        num_workers=0
    )
    
    # Get expected input shape from ONNX model
    expected_shape = session.get_inputs()[0].shape
    if isinstance(expected_shape[1], str):  # Dynamic dimension
        expected_seq_len = None
    else:
        expected_seq_len = expected_shape[1]
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= num_warmup:
            break
        if model_name in ["sasrec", "lstm"]:
            inputs, _, _ = batch
        else:
            inputs, _ = batch
        
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        
        # Pad or truncate to expected length for LSTM/sequential models
        if expected_seq_len is not None and model_name in ["sasrec", "lstm"]:
            if inputs.shape[1] > expected_seq_len:
                inputs = inputs[:, :expected_seq_len]
            elif inputs.shape[1] < expected_seq_len:
                pad_width = ((0, 0), (0, expected_seq_len - inputs.shape[1]))
                inputs = np.pad(inputs, pad_width, mode='constant', constant_values=0)
        
        session.run([output_name], {input_name: inputs})
    
    # Measure
    times = []
    total_samples = 0
    
    for i, batch in enumerate(tqdm(loader, desc="ONNX inference speed")):
        if i >= num_iterations:
            break
        
        if model_name in ["sasrec", "lstm"]:
            inputs, _, _ = batch
        else:
            inputs, _ = batch
        
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
            batch_samples = inputs.shape[0]
        else:
            batch_samples = len(inputs)
        
        # Pad or truncate to expected length for LSTM/sequential models
        if expected_seq_len is not None and model_name in ["sasrec", "lstm"]:
            if inputs.shape[1] > expected_seq_len:
                inputs = inputs[:, :expected_seq_len]
            elif inputs.shape[1] < expected_seq_len:
                pad_width = ((0, 0), (0, expected_seq_len - inputs.shape[1]))
                inputs = np.pad(inputs, pad_width, mode='constant', constant_values=0)
        
        start_time = time.perf_counter()
        session.run([output_name], {input_name: inputs})
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)
        total_samples += batch_samples
    
    avg_time_ms = np.mean(times)
    throughput = total_samples / (sum(times) / 1000)
    
    return {
        'avg_time_ms': avg_time_ms,
        'throughput_samples_per_sec': throughput,
        'total_samples': total_samples,
        'num_batches': len(times)
    }


def measure_inference_speed(
    model: nn.Module,
    model_name: str,
    dataset_cfg: Dict[str, Any],
    n_samples: int,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> Dict[str, float]:
    """Measure inference speed on a dataset."""
    dataset_class = model_name_to_dataset_class.get(model_name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset for model: {model_name}")
    
    # Load dataset
    try:
        if model_name == "sasrec":
            root_dir = dataset_cfg.get('root_dir', os.path.join('external_repos', 'TIFUKNN', 'data'))
            dataset_name = dataset_cfg.get('dataset', 'Dunnhumby')
            min_len = dataset_cfg.get('min_len', 1)
            dataset = dataset_class(root_dir=root_dir, dataset=dataset_name, split='val', min_len=min_len)
        elif model_name == "simple_cnn":
            root_dir = dataset_cfg.get('root_dir', './data')
            dataset = dataset_class(root_dir=root_dir, split='val', download=True)
        elif model_name == "lstm":
            dataset = dataset_class(split='val')
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        return {'avg_time_ms': float('nan'), 'throughput_samples_per_sec': float('nan')}
    
    # Create dataloader with limited samples
    from torch.utils.data import Subset
    
    actual_n_samples = min(n_samples, len(dataset))
    indices = list(range(actual_n_samples))
    subset = Subset(dataset, indices)
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_class.collate_fn if hasattr(dataset_class, 'collate_fn') else None,
        num_workers=0
    )
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_warmup:
                break
            if model_name == "sasrec":
                inputs, lengths, _ = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                _ = model(inputs, lengths)
            elif model_name == "simple_cnn":
                inputs, _ = batch
                inputs = inputs.to(device)
                _ = model(inputs)
            elif model_name == "lstm":
                inputs, lengths, _ = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                _ = model(inputs, lengths)
    
    # Measure inference time
    times = []
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Measuring inference speed")):
            if i >= num_iterations:
                break
            
            # Get batch size
            if model_name == "sasrec":
                inputs, lengths, _ = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                batch_samples = inputs.size(0)
            elif model_name == "simple_cnn":
                inputs, _ = batch
                inputs = inputs.to(device)
                batch_samples = inputs.size(0)
            elif model_name == "lstm":
                inputs, lengths, _ = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                batch_samples = inputs.size(0)
            
            # Measure time
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.perf_counter()
            
            if model_name == "sasrec":
                _ = model(inputs, lengths)
            elif model_name == "simple_cnn":
                _ = model(inputs)
            elif model_name == "lstm":
                _ = model(inputs, lengths)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            total_samples += batch_samples
    
    avg_time_ms = np.mean(times)
    throughput = total_samples / (sum(times) / 1000)  # samples per second
    
    return {
        'avg_time_ms': avg_time_ms,
        'throughput_samples_per_sec': throughput,
        'total_samples': total_samples,
        'num_batches': len(times)
    }


def plot_inference_speed_comparison(results: List[Dict[str, Any]], output_dir: str):
    """Plot bar chart comparing inference speeds."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    model_ids = []
    avg_times = []
    throughputs = []
    
    for result in results:
        model_id = result['model_id']
        speed_metrics = result.get('inference_speed', {})
        
        if not np.isnan(speed_metrics.get('avg_time_ms', float('nan'))):
            model_ids.append(model_id)
            avg_times.append(speed_metrics['avg_time_ms'])
            throughputs.append(speed_metrics['throughput_samples_per_sec'])
    
    if not model_ids:
        print("No valid inference speed data to plot")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average inference time
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_ids)))
    bars1 = ax1.bar(range(len(model_ids)), avg_times, color=colors)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average Inference Time (ms)')
    ax1.set_title('Inference Time Comparison')
    ax1.set_xticks(range(len(model_ids)))
    ax1.set_xticklabels(model_ids, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, avg_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Throughput
    bars2 = ax2.bar(range(len(model_ids)), throughputs, color=colors)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Throughput Comparison')
    ax2.set_xticks(range(len(model_ids)))
    ax2.set_xticklabels(model_ids, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'inference_speed_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved inference speed comparison: {output_path}")


def plot_quality_metrics_comparison(results: List[Dict[str, Any]], output_dir: str):
    """Plot bar charts comparing quality metrics across models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by metric type
    metric_data = {}
    
    for result in results:
        model_id = result['model_id']
        quality_metrics = result.get('quality_metrics', {})
        
        for metric_name, metric_value in quality_metrics.items():
            if not np.isnan(metric_value):
                if metric_name not in metric_data:
                    metric_data[metric_name] = {'model_ids': [], 'values': []}
                metric_data[metric_name]['model_ids'].append(model_id)
                metric_data[metric_name]['values'].append(metric_value)
    
    if not metric_data:
        print("No valid quality metrics data to plot")
        return
    
    # Create subplot for each metric
    n_metrics = len(metric_data)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (metric_name, data) in enumerate(metric_data.items()):
        ax = axes[idx]
        model_ids = data['model_ids']
        values = data['values']
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(model_ids)))
        bars = ax.bar(range(len(model_ids)), values, color=colors)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(range(len(model_ids)))
        ax.set_xticklabels(model_ids, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'quality_metrics_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quality metrics comparison: {output_path}")


def plot_onnx_inference_speed_comparison(results: List[Dict[str, Any]], output_dir: str):
    """Plot bar chart comparing PyTorch vs ONNX INT8 inference speeds."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data
    model_ids = []
    pytorch_times = []
    onnx_times = []
    
    for result in results:
        # Skip if no ONNX results
        if 'onnx_results' not in result or not result['onnx_results']:
            continue
        
        model_id = result['model_id']
        
        # PyTorch speed
        pytorch_speed = result.get('inference_speed', {})
        pytorch_time = pytorch_speed.get('avg_time_ms', float('nan'))
        
        # ONNX INT8 speed
        onnx_speed = result['onnx_results'].get('inference_speed_int8', {})
        onnx_time = onnx_speed.get('avg_time_ms', float('nan'))
        
        if not np.isnan(pytorch_time) and not np.isnan(onnx_time):
            model_ids.append(model_id)
            pytorch_times.append(pytorch_time)
            onnx_times.append(onnx_time)
    
    if not model_ids:
        print("No valid ONNX speed comparison data to plot")
        return
    
    # Create bar chart
    x = np.arange(len(model_ids))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Inference time comparison
    bars1 = ax1.bar(x - width/2, pytorch_times, width, label='PyTorch', color='steelblue')
    bars2 = ax1.bar(x + width/2, onnx_times, width, label='ONNX INT8', color='coral')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Inference Time (ms/batch)')
    ax1.set_title('PyTorch vs ONNX INT8 Inference Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_ids, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Speedup factor
    speedups = [pt / ot if ot > 0 else 0 for pt, ot in zip(pytorch_times, onnx_times)]
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars3 = ax2.bar(x, speedups, color=colors, alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (1x)')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Speedup (PyTorch / ONNX INT8)')
    ax2.set_title('ONNX INT8 Speedup Factor')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_ids, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, speedup in zip(bars3, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'onnx_inference_speed_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ONNX inference speed comparison: {output_path}")


def plot_onnx_quality_comparison(results: List[Dict[str, Any]], output_dir: str):
    """Plot bar chart comparing PyTorch vs ONNX INT8 quality metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by metric type
    metric_comparisons = {}
    
    for result in results:
        # Skip if no ONNX results
        if 'onnx_results' not in result or not result['onnx_results']:
            continue
        
        model_id = result['model_id']
        
        # PyTorch quality
        pytorch_quality = result.get('quality_metrics', {})
        
        # ONNX INT8 quality
        onnx_quality = result['onnx_results'].get('onnx_quality_metrics', {})
        
        # Compare metrics that exist in both
        for metric_name in pytorch_quality.keys():
            if metric_name in onnx_quality:
                pytorch_val = pytorch_quality[metric_name]
                onnx_val = onnx_quality[metric_name]
                
                if not np.isnan(pytorch_val) and not np.isnan(onnx_val):
                    if metric_name not in metric_comparisons:
                        metric_comparisons[metric_name] = {
                            'model_ids': [],
                            'pytorch_values': [],
                            'onnx_values': []
                        }
                    
                    metric_comparisons[metric_name]['model_ids'].append(model_id)
                    metric_comparisons[metric_name]['pytorch_values'].append(pytorch_val)
                    metric_comparisons[metric_name]['onnx_values'].append(onnx_val)
    
    if not metric_comparisons:
        print("No valid ONNX quality comparison data to plot")
        return
    
    # Create subplots for each metric
    n_metrics = len(metric_comparisons)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 6 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, data) in enumerate(metric_comparisons.items()):
        ax = axes[idx]
        model_ids = data['model_ids']
        pytorch_values = data['pytorch_values']
        onnx_values = data['onnx_values']
        
        x = np.arange(len(model_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pytorch_values, width, label='PyTorch', color='steelblue')
        bars2 = ax.bar(x + width/2, onnx_values, width, label='ONNX INT8', color='coral')
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}: PyTorch vs ONNX INT8')
        ax.set_xticks(x)
        ax.set_xticklabels(model_ids, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Add quality degradation annotation
        for i, (pt_val, ox_val) in enumerate(zip(pytorch_values, onnx_values)):
            degradation = ((pt_val - ox_val) / pt_val * 100) if pt_val != 0 else 0
            color = 'red' if degradation > 1 else 'green'
            ax.text(x[i], max(pt_val, ox_val) * 1.02,
                   f'{degradation:+.1f}%', ha='center', fontsize=8, color=color, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'onnx_quality_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ONNX quality comparison: {output_path}")


def log_to_mlflow(results: List[Dict[str, Any]], experiment_name: str = "QAT-Benchmark"):
    """Log benchmark results to MLflow."""
    try:
        import mlflow
        from main import get_mlflow_tracking_uri
        
        mlflow_tracking_uri = get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        for result in results:
            model_id = result['model_id']
            metadata = result['metadata']
            
            with mlflow.start_run(run_name=f"benchmark-{model_id}"):
                # Log parameters
                mlflow.log_param("model_name", metadata['model_name'])
                mlflow.log_param("quantizer_name", metadata['quantizer_name'])
                mlflow.log_param("config_hash", metadata['config_hash'])
                
                # Log model config
                for k, v in metadata.get('model_cfg', {}).items():
                    mlflow.log_param(f"model.{k}", v)
                
                # Log quantizer config
                for k, v in metadata.get('quantizer_cfg', {}).items():
                    mlflow.log_param(f"quantizer.{k}", v)
                
                # Log PyTorch model size
                mlflow.log_metric("pytorch_model_size_bytes", result.get('model_size_bytes', 0))
                mlflow.log_metric("pytorch_model_size_mb", result.get('model_size_mb', 0))
                
                # Log PyTorch inference speed metrics
                speed_metrics = result.get('inference_speed', {})
                if speed_metrics:
                    mlflow.log_metric("pytorch_avg_inference_time_ms", speed_metrics.get('avg_time_ms', 0))
                    mlflow.log_metric("pytorch_throughput_samples_per_sec", speed_metrics.get('throughput_samples_per_sec', 0))
                    mlflow.log_metric("pytorch_total_samples_measured", speed_metrics.get('total_samples', 0))
                
                # Log ONNX metrics if available
                if 'onnx' in result:
                    onnx_metrics = result['onnx']
                    mlflow.log_metric("onnx_int8_size_mb", onnx_metrics.get('onnx_int8_size_mb', 0))
                    mlflow.log_metric("onnx_fp32_size_mb", onnx_metrics.get('onnx_fp32_size_mb', 0))
                    
                    onnx_speed = onnx_metrics.get('inference_speed_int8', {})
                    if onnx_speed:
                        mlflow.log_metric("onnx_int8_avg_inference_time_ms", onnx_speed.get('avg_time_ms', 0))
                        mlflow.log_metric("onnx_int8_throughput_samples_per_sec", onnx_speed.get('throughput_samples_per_sec', 0))
                        
                        # Log speedup
                        pytorch_time = speed_metrics.get('avg_time_ms', 0)
                        onnx_time = onnx_speed.get('avg_time_ms', 0)
                        if pytorch_time > 0 and onnx_time > 0:
                            mlflow.log_metric("onnx_speedup", pytorch_time / onnx_time)
                
                # Log quality metrics
                quality_metrics = result.get('quality_metrics', {})
                for metric_name, metric_value in quality_metrics.items():
                    if not np.isnan(metric_value):
                        mlflow.log_metric(f"quality_{metric_name}", metric_value)
                
                # Log weight statistics
                weight_stats = result.get('weight_stats', {})
                for weight_name, stats in weight_stats.items():
                    mlflow.log_metric(f"weight_mean_{weight_name}", stats['mean'])
                    mlflow.log_metric(f"weight_std_{weight_name}", stats['std'])
                    mlflow.log_metric(f"weight_min_{weight_name}", stats['min'])
                    mlflow.log_metric(f"weight_max_{weight_name}", stats['max'])
        
        print(f"Results logged to MLflow experiment: {experiment_name}")
        return True
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")
        return False


def run_benchmark(
    models_dir: str,
    n_samples: int = 1000,
    batch_size: int = 32,
    output_dir: str = "benchmark_results",
    device: Optional[str] = None,
    log_mlflow: bool = False,
    num_warmup: int = 10,
    num_iterations: int = 100,
    export_onnx: bool = False,
    onnx_quantization_type: str = 'dynamic'
):
    """Run benchmark on all models in the specified directory."""
    
    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Benchmarking models from: {models_dir}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if not model_files:
        print(f"No model files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} model file(s)")
    
    results = []
    weights_for_plotting = {}
    
    for model_file in tqdm(model_files, desc="Processing models"):
        model_path = os.path.join(models_dir, model_file)
        model_id = os.path.splitext(model_file)[0]
        
        print(f"\n{'='*60}")
        print(f"Processing: {model_id}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model, metadata = load_model_checkpoint(model_path, device)
            print(f"Model: {metadata['model_name']}, Quantizer: {metadata['quantizer_name']}")
            
            # Prepare model for inference (apply PTQ if needed)
            print("Preparing model for inference...")
            model = prepare_model_for_inference(
                model=model,
                quantizer_name=metadata['quantizer_name'],
                quantizer_cfg=metadata.get('quantizer_cfg', {}),
                model_name=metadata['model_name'],
                dataset_cfg=metadata.get('dataset_cfg', {}),
                device=device
            )
            
            # Measure model size
            model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            model_size_mb = model_size_bytes / (1024 * 1024)
            print(f"Model size: {model_size_mb:.2f} MB ({model_size_bytes} bytes)")
            
            # Extract weights
            weights = extract_weights(model, metadata['model_name'])
            print(f"Extracted {len(weights)} weight tensor(s)")
            
            # Compute weight statistics
            weight_stats = {}
            for weight_name, weight_values in weights.items():
                weight_stats[weight_name] = {
                    'mean': float(np.mean(weight_values)),
                    'std': float(np.std(weight_values)),
                    'min': float(np.min(weight_values)),
                    'max': float(np.max(weight_values)),
                }
            
            weights_for_plotting[model_id] = weights
            
            # Evaluate model quality (use more samples for better metrics)
            print("Evaluating model quality...")
            eval_n_samples = max(n_samples, 2000)  # Use at least 2000 samples for quality eval
            quality_metrics = evaluate_model_quality(
                model=model,
                model_name=metadata['model_name'],
                dataset_cfg=metadata.get('dataset_cfg', {}),
                n_samples=eval_n_samples,
                batch_size=batch_size,
                device=device
            )
            if quality_metrics:
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in quality_metrics.items()])
                print(f"Quality metrics: {metrics_str}")
            
            # Measure inference speed (PyTorch)
            print("Measuring PyTorch inference speed...")
            inference_speed = measure_inference_speed(
                model,
                metadata['model_name'],
                metadata['dataset_cfg'],
                n_samples,
                batch_size,
                device,
                num_warmup=num_warmup,
                num_iterations=num_iterations
            )
            print(f"PyTorch inference: {inference_speed['avg_time_ms']:.2f} ms/batch, "
                  f"{inference_speed['throughput_samples_per_sec']:.1f} samples/sec")
            
            # ONNX export and benchmarking
            onnx_results = {}
            if export_onnx and metadata['quantizer_name'] != 'no_quant':
                try:
                    print("Exporting to ONNX INT8...")
                    onnx_results = export_and_benchmark_onnx(
                        model=model,
                        model_name=metadata['model_name'],
                        quantizer_name=metadata['quantizer_name'],
                        model_cfg=metadata.get('model_cfg', {}),
                        dataset_cfg=metadata.get('dataset_cfg', {}),
                        output_dir=output_dir,
                        n_samples=n_samples,
                        batch_size=batch_size,
                        device=device,
                        quantization_type=onnx_quantization_type,
                        num_warmup=num_warmup,
                        num_iterations=num_iterations
                    )
                    print(f"ONNX INT8 inference: {onnx_results.get('inference_speed_int8', {}).get('avg_time_ms', 0):.2f} ms/batch")
                    print(f"ONNX model size: {onnx_results.get('onnx_size_mb', 0):.2f} MB")
                    
                    # Print ONNX quality metrics
                    onnx_quality = onnx_results.get('onnx_quality_metrics', {})
                    if onnx_quality:
                        quality_str = ', '.join([f"{k}={v:.4f}" for k, v in onnx_quality.items()])
                        print(f"ONNX quality metrics: {quality_str}")
                except Exception as e:
                    print(f"Warning: ONNX export failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Store results
            result_entry = {
                'model_id': model_id,
                'metadata': metadata,
                'model_size_bytes': model_size_bytes,
                'model_size_mb': model_size_mb,
                'weight_stats': weight_stats,
                'weights': weights,  # Add raw weights for ONNX comparison plots
                'quality_metrics': quality_metrics,
                'inference_speed': inference_speed,
            }
            
            # Add ONNX results if available
            if onnx_results:
                result_entry['onnx_results'] = onnx_results
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate plots
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")
    
    plot_weight_distribution(weights_for_plotting, output_dir)
    plot_inference_speed_comparison(results, output_dir)
    plot_quality_metrics_comparison(results, output_dir)
    
    # Plot ONNX comparisons if available
    plot_onnx_inference_speed_comparison(results, output_dir)
    plot_onnx_quality_comparison(results, output_dir)
    plot_onnx_weight_distributions(results, output_dir)
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_path, 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for result in results:
            # Convert NaN to None for valid JSON
            def clean_nans(obj):
                if isinstance(obj, dict):
                    return {k: clean_nans(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nans(item) for item in obj]
                elif isinstance(obj, float) and np.isnan(obj):
                    return None
                return obj
            
            serializable_result = {
                'model_id': result['model_id'],
                'metadata': result['metadata'],
                'model_size_bytes': result.get('model_size_bytes', 0),
                'model_size_mb': result.get('model_size_mb', 0),
                'weight_stats': result['weight_stats'],
                'quality_metrics': result.get('quality_metrics', {}),
                'inference_speed': result['inference_speed'],
            }
            serializable_results.append(clean_nans(serializable_result))
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")
    
    # Save summary CSV
    summary_path = os.path.join(output_dir, 'benchmark_summary.csv')
    summary_data = []
    for result in results:
        row = {
            'model_id': result['model_id'],
            'model_name': result['metadata']['model_name'],
            'quantizer': result['metadata']['quantizer_name'],
            'pytorch_model_size_mb': result.get('model_size_mb', np.nan),
        }
        # Add PyTorch quality metrics
        for metric_name, metric_value in result.get('quality_metrics', {}).items():
            row[f'pytorch_{metric_name}'] = metric_value
        # Add PyTorch speed metrics
        row['pytorch_inference_time_ms'] = result['inference_speed'].get('avg_time_ms', np.nan)
        row['pytorch_throughput_samples_per_sec'] = result['inference_speed'].get('throughput_samples_per_sec', np.nan)
        
        # Add ONNX metrics if available
        if 'onnx_results' in result:
            onnx_res = result['onnx_results']
            row['onnx_int8_size_mb'] = onnx_res.get('onnx_int8_size_mb', np.nan)
            row['onnx_int8_inference_time_ms'] = onnx_res.get('inference_speed_int8', {}).get('avg_time_ms', np.nan)
            row['onnx_int8_throughput_samples_per_sec'] = onnx_res.get('inference_speed_int8', {}).get('throughput_samples_per_sec', np.nan)
            
            # Add ONNX quality metrics
            for metric_name, metric_value in onnx_res.get('onnx_quality_metrics', {}).items():
                row[f'onnx_int8_{metric_name}'] = metric_value
            
            # Speedup metrics
            pytorch_time = result['inference_speed'].get('avg_time_ms', float('nan'))
            onnx_time = onnx_res.get('inference_speed_int8', {}).get('avg_time_ms', float('nan'))
            if not np.isnan(pytorch_time) and not np.isnan(onnx_time) and onnx_time > 0:
                row['onnx_speedup'] = pytorch_time / onnx_time
            else:
                row['onnx_speedup'] = np.nan
        
        summary_data.append(row)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
    
    # Log to MLflow
    if log_mlflow:
        print(f"\n{'='*60}")
        print("Logging to MLflow...")
        print(f"{'='*60}")
        log_to_mlflow(results)
    
    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}")
    print(f"Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantized models")
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing saved model files"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to use for inference speed measurement (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results and plots (default: benchmark_results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log results to MLflow"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations for speed measurement (default: 100)"
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export models to ONNX INT8 and benchmark them"
    )
    parser.add_argument(
        "--onnx-quantization-type",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="ONNX quantization type (default: dynamic)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        models_dir=args.models_dir,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        log_mlflow=args.log_mlflow,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        export_onnx=args.export_onnx,
        onnx_quantization_type=args.onnx_quantization_type
    )


if __name__ == "__main__":
    main()

