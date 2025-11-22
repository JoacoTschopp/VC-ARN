"""
Child Network Builder: Dynamically builds CNNs from DNA specifications.

DNA encodes architectures as lists of [kernel, filters, stride, pool].
"""

import torch
import torch.nn as nn
import numpy as np
from .utils import validate_dna, clip_dna


class ChildNetworkBuilder:
    """Builds child networks from DNA specifications."""
    
    @staticmethod
    def build_from_dna(
        dna: np.ndarray,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        input_channels: int = 3
    ) -> nn.Module:
        """
        Builds a CNN from DNA representation.
        
        Args:
            dna: [num_layers, 4] array with [kernel, filters, stride, pool]
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            input_channels: Input channels (3 for RGB)
        
        Returns:
            model: nn.Module implementing the specified architecture
        """
        # Process DNA
        if dna.ndim == 1:
            # Flattened DNA: [k1,f1,s1,p1, k2,f2,s2,p2, ...]
            dna = dna.reshape(-1, 4)
        
        # Clip and validate DNA
        dna = clip_dna(dna)
        
        if not validate_dna(dna, verbose=False):
            # If it still fails, fall back to a base architecture
            print(f"Warning: Invalid DNA, using base architecture")
            dna = np.array([[3, 64, 1, 1]] * dna.shape[0])
        
        # Build model
        layers = []
        in_channels = input_channels
        
        for layer_idx, (kernel, filters, stride, pool) in enumerate(dna):
            # Cast to int
            kernel = int(kernel)
            filters = int(filters)
            stride = int(stride)
            pool = int(pool)
            
            # Enforce sane limits just in case
            kernel = max(1, min(7, kernel))
            filters = max(32, min(512, filters))
            stride = max(1, min(2, stride))
            pool = max(1, min(3, pool))
            
            # Force odd kernel size for symmetric padding
            if kernel % 2 == 0:
                kernel += 1
            
            # Conv layer
            layers.append(nn.Conv2d(
                in_channels,
                filters,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                bias=False
            ))
            
            # BatchNorm
            layers.append(nn.BatchNorm2d(filters))
            
            # ReLU
            layers.append(nn.ReLU(inplace=True))
            
            # MaxPool (if pool > 1)
            if pool > 1:
                layers.append(nn.MaxPool2d(
                    kernel_size=pool,
                    stride=1,
                    padding=pool // 2
                ))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = filters
        
        # Classifier
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, num_classes))
        
        # Build sequential model
        model = nn.Sequential(*layers)
        
        return model
    
    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        """
        Collects basic model statistics.
        
        Args:
            model: PyTorch module
        
        Returns:
            Dict with metadata (params, layers, size_mb)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Count convolutional layers
        num_conv_layers = sum(
            1 for m in model.modules() if isinstance(m, nn.Conv2d)
        )
        
        # Estimated size in MB (4 bytes per parameter)
        size_mb = total_params * 4 / (1024 ** 2)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_conv_layers': num_conv_layers,
            'size_mb': size_mb
        }
