"""
DNA Handling Utilities for NAS

Functions to encode, decode, and validate DNA representations
of neural network architectures.
"""

import numpy as np
from typing import List, Tuple
from .configs import get_dna_limits


def encode_dna(architecture: np.ndarray) -> List[int]:
    """
    Encodes architecture as DNA (list of integers).
    
    DNA represents a CNN architecture as a sequence of components:
    [kernel_size, num_filters, stride, pool_size, ...]
    
    Args:
        architecture: Array [num_layers, 4] with float values
    
    Returns:
        List of integers representing the DNA
    
    Example:
        >>> arch = np.array([[3, 128, 1, 1], [5, 256, 1, 2]])
        >>> dna = encode_dna(arch)
        >>> print(dna)
        [3, 128, 1, 1, 5, 256, 1, 2]
    """
    if architecture.ndim == 1:
        return architecture.astype(int).tolist()
    
    # Flatten and convert to int
    return architecture.flatten().astype(int).tolist()


def decode_dna(dna: List[int], components_per_layer: int = 4) -> np.ndarray:
    """
    Decodes DNA back into an architecture matrix.
    
    Args:
        dna: List of integers representing the DNA
        components_per_layer: Number of components per layer (default: 4)
    
    Returns:
        Array [num_layers, components_per_layer] describing the architecture
    
    Example:
        >>> dna = [3, 128, 1, 1, 5, 256, 1, 2]
        >>> arch = decode_dna(dna)
        >>> print(arch.shape)
        (2, 4)
    """
    dna_array = np.array(dna, dtype=int)
    num_layers = len(dna_array) // components_per_layer
    
    if len(dna_array) % components_per_layer != 0:
        raise ValueError(
            f"DNA length ({len(dna_array)}) must be divisible by "
            f"components_per_layer ({components_per_layer})"
        )
    
    return dna_array.reshape(num_layers, components_per_layer)


def validate_dna(dna: np.ndarray, verbose: bool = False) -> bool:
    """
    Validates that DNA values lie within allowed limits.
    
    Args:
        dna: Array [num_layers, 4] with [kernel, filters, stride, pool]
        verbose: If True, prints validation errors
    
    Returns:
        True when DNA is valid, False otherwise
    """
    if dna.ndim != 2 or dna.shape[1] != 4:
        if verbose:
            print(f"Invalid DNA shape: {dna.shape}, expected (N, 4)")
        return False
    
    limits = get_dna_limits()
    component_names = ['kernel_size', 'num_filters', 'stride', 'pool_size']
    
    for layer_idx, layer in enumerate(dna):
        for comp_idx, (value, comp_name) in enumerate(zip(layer, component_names)):
            min_val, max_val = limits[comp_name]
            
            if value < min_val or value > max_val:
                if verbose:
                    print(
                        f"Layer {layer_idx}, {comp_name}: {value} "
                        f"out of range [{min_val}, {max_val}]"
                    )
                return False
    
    return True


def clip_dna(dna: np.ndarray) -> np.ndarray:
    """
    Clips DNA to the configured boundaries.
    
    Args:
        dna: Array [num_layers, 4] with values potentially out of range
    
    Returns:
        DNA constrained to valid limits
    """
    limits = get_dna_limits()
    clipped = dna.copy()
    
    component_names = ['kernel_size', 'num_filters', 'stride', 'pool_size']
    
    for comp_idx, comp_name in enumerate(component_names):
        min_val, max_val = limits[comp_name]
        clipped[:, comp_idx] = np.clip(clipped[:, comp_idx], min_val, max_val)
    
    return clipped.astype(int)


def random_dna(num_layers: int = 3, seed: int = None) -> np.ndarray:
    """
    Generates a valid random DNA sequence.
    
    Args:
        num_layers: Number of layers to include
        seed: Optional seed for reproducibility
    
    Returns:
        Array [num_layers, 4] containing random DNA values
    """
    if seed is not None:
        np.random.seed(seed)
    
    limits = get_dna_limits()
    dna = np.zeros((num_layers, 4), dtype=int)
    
    component_names = ['kernel_size', 'num_filters', 'stride', 'pool_size']
    
    for comp_idx, comp_name in enumerate(component_names):
        min_val, max_val = limits[comp_name]
        dna[:, comp_idx] = np.random.randint(min_val, max_val + 1, size=num_layers)
    
    return dna


def dna_to_string(dna: np.ndarray) -> str:
    """
    Converts DNA into a human-readable string representation.
    
    Args:
        dna: Array [num_layers, 4]
    
    Returns:
        Formatted string describing the DNA
    """
    lines = ["DNA Architecture:"]
    lines.append("-" * 50)
    
    for layer_idx, (kernel, filters, stride, pool) in enumerate(dna):
        lines.append(
            f"Layer {layer_idx + 1}: "
            f"K={kernel}x{kernel}, F={filters}, S={stride}, P={pool}x{pool}"
        )
    
    return "\n".join(lines)
