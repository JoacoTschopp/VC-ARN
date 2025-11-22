"""
Neural Architecture Search with Reinforcement Learning

This module implements NAS using the REINFORCE algorithm to discover
CNN architectures optimized for CIFAR-10.

Core components:
    - NASController: LSTM that generates architectures (DNA)
    - ChildNetworkBuilder: Builds CNNs from DNA
    - REINFORCEOptimizer: Trains the controller with policy gradients
    - NASTrainer: Orchestrates the search process

Reference:
    Zoph, B., & Le, Q. V. (2017).
    Neural Architecture Search with Reinforcement Learning. ICLR.
"""

from .controller import NASController
from .child_builder import ChildNetworkBuilder
from .reinforce import REINFORCEOptimizer
from .trainer import NASTrainer
from .configs import get_nas_config
from .utils import encode_dna, decode_dna, validate_dna, random_dna, dna_to_string, clip_dna

__all__ = [
    'NASController',
    'ChildNetworkBuilder', 
    'REINFORCEOptimizer',
    'NASTrainer',
    'get_nas_config',
    'encode_dna',
    'decode_dna',
    'validate_dna',
    'random_dna',
    'dna_to_string',
    'clip_dna',
]

__version__ = '1.0.0'
