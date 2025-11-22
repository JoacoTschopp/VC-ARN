"""
Neural Architecture Search Hyperparameter Configuration

Defines parameters used in the architecture search process,
based on Zoph & Le (2017) paper.
"""

from typing import Dict, Any


def get_nas_config(config_name: str = 'default') -> Dict[str, Any]:
    """
    Returns NAS hyperparameter configuration.
    
    Args:
        config_name: Configuration name ('default', 'fast', 'thorough')
    
    Returns:
        Dictionary with hyperparameters
    """
    
    configs = {
        'default': {
            # Controller parameters
            'max_layers': 3,
            'components_per_layer': 4,  # [kernel, filters, stride, pool]
            'lstm_hidden_size': 100,
            'controller_lr': 0.99,
            'lr_decay': 0.96,
            'lr_decay_steps': 500,
            'beta': 1e-4,  # L2 regularization
            'baseline_ema_alpha': 0.95,
            
            # Child Network parameters
            'child_lr': 3e-5,
            'child_epochs': 100,
            'child_batch_size': 20,
            'child_dropout': 0.2,
            
            # NAS Search parameters
            'max_episodes': 2000,
            'children_per_episode': 10,
            'save_every': 50,  # Save checkpoint every N episodes
            
            # Paths
            'checkpoint_dir': 'checkpoints/nas',
            'log_dir': 'logs/nas',
        },
        
        'fast': {
            # Fast configuration for testing
            'max_layers': 2,
            'components_per_layer': 4,
            'lstm_hidden_size': 50,
            'controller_lr': 0.99,
            'lr_decay': 0.96,
            'lr_decay_steps': 100,
            'beta': 1e-4,
            'baseline_ema_alpha': 0.95,
            
            'child_lr': 3e-5,
            'child_epochs': 20,
            'child_batch_size': 32,
            'child_dropout': 0.2,
            
            'max_episodes': 100,
            'children_per_episode': 5,
            'save_every': 10,
            
            'checkpoint_dir': 'checkpoints/nas_fast',
            'log_dir': 'logs/nas_fast',
        },
        
        'thorough': {
            # Exhaustive search
            'max_layers': 5,
            'components_per_layer': 4,
            'lstm_hidden_size': 150,
            'controller_lr': 0.99,
            'lr_decay': 0.96,
            'lr_decay_steps': 1000,
            'beta': 1e-4,
            'baseline_ema_alpha': 0.95,
            
            'child_lr': 3e-5,
            'child_epochs': 150,
            'child_batch_size': 16,
            'child_dropout': 0.2,
            
            'max_episodes': 5000,
            'children_per_episode': 15,
            'save_every': 100,
            
            'checkpoint_dir': 'checkpoints/nas_thorough',
            'log_dir': 'logs/nas_thorough',
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Config '{config_name}' does not exist. Options: {list(configs.keys())}")
    
    return configs[config_name]


# DNA component limits
DNA_LIMITS = {
    'kernel_size': (1, 7),      # Min: 1x1, Max: 7x7
    'num_filters': (32, 512),   # Min: 32, Max: 512 filters
    'stride': (1, 2),           # 1 or 2
    'pool_size': (1, 3),        # 1 (no pool), 2 or 3
}


def get_dna_limits() -> Dict[str, tuple]:
    """Returns DNA validation limits."""
    return DNA_LIMITS.copy()
