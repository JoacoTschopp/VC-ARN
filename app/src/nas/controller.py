"""
NAS Controller: LSTM that generates neural network architectures

The controller is a Reinforcement Learning model that learns to generate
high-potential CNN architectures by maximizing validation accuracy.
"""

import torch
import torch.nn as nn
import numpy as np


class NASController(nn.Module):
    """
    LSTM-based controller for Neural Architecture Search.
    
    It generates architectures encoded as DNA:
        DNA = [[kernel, filters, stride, pool], ...]
    
    Args:
        num_layers: Number of layers to generate
        components_per_layer: Components per layer (default: 4)
        hidden_size: LSTM hidden state size
        device: Torch device where the model runs
    """
    
    def __init__(
        self, 
        num_layers: int = 3,
        components_per_layer: int = 4,
        hidden_size: int = 100,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.components_per_layer = components_per_layer
        self.num_outputs = num_layers * components_per_layer
        self.hidden_size = hidden_size
        self.device = device
        
        # LSTM to generate the sequence
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # FC to map hidden state -> DNA
        self.fc = nn.Linear(hidden_size, self.num_outputs)
        
        # Initialize bias to 0.01 (as in the paper)
        nn.init.constant_(self.fc.bias, 0.01)
        
        self.to(device)
    
    def forward(self, prev_architecture: torch.Tensor):
        """
        Generates a new architecture conditioned on the previous one.
        
        Args:
            prev_architecture: [batch, num_outputs] previous architecture
        
        Returns:
            architecture: [batch, num_outputs] new architecture
        """
        # Expand dimension for LSTM: [batch, num_outputs, 1]
        x = prev_architecture.unsqueeze(-1)
        
        # LSTM forward: [batch, num_outputs, hidden_size]
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep: [batch, hidden_size]
        last_output = lstm_out[:, -1, :]
        
        # Generate DNA: [batch, num_outputs]
        architecture = self.fc(last_output)
        
        return architecture
    
    def sample(self, num_samples: int = 1):
        """
        Generates architectures by sampling the controller.
        
        Args:
            num_samples: Number of architectures to produce
        
        Returns:
            dna_list: List of DNAs as numpy arrays
        """
        self.eval()
        with torch.no_grad():
            # Initialize with a baseline architecture
            prev_arch = torch.tensor(
                [[10.0, 128.0, 1.0, 1.0] * self.num_layers],
                dtype=torch.float32,
                device=self.device
            ).repeat(num_samples, 1)
            
            # Generate new architecture
            architecture = self.forward(prev_arch)
            
            # Scale and convert to integers (as in the paper)
            # LSTM outputs are small, multiply by 100
            dna = torch.abs(architecture) * 100
            dna = dna.clamp(min=1, max=512)  # Clamp to reasonable ranges
            dna = dna.int()
            
        self.train()
        return dna.cpu().numpy()
    
    def get_policy_loss(self, architectures: torch.Tensor, rewards: torch.Tensor):
        """
        Computes the policy gradient (REINFORCE) loss.
        
        Args:
            architectures: [batch, num_outputs] generated architectures
            rewards: [batch] rewards obtained (validation accuracies)
        
        Returns:
            loss: Scalar loss for backpropagation
        """
        # Normalize architectures to approximately [0, 1]
        normalized_arch = architectures / 100.0
        
        # Forward pass
        predictions = self.forward(normalized_arch)
        
        # Reconstruction loss (MSE to original architecture)
        reconstruction_loss = nn.functional.mse_loss(
            predictions, 
            normalized_arch,
            reduction='none'
        ).mean(dim=1)  # [batch]
        
        # REINFORCE: multiply loss by rewards
        # Negative rewards because we minimize the loss
        policy_loss = (reconstruction_loss * (-rewards)).mean()
        
        return policy_loss
    
    def count_parameters(self) -> int:
        """Counts trainable parameters of the controller."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
