"""
REINFORCE: Policy Gradient optimizer for the NAS Controller.

Implements the REINFORCE algorithm (Williams, 1992) with EMA baseline
to train the controller to generate high-reward architectures.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Dict


class REINFORCEOptimizer:
    """
    REINFORCE optimizer with EMA baseline.
    
    The controller is trained to maximize the expected reward (validation accuracy).
    """
    
    def __init__(
        self,
        controller,
        lr: float = 0.99,
        decay: float = 0.96,
        decay_steps: int = 500,
        beta: float = 1e-4,
        ema_alpha: float = 0.95
    ):
        """
        Args:
            controller: NASController instance
            lr: Initial learning rate
            decay: Decay factor
            decay_steps: Steps between decay applications
            beta: L2 regularization weight
            ema_alpha: Baseline EMA smoothing factor
        """
        self.controller = controller
        self.beta = beta
        self.ema_alpha = ema_alpha
        
        # Optimizer
        self.optimizer = optim.RMSprop(
            controller.parameters(),
            lr=lr
        )
        
        # LR Scheduler (exponential decay every decay_steps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=decay_steps,
            gamma=decay
        )
        
        # Baseline (reward EMA)
        self.baseline = 0.0
        self.reward_history = []
        self.step_count = 0
    
    def update_baseline(self, reward: float):
        """
        Updates baseline using Exponential Moving Average.
        
        Args:
            reward: Current reward
        """
        if len(self.reward_history) == 0:
            self.baseline = reward
        else:
            self.baseline = (
                self.ema_alpha * self.baseline +
                (1 - self.ema_alpha) * reward
            )
        
        self.reward_history.append(reward)
    
    def step(
        self,
        architectures: torch.Tensor,
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        Updates the controller via REINFORCE.
        
        Args:
            architectures: [batch, num_outputs] generated architectures
            rewards: List of rewards (validation accuracies)
        
        Returns:
            Dict containing update metrics
        """
        self.optimizer.zero_grad()
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=architectures.device)
        
        # Compute advantage (reward - baseline)
        mean_reward = float(np.mean(rewards))
        self.update_baseline(mean_reward)
        advantages = rewards_tensor - self.baseline
        
        # Normalize advantages to improve stability
        if len(rewards) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = self.controller.get_policy_loss(architectures, advantages)
        
        # L2 regularization
        l2_loss = sum(
            torch.sum(param ** 2)
            for param in self.controller.parameters()
        )
        total_loss = policy_loss + self.beta * l2_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping (avoid exploding gradients)
        torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(),
            max_norm=1.0
        )
        
        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        
        self.step_count += 1
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'l2_loss': (self.beta * l2_loss).item(),
            'reward': mean_reward,
            'baseline': self.baseline,
            'advantage': float(advantages.mean().item()) if len(rewards) > 1 else 0.0,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Returns optimizer statistics."""
        return {
            'baseline': self.baseline,
            'steps': self.step_count,
            'lr': self.optimizer.param_groups[0]['lr'],
            'mean_reward_last_10': float(np.mean(self.reward_history[-10:])) if self.reward_history else 0.0,
            'mean_reward_all': float(np.mean(self.reward_history)) if self.reward_history else 0.0,
        }
