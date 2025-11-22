"""
NAS Trainer: Architecture search orchestrator.

Integrates Controller + Child Networks + REINFORCE to execute NAS.
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .controller import NASController
from .child_builder import ChildNetworkBuilder
from .reinforce import REINFORCEOptimizer
from .utils import decode_dna, dna_to_string
from ..train_pipeline import TrainingPipeline


class NASTrainer:
    """Neural Architecture Search orchestrator."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary containing NAS configuration
        """
        self.config = config
        self.device = self._detect_device()
        
        # Controller
        self.controller = NASController(
            num_layers=config.get('max_layers', 3),
            components_per_layer=config.get('components_per_layer', 4),
            hidden_size=config.get('lstm_hidden_size', 100),
            device=self.device
        )
        
        # REINFORCE optimizer
        self.reinforce = REINFORCEOptimizer(
            self.controller,
            lr=config.get('controller_lr', 0.99),
            decay=config.get('lr_decay', 0.96),
            decay_steps=config.get('lr_decay_steps', 500),
            beta=config.get('beta', 1e-4),
            ema_alpha=config.get('baseline_ema_alpha', 0.95)
        )
        
        # History trackers
        self.reward_history = []
        self.architecture_history = []
        self.best_architecture = None
        self.best_reward = 0.0
        self.episode_count = 0
        
        # Paths
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/nas'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.get('log_dir', 'logs/nas'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"nas_search_{timestamp}.log"
        
        self._log(f"NAS Trainer initialized on device: {self.device}")
        self._log(f"Controller parameters: {self.controller.count_parameters():,}")
    
    def _detect_device(self):
        """Detects the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _log(self, message: str, level: str = 'INFO'):
        """
        Writes a formatted log message to console and file.
        
        Args:
            message: Message to log
            level: Log level (INFO, SUCCESS, STEP, METRIC, ERROR, ...)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prefix per log level
        prefixes = {
            'INFO': '📋',
            'SUCCESS': '✅',
            'STEP': '🔹',
            'METRIC': '📊',
            'ERROR': '❌',
            'ARCHITECTURE': '🏗️',
            'TRAINING': '🎯',
            'REWARD': '🏆',
        }
        
        prefix = prefixes.get(level, '  ')
        log_message = f"[{timestamp}] {prefix} {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def _train_child(
        self,
        dna: np.ndarray,
        child_id: str,
        train_loader,
        val_loader
    ) -> float:
        """
        Trains a child network and returns its reward.
        
        Args:
            dna: Architecture DNA
            child_id: Identifier for logging
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        
        Returns:
            reward: Validation accuracy
        """
        try:
            # Build child model
            model = ChildNetworkBuilder.build_from_dna(
                dna,
                dropout_rate=self.config.get('child_dropout', 0.2)
            )
            
            # Model info for logging
            model_info = ChildNetworkBuilder.get_model_info(model)
            self._log(
                f"Child {child_id} - Architecture built: "
                f"{model_info['total_params']:,} parameters, "
                f"{model_info['num_conv_layers']} conv layers, "
                f"{model_info['size_mb']:.2f} MB",
                level='ARCHITECTURE'
            )
            
            # Training configuration
            child_config = {
                'experiment_name': f'nas_child_{child_id}',
                'lr': self.config.get('child_lr', 3e-5),
                'epochs': self.config.get('child_epochs', 100),
                'batch_size': self.config.get('child_batch_size', 20),
                'optimizer': 'Adam',
                'es_patience': 10,
                'use_scheduler': True,
                'lr_patience': 5,
                'checkpoint_dir': str(self.checkpoint_dir / 'children' / child_id),
                'experiment_dir': str(self.checkpoint_dir / 'children' / child_id),
                'show_plots': False,
            }
            
            # Train child network
            pipeline = TrainingPipeline(model, child_config)
            pipeline.train(train_loader, val_loader)
            
            # Retrieve best validation accuracy
            reward = pipeline.best_val_acc
            
            self._log(
                f"Child {child_id} - Training completed: "
                f"Best Val Acc = {reward:.4f}",
                level='REWARD'
            )
            
            return reward
            
        except Exception as e:
            self._log(
                f"Child {child_id} - ERROR during training: {str(e)}",
                level='ERROR'
            )
            return 0.0  # Negative reward for invalid DNAs
    
    def search(
        self,
        train_loader,
        val_loader,
        num_episodes: int = None,
        children_per_episode: int = None
    ):
        """
        Runs the architecture search loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_episodes: Number of episodes (overrides config)
            children_per_episode: Children evaluated per episode (overrides config)
        """
        num_episodes = num_episodes or self.config.get('max_episodes', 2000)
        children_per_episode = children_per_episode or self.config.get('children_per_episode', 10)
        save_every = self.config.get('save_every', 50)
        
        self._log("=" * 70, level='INFO')
        self._log("STARTING NEURAL ARCHITECTURE SEARCH", level='INFO')
        self._log("=" * 70, level='INFO')
        self._log("")
        self._log("SEARCH CONFIGURATION:", level='STEP')
        self._log(f"  • Total episodes: {num_episodes}")
        self._log(f"  • Architectures per episode: {children_per_episode}")
        self._log(f"  • Total architectures to evaluate: {num_episodes * children_per_episode:,}")
        self._log(f"  • Compute device: {self.device}")
        self._log(f"  • Checkpoint frequency: every {save_every} episodes")
        self._log(f"  • Layers per architecture: {self.config.get('max_layers', 3)}")
        self._log(f"  • Training epochs per child: {self.config.get('child_epochs', 100)}")
        self._log("")
        self._log("PROCESS OVERVIEW:", level='STEP')
        self._log("  1. Controller (LSTM) generates architecture DNA")
        self._log("  2. DNA is decoded into a CNN architecture")
        self._log("  3. Child Network is trained on CIFAR-10")
        self._log("  4. Validation accuracy becomes the reward")
        self._log("  5. Controller is updated with REINFORCE")
        self._log("=" * 70, level='INFO')
        
        for episode in range(num_episodes):
            self.episode_count = episode + 1
            episode_rewards = []
            episode_architectures = []
            
            self._log("")
            self._log(f"━━━ EPISODE {self.episode_count}/{num_episodes} ━━━", level='STEP')
            self._log(f"Generating and evaluating {children_per_episode} architectures...")
            
            # Generar y evaluar children
            for child_idx in range(children_per_episode):
                # Sample architecture from the controller
                dna_flat = self.controller.sample(num_samples=1)[0]
                dna = decode_dna(
                    dna_flat.tolist(), 
                    components_per_layer=self.config.get('components_per_layer', 4)
                )
                
                # Train child
                child_id = f"ep{self.episode_count}_child{child_idx + 1}"
                self._log("")
                self._log(
                    f"→ Evaluating architecture {child_idx + 1}/{children_per_episode}",
                    level='TRAINING'
                )
                self._log(f"DNA: {dna.flatten().tolist()}")
                reward = self._train_child(dna, child_id, train_loader, val_loader)
                
                episode_rewards.append(reward)
                episode_architectures.append(dna_flat)
                
                # Actualizar best
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_architecture = dna
                    self._log("")
                    self._log(
                        f"NEW BEST ARCHITECTURE FOUND! Reward: {reward:.4f}",
                        level='SUCCESS'
                    )
                    self._log(dna_to_string(dna))
                    self._log("This architecture outperforms all previous ones.")
                    
                    # Guardar mejor arquitectura
                    self._save_best_architecture()
            
            # Update controller with REINFORCE
            architectures_tensor = torch.tensor(
                np.vstack(episode_architectures),
                dtype=torch.float32,
                device=self.device
            )
            
            metrics = self.reinforce.step(architectures_tensor, episode_rewards)
            
            # Episode metrics logging
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            self.reward_history.append(mean_reward)
            
            self._log("")
            self._log(f"━━━ EPISODE {self.episode_count} SUMMARY ━━━", level='METRIC')
            self._log(f"  • Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
            self._log(f"  • Best child this episode: {max(episode_rewards):.4f}")
            self._log(f"  • Worst child this episode: {min(episode_rewards):.4f}")
            self._log(f"  • Baseline EMA: {metrics['baseline']:.4f}")
            self._log(f"  • Global best architecture: {self.best_reward:.4f}")
            self._log(f"  • Controller learning rate: {metrics['lr']:.6f}")
            self._log(f"  • Mean advantage: {metrics['advantage']:.4f}")
            
            # Progress indicator
            progress_pct = (self.episode_count / num_episodes) * 100
            self._log(f"\n  Progress: {self.episode_count}/{num_episodes} ({progress_pct:.1f}%)")
            
            # Periodic checkpointing
            if self.episode_count % save_every == 0:
                self._log("")
                self._log(f"Saving checkpoint for episode {self.episode_count}...", level='STEP')
                self.save_checkpoint(f"nas_episode_{self.episode_count}.pth")
        
        self._log("")
        self._log("=" * 70, level='SUCCESS')
        self._log("NEURAL ARCHITECTURE SEARCH COMPLETED", level='SUCCESS')
        self._log("=" * 70, level='SUCCESS')
        self._log("")
        self._log("FINAL RESULTS:", level='METRIC')
        self._log(f"  • Best reward (Val Acc): {self.best_reward:.4f}")
        self._log(f"  • Total episodes: {self.episode_count}")
        self._log(f"  • Total architectures evaluated: {self.episode_count * children_per_episode}")
        self._log(f"  • Mean reward (last 10): {np.mean(self.reward_history[-10:]):.4f}")
        self._log(f"  • Mean reward (overall): {np.mean(self.reward_history):.4f}")
        self._log("")
        if self.best_architecture is not None:
            self._log("BEST ARCHITECTURE FOUND:", level='SUCCESS')
            self._log(dna_to_string(self.best_architecture))
        self._log("=" * 70, level='SUCCESS')
        
        # Save final checkpoint
        self.save_checkpoint("nas_final.pth")
    
    def save_checkpoint(self, filename: str):
        """Saves a NAS checkpoint."""
        checkpoint = {
            'episode': self.episode_count,
            'controller_state_dict': self.controller.state_dict(),
            'optimizer_state_dict': self.reinforce.optimizer.state_dict(),
            'scheduler_state_dict': self.reinforce.scheduler.state_dict(),
            'baseline': self.reinforce.baseline,
            'reward_history': self.reward_history,
            'architecture_history': self.architecture_history,
            'best_architecture': self.best_architecture.tolist() if self.best_architecture is not None else None,
            'best_reward': self.best_reward,
            'config': self.config,
            'reinforce_stats': self.reinforce.get_stats(),
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        self._log(f"Checkpoint saved successfully: {path}", level='SUCCESS')
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resumes NAS search from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.episode_count = checkpoint.get('episode', 0)
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.reinforce.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.reinforce.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.reinforce.baseline = checkpoint.get('baseline', 0.0)
        self.reward_history = checkpoint.get('reward_history', [])
        self.architecture_history = checkpoint.get('architecture_history', [])
        
        best_arch_list = checkpoint.get('best_architecture')
        if best_arch_list is not None:
            self.best_architecture = np.array(best_arch_list).reshape(-1, 4)
        
        self.best_reward = checkpoint.get('best_reward', 0.0)
        
        self._log("")
        self._log(f"Checkpoint loaded successfully: {checkpoint_path}", level='SUCCESS')
        self._log("RESTORED STATE:", level='INFO')
        self._log(f"  • Episodes already completed: {self.episode_count}")
        self._log(f"  • Best reward found: {self.best_reward:.4f}")
        self._log(f"  • Current baseline: {self.reinforce.baseline:.4f}")
        self._log(f"  • Architectures logged: {len(self.architecture_history)}")
        self._log("")
    
    def _save_best_architecture(self):
        """Persists the best architecture found so far."""
        if self.best_architecture is None:
            return
        
        best_arch_path = self.checkpoint_dir / "best_architecture.json"
        
        arch_data = {
            'dna': self.best_architecture.tolist(),
            'reward': float(self.best_reward),
            'episode': self.episode_count,
            'timestamp': datetime.now().isoformat(),
            'description': dna_to_string(self.best_architecture),
        }
        
        with open(best_arch_path, 'w') as f:
            json.dump(arch_data, f, indent=2)
        
        self._log(f"Best architecture saved: {best_arch_path}", level='SUCCESS')
