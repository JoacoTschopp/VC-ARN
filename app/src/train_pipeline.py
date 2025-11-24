import os
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import inspect
import uuid
import json
import pandas as pd
import torch.onnx

# ==============================================================================
# PIPELINE DE ENTRENAMIENTO
# ==============================================================================

class TrainingPipeline:
    """
    Pipeline completo de entrenamiento, validación y evaluación
    Incluye:
    - Detección automática de device (CUDA/MPS/CPU)
    - Entrenamiento con early stopping
    - Sistema de checkpoints
    - Evaluación y métricas
    - Visualizaciones profesionales
    - Guardado de configuraciones y resultados
    """

    def __init__(self, model, config):
        """
        Args:
            model: Modelo de PyTorch (nn.Module)
            config: Dict con configuración (lr, epochs, batch_size, patience, etc.)
        """
        # Detección automática de device
        self.device = self._detect_device()
        print(f"Device detectado: {self.device}")

        # Modelo y configuración
        self.model = model.to(self.device)
        self.config = config

        # Optimizador
        optimizer_type = config.get('optimizer', 'SGD')
        opt_class = getattr(optim, optimizer_type)
        params = inspect.signature(opt_class.__init__).parameters
        opt_kwargs = {"lr": config["lr"]}

        for key, value in config.items():
            if key in params:
                opt_kwargs[key] = value

        self.optimizer = opt_class(self.model.parameters(), **opt_kwargs)

        # Mantenemos solo esta loss por las clases y el label smoothing para pruebas
        self.loss_function = nn.CrossEntropyLoss(
            #https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            label_smoothing=config.get('label_smoothing', 0.0) # Por defecto es 0.0
        ).to(self.device)

        # LR Scheduler (on plateau, pero se podria cambiar para aceptar otros tipos)
        es_patience = self.config.get('es_patience', 10)
        lr_patience = self.config.get('lr_patience', 3)

        if lr_patience > es_patience: # Advertir si el patiencie esta mal configurado
            print(f"Warning: lr_patience ({lr_patience}) > early_stopping_patience ({es_patience})")

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            patience=lr_patience,
            threshold=1e-3,
            threshold_mode='rel',
            cooldown=0,
            min_lr=1e-5,
            ) if self.config['lr_scheduler'] else None

        # Configuración de visualización de plots
        self.show_plots = self.config.get('show_plots', True)
        self.plot_display_time = self.config.get('plot_display_time')

        # Estado del entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.current_epoch = 0

        # Directorio del experimento
        self.experiment_id = str(uuid.uuid4())
        self.experiment_name = config.get("experiment_name", "experimento_sin_nombre")
        self.experiment_dir = os.path.join(
            config.get('base_dir', '.'),
            f"{self.experiment_name}_{self.experiment_id}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Directorio de resultados
        self.results_dir = os.path.join(self.experiment_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Directorio de checkpoints
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Archivo de log de experimentos (JSON Lines), general
        self.experiments_log_path = os.path.join(
            config.get('base_dir', '.'),
            config.get("experiments_log_file", "experiments_log.jsonl")
        )

    def _detect_device(self):
        """Detecta el mejor dispositivo disponible"""
        if torch.cuda.is_available():
            return torch.device('cuda')  # NVIDIA GPU
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')   # Apple Silicon (M1/M2/M3)
        else:
            return torch.device('cpu')   # CPU fallback

    def _train_epoch(self, train_loader):
        """Entrena una época completa"""
        self.model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def _validate_epoch(self, val_loader):
        """Valida una época completa"""
        self.model.eval()
        running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)
                running_loss += loss.item() * data.size(0)

                probs = self.model.final_activation(output)
                preds = probs.argmax(dim=1)
                correct += (preds == target).sum().item()

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / len(val_loader.dataset)
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader):
        """
        Entrenamiento completo con early stopping y checkpoints

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
        """
        print("\n" + "="*70)
        print("ENTRENAMIENTO DEL MODELO")
        print("="*70)
        print(f"Épocas: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['lr']}")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {self.config['es_patience']}")
        print("="*70 + "\n")

        patience_counter = 0

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                self.current_epoch = epoch

                # Entrenar
                train_loss = self._train_epoch(train_loader)
                self.train_losses.append(train_loss)

                # Validar
                val_loss, val_acc = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_acc)

                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_loss)
                    else:
                        # Otros schedulers (StepLR, Cosine, etc.) no necesitan métrica
                        self.lr_scheduler.step()

                # Logging
                print(f"Epoch {epoch:02d} | "
                      f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2%}",
                      end="")

                # Guardar mejor modelo (solo si mejora al menos 0.1 en accuracy)
                if (val_acc - self.best_val_acc) >= 0.001:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    patience_counter = 0
                    self.save_checkpoint('best_model.pth', is_best=True)
                    print(" ✓ MEJOR", end="")
                else:
                    patience_counter += 1

                print()

                # Checkpoint periódico
                if epoch % 5 == 0:
                    self.save_checkpoint('last_checkpoint.pth')
                    print(f"  → Checkpoint guardado")

                # Early stopping
                if patience_counter >= self.config['es_patience']:
                    print(f"\n! Early stopping en época {epoch}")
                    print(f"  No hubo mejora en {self.config['es_patience']} épocas")
                    print(f"  Mejor accuracy: {self.best_val_acc:.2%} (época {self.best_epoch})")
                    break

        except KeyboardInterrupt:
            print("\n" + "="*70)
            print("! ENTRENAMIENTO INTERRUMPIDO")
            print("="*70)
            self.save_checkpoint('interrupted_checkpoint.pth')
            print(f"Estado guardado en: {self.checkpoint_dir}interrupted_checkpoint.pth")
            print("="*70)

        # Resumen final
        print("\n" + "="*70)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("="*70)
        print(f"Épocas completadas: {len(self.train_losses)}")
        print(f"Mejor accuracy de validación: {self.best_val_acc:.2%} (época {self.best_epoch})")
        print(f"Accuracy final: {self.val_metrics[-1]:.2%}")
        print("="*70)

        # Cargar mejor modelo
        self.load_checkpoint('best_model.pth')
        print(f"\n✓ Mejor modelo cargado automáticamente")

    def save_checkpoint(self, filename, is_best=False):
        """Guarda checkpoint del estado actual"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        self._convert_onnx()

    def load_checkpoint(self, filename):
        """Carga checkpoint desde archivo"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            print(f"! Checkpoint no encontrado: {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_metrics = checkpoint['val_metrics']

        print(f"✓ Checkpoint cargado: {filename}")
        print(f"  Época: {self.current_epoch}, Acc: {self.best_val_acc:.2%}")
        return True

    def resume_training(self, checkpoint_file, train_loader, val_loader):
        """Reanuda entrenamiento desde checkpoint"""
        if not self.load_checkpoint(checkpoint_file):
            print("No se puede reanudar el entrenamiento")
            return

        print(f"\nReanudando desde época {self.current_epoch + 1}...")

        # Ajustar configuración para continuar
        remaining_epochs = self.config['epochs'] - self.current_epoch
        if remaining_epochs <= 0:
            print("El entrenamiento ya se completó")
            return

        # Continuar entrenamiento
        original_epochs = self.config['epochs']
        self.train(train_loader, val_loader)

    def evaluate(self, test_loader, dataset_name="Test"):
        """
        Evalúa el modelo en un conjunto de datos

        Args:
            test_loader: DataLoader de test
            dataset_name: Nombre del dataset para logging

        Returns:
            Dict con resultados: accuracy, predictions, labels, probabilities
        """
        print("\n" + "="*70)
        print(f"EVALUACIÓN EN {dataset_name.upper()}")
        print("="*70)

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                probs = self.model.final_activation(output)
                preds = probs.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Calcular accuracy
        accuracy = (all_predictions == all_labels).sum() / len(all_labels)

        print(f"\nAccuracy en {dataset_name}: {accuracy:.2%}")
        print(f"   Correctas: {(all_predictions == all_labels).sum()}/{len(all_labels)}")
        print("="*70)

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

    def register_experiment(
        self,
        train_transforms,
        val_transforms,
        extra_results=None,
        max_model_lines=80,
        max_transforms_lines=100,
        save_markdown=True,
    ):
        """
        Registra el experimento actual en un log (JSONL) y opcionalmente
        genera un resumen en Markdown.

        Args:
            train_transforms, val_transforms:
                Transformaciones (objetos o strings).
            extra_results:
                Dict opcional con métricas adicionales (por ej: test_accuracy,
                f1_score, etc.).
            max_model_lines:
                Máximo de líneas de la arquitectura a guardar.
            max_transforms_lines:
                Máximo de líneas totales de transformaciones.
            save_markdown:
                Si True, guarda un resumen legible en .md
        Returns:
            experiment_record (dict): registro completo guardado en el log.
        """

        # ------------------------------------------------------------------
        # 1) Info de modelo (string recortado)
        # ------------------------------------------------------------------
        model_str = str(self.model)
        model_lines = model_str.splitlines()
        if len(model_lines) > max_model_lines:
            model_lines = model_lines[:max_model_lines] + ["...", "(truncado)"]
        model_str_limited = "\n".join(model_lines)

        # ------------------------------------------------------------------
        # 2) Info de hiperparámetros / optimizer / scheduler
        # ------------------------------------------------------------------
        opt_name = type(self.optimizer).__name__
        opt_state = self.optimizer.state_dict()
        opt_params = opt_state.get('param_groups', [{}])[0]

        optimizer_info = {
            "type": opt_name,
            "params": {
                k: v for k, v in opt_params.items()
                if k in ["lr", "momentum", "weight_decay", "nesterov", "betas", "eps"]
            }
        }

        if self.lr_scheduler is not None:
            sched_name = type(self.lr_scheduler).__name__
            sched_state = self.lr_scheduler.state_dict()
            lr_scheduler_info = {
                "type": sched_name,
                "state": sched_state
            }
        else:
            lr_scheduler_info = None

        # ------------------------------------------------------------------
        # 3) Transformaciones como texto
        # ------------------------------------------------------------------
        def _tf_to_str(name, tf_obj):
            if tf_obj is None:
                return f"{name}: (no especificado)"
            if isinstance(tf_obj, str):
                base = f"{name}:\n{tf_obj}"
            else:
                base = f"{name}:\n{str(tf_obj)}"

            lines = [l for l in base.splitlines()]
            if len(lines) > max_transforms_lines:
                lines = lines[:max_transforms_lines] + ["...", "(truncado)"]
            return "\n".join(lines)

        train_tf_str = _tf_to_str("Train transforms", train_transforms)
        val_tf_str   = _tf_to_str("Validation transforms", val_transforms)

        # ------------------------------------------------------------------
        # 4) Resultados (usando lo que ya tiene el pipeline)
        # ------------------------------------------------------------------
        if len(self.val_metrics) > 0:
            best_val_acc = float(self.best_val_acc)
            final_val_acc = float(self.val_metrics[-1])
        else:
            best_val_acc = None
            final_val_acc = None

        if len(self.val_losses) > 0:
            min_val_loss = float(np.min(self.val_losses))
            final_val_loss = float(self.val_losses[-1])
        else:
            min_val_loss = None
            final_val_loss = None

        results_info = {
            "best_epoch": int(self.best_epoch),
            "best_val_acc": best_val_acc,
            "final_val_acc": final_val_acc,
            "min_val_loss": min_val_loss,
            "final_val_loss": final_val_loss,
            "train_epochs": len(self.train_losses),
        }

        if extra_results is not None:
            results_info.update(extra_results)

        # ------------------------------------------------------------------
        # 5) Armar registro de experimento
        # ------------------------------------------------------------------
        now = datetime.now()
        timestamp = now.isoformat()

        experiment_record = {
            "id": self.experiment_id,
            "timestamp": timestamp,
            "experiment_name": self.experiment_name,
            "device": str(self.device),
            "model_class": self.model.__class__.__name__,
            "model_repr": model_str_limited,
            "config": dict(self.config),
            "optimizer": optimizer_info,
            "lr_scheduler": lr_scheduler_info,
            "train_transforms": train_tf_str,
            "val_transforms": val_tf_str,
            "results": results_info,
        }

        # ------------------------------------------------------------------
        # 6) Guardar en log JSONL
        # ------------------------------------------------------------------
        with open(self.experiments_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_record, ensure_ascii=False) + "\n")

        print(f"✓ Experimento registrado en: {self.experiments_log_path}")
        print(f"  id={self.experiment_id} | nombre={self.experiment_name}")

        # ------------------------------------------------------------------
        # 7) Resumen Markdown opcional
        # ------------------------------------------------------------------
        if save_markdown:
            md_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{self.experiment_name}.md"
            md_path = os.path.join(self.results_dir, md_name)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Resumen de experimento\n\n")
                f.write(f"- **ID**: `{self.experiment_id}`\n")
                f.write(f"- **Nombre**: `{self.experiment_name}`\n")
                f.write(f"- **Fecha**: `{timestamp}`\n")
                f.write(f"- **Device**: `{self.device}`\n\n")

                f.write("## Configuración\n\n")
                for k, v in self.config.items():
                    f.write(f"- **{k}**: `{v}`\n")
                f.write("\n")

                f.write("## Resultados\n\n")
                for k, v in results_info.items():
                    f.write(f"- **{k}**: `{v}`\n")
                f.write("\n")

                f.write("## Optimizer\n\n")
                f.write(f"- Tipo: `{optimizer_info['type']}`\n")
                for k, v in optimizer_info["params"].items():
                    f.write(f"- {k}: `{v}`\n")
                f.write("\n")

                if lr_scheduler_info is not None:
                    f.write("## LR Scheduler\n\n")
                    f.write(f"- Tipo: `{lr_scheduler_info['type']}`\n\n")

                f.write("## Arquitectura del modelo (truncada)\n\n")
                f.write("```text\n")
                f.write(model_str_limited)
                f.write("\n```\n\n")

                f.write("## Transformaciones\n\n")
                f.write("```text\n")
                f.write(train_tf_str)
                f.write("\n\n")
                f.write(val_tf_str)
                f.write("\n```\n")

            print(f"✓ Resumen Markdown guardado en: {md_path}")

        return experiment_record

    def summarize_experiments(self, sort_by="results.best_val_acc", top_k=10):
        """
        Lee el log de experimentos y muestra un resumen tabular básico.

        Args:
            sort_by: clave para ordenar (ej: 'results.best_val_acc').
            top_k: cantidad de filas a mostrar.
        """

        if not os.path.exists(self.experiments_log_path):
            print(f"! No existe log de experimentos en {self.experiments_log_path}")
            return None

        records = []
        with open(self.experiments_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)

        if not records:
            print("! Log vacío")
            return None

        # Aplanar algunas columnas útiles
        flat = []
        for r in records:
            row = {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "name": r.get("experiment_name", ""),
                "device": r.get("device", ""),
                "epochs": r["results"].get("train_epochs"),
                "best_epoch": r["results"].get("best_epoch"),
                "best_val_acc": r["results"].get("best_val_acc"),
                "final_val_acc": r["results"].get("final_val_acc"),
                "min_val_loss": r["results"].get("min_val_loss"),
                "final_val_loss": r["results"].get("final_val_loss"),
                "lr": r["optimizer"]["params"].get("lr"),
                "optimizer": r["optimizer"]["type"],
            }
            flat.append(row)

        df = pd.DataFrame(flat)

        # Ordenar
        if sort_by == "results.best_val_acc":
            df = df.sort_values("best_val_acc", ascending=False)
        elif sort_by == "results.final_val_acc":
            df = df.sort_values("final_val_acc", ascending=False)

        print("\n=== RESUMEN DE EXPERIMENTOS ===")
        print(df.head(top_k).to_string(index=False))
        return df

    def plot_training_curves(self):
        """Genera gráficos de curvas de entrenamiento"""
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)

        colors = sns.color_palette("husl", 3)

        print("\n" + "="*70)
        print("CURVAS DE APRENDIZAJE")
        print("="*70)
        print(f"Épocas: {len(self.train_losses)}")
        print(f"Mejor accuracy: {max(self.val_metrics):.2%} (época {np.argmax(self.val_metrics)+1})")
        print(f"Overfitting gap: {self.val_losses[-1] - self.train_losses[-1]:.4f}")
        print("="*70)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = np.arange(1, len(self.train_losses) + 1)

        # Loss
        axes[0].plot(epochs, self.train_losses, color=colors[0], linewidth=2.5,
                    label='Train', alpha=0.8)
        axes[0].plot(epochs, self.val_losses, color=colors[1], linewidth=2.5,
                    label='Validation', alpha=0.8)
        best_epoch = np.argmin(self.val_losses)
        axes[0].scatter(best_epoch + 1, self.val_losses[best_epoch],
                       color='green', s=150, marker='*', zorder=5)
        axes[0].set_xlabel('Época', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Evolución de la Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, np.array(self.val_metrics) * 100, color=colors[1],
                    linewidth=2.5, alpha=0.8)
        best_acc_epoch = np.argmax(self.val_metrics)
        axes[1].scatter(best_acc_epoch + 1, self.val_metrics[best_acc_epoch] * 100,
                       color='green', s=150, marker='*', zorder=5)
        axes[1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Baseline')
        axes[1].set_xlabel('Época', fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[1].set_title('Evolución del Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)

        # Overfitting gap
        gap = np.array(self.val_losses) - np.array(self.train_losses)
        axes[2].plot(epochs, gap, color='purple', linewidth=2.5, alpha=0.8)
        axes[2].fill_between(epochs, 0, gap, color='purple', alpha=0.2)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[2].set_xlabel('Época', fontweight='bold')
        axes[2].set_ylabel('Gap (Val - Train)', fontweight='bold')
        axes[2].set_title('Medida de Overfitting', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        if gap[-1] > 0.5:
            axes[2].text(0.5, 0.95, '! OVERFITTING', transform=axes[2].transAxes,
                        fontsize=11, color='red', fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        fig.tight_layout()
        self._finalize_plot(fig, "training_curves.png")

    def plot_confusion_matrix(self, predictions, labels, class_names):
        """Genera matriz de confusión"""
        cm = confusion_matrix(labels, predictions)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicción', fontweight='bold')
        ax.set_ylabel('Real', fontweight='bold')
        ax.set_title('Matriz de Confusión', fontweight='bold', pad=15)
        fig.tight_layout()
        self._finalize_plot(fig, "confusion_matrix.png")

    def plot_examples(self, images, predictions, labels, class_names,
                     mean, std, n_correct=10, n_incorrect=10):
        """Muestra ejemplos de predicciones correctas e incorrectas"""

        def denormalize(img, mean, std):
            img_denorm = img.copy()
            for c in range(3):
                img_denorm[c] = img_denorm[c] * std[c] + mean[c]
            img_denorm = np.clip(img_denorm, 0, 1)
            return np.transpose(img_denorm, (1, 2, 0))

        correct_mask = (predictions == labels)
        correct_idx = np.where(correct_mask)[0]
        incorrect_idx = np.where(~correct_mask)[0]

        # Ejemplos correctos
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Predicciones CORRECTAS', fontsize=16, fontweight='bold')

        sample_correct = np.random.choice(correct_idx,
                                         size=min(n_correct, len(correct_idx)),
                                         replace=False)

        for idx, ax in enumerate(axes.flat):
            if idx < len(sample_correct):
                i = sample_correct[idx]
                img = denormalize(images[i], mean, std)
                ax.imshow(img)
                ax.set_title(f'Real: {class_names[labels[i]]}\n'
                           f'Pred: {class_names[predictions[i]]}',
                           color='green', fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')

        fig.tight_layout()
        self._finalize_plot(fig, "correct_examples.png")

        # Ejemplos incorrectos
        if len(incorrect_idx) > 0:
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle('Predicciones INCORRECTAS', fontsize=16, fontweight='bold')

            sample_incorrect = np.random.choice(incorrect_idx,
                                               size=min(n_incorrect, len(incorrect_idx)),
                                               replace=False)

            for idx, ax in enumerate(axes.flat):
                if idx < len(sample_incorrect):
                    i = sample_incorrect[idx]
                    img = denormalize(images[i], mean, std)
                    ax.imshow(img)
                    ax.set_title(f'Real: {class_names[labels[i]]}\n'
                               f'Pred: {class_names[predictions[i]]}',
                               color='red', fontsize=9)
                    ax.axis('off')
                else:
                    ax.axis('off')

            fig.tight_layout()
            self._finalize_plot(fig, "incorrect_examples.png")

    def _finalize_plot(self, fig, filename):
        """Guarda la figura y la muestra según la configuración."""
        filepath = os.path.join(self.results_dir, filename)
        fig.savefig(filepath, bbox_inches='tight')

        if not self.show_plots:
            plt.close(fig)
            return

        if self.plot_display_time is not None and self.plot_display_time > 0:
            plt.show(block=False)
            plt.pause(self.plot_display_time)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
    
    # Function to Convert to ONNX 
    def _convert_onnx(self): 

        # Seteamos el modelo en modo inferencia
        self.model.eval() 

        # Creamos un tensor de entrada dummy en el mismo device que el modelo
        dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True, device=self.device)

        # Donde guardamos el modelo
        model_path = os.path.join(self.results_dir, "ImageClassifier.onnx")

        # Exportamos el modelo   
        torch.onnx.export(self.model,         # modelo a exportar 
         dummy_input,       # input (or a tuple for multiple inputs) 
         model_path,       # donde guardamos el modelo  
         export_params=True,  # guardar los parámetros del modelo 
         opset_version=10,    # version de ONNX 
         do_constant_folding=True,  # optimización 
         input_names = ['modelInput'],   # nombres de los inputs 
         output_names = ['modelOutput'], # nombres de los outputs 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # axes con longitud variable 
                                'modelOutput' : {0 : 'batch_size'}}) 


print("✓ Clase TrainingPipeline cargada exitosamente")
