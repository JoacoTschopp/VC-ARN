# Plan de Refactorización: Branch NASCNN15

**Proyecto:** VC-ARN  
**Objetivo:** Consolidar proyecto en arquitectura única NASCNN15 + integrar NAS con RL  
**Fecha inicio:** 21 Noviembre 2025  
**Duración estimada:** 5 semanas  

---

## 🎯 Objetivos del Branch

### Objetivos Principales

1. ✅ **Arquitectura única**: Mantener solo NASCNN15, eliminar BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR
2. ✅ **Dataset único**: Eliminar todas las referencias a CIFAR-10.1, usar solo CIFAR-10
3. ✅ **NAS integrado**: Migrar código de `Neural-Architecture-Search-using-Reinforcement-Learning/` a `app/src/nas/`
4. ✅ **Checkpoint support**: Implementar capacidad de reanudar búsqueda NAS desde cualquier episodio

### Métricas de Éxito

- ✓ Código compila y ejecuta sin errores
- ✓ NASCNN15 alcanza 91%+ accuracy en CIFAR-10 test
- ✓ NAS descubre arquitecturas comparables
- ✓ Checkpoints funcionan correctamente
- ✓ Documentación completa actualizada

---

## 📋 Sprint 1: Limpieza del Código Base

**Duración:** 1 semana (22-28 Nov)  
**Prioridad:** 🔴 Alta  

### Objetivos

- Eliminar arquitecturas no utilizadas
- Remover todas las referencias a CIFAR-10.1
- Código base limpio y funcional con solo NASCNN15

### Tareas Detalladas

#### 1.1 Preparación

```bash
# Crear rama de trabajo
git checkout -b refactor/nascnn15-only

# Backup del estado actual
git tag backup-pre-refactor
```

**Checklist:**
- [ ] Branch creada
- [ ] Tag de backup creado
- [ ] Entorno virtual activado

---

#### 1.2 Limpiar app/src/arqui_cnn.py

**Archivo:** `app/src/arqui_cnn.py`  
**Estado actual:** 494 líneas con 5 arquitecturas  
**Estado objetivo:** ~150 líneas con solo NASCNN15  

**Acción:**
```python
# ELIMINAR (líneas 1-303):
# - class BaseModel(nn.Module)
# - class SimpleCNN(nn.Module)
# - class ImprovedCNN(nn.Module)
# - class ResidualBlock(nn.Module)
# - class ResNetCIFAR(nn.Module)

# MANTENER (líneas 313-494):
# - class NASCNN15(nn.Module)
```

**Checklist:**
- [ ] Eliminar clases no utilizadas
- [ ] Verificar que NASCNN15 está completa
- [ ] Actualizar docstring del archivo
- [ ] Verificar imports necesarios

**Comando de validación:**
```bash
python -c "from app.src.arqui_cnn import NASCNN15; print('✓ NASCNN15 importada correctamente')"
```

---

#### 1.3 Limpiar app/src/load.py

**Archivo:** `app/src/load.py`  
**Estado actual:** 204 líneas con soporte CIFAR-10 + CIFAR-10.1  
**Estado objetivo:** ~120 líneas solo CIFAR-10  

**Eliminar:**
```python
# Líneas 15-33: class Cifar101Dataset(Dataset)
# Líneas 37-77: load_data() - descarga CIFAR-10.1
# Líneas 160-204: load_cifar101() - loader CIFAR-10.1
```

**Mantener:**
```python
# Líneas 80-157: load_cifar10() - Función principal
```

**Actualizar docstrings:**
```python
def load_cifar10(...):
    """
    Carga CIFAR-10 con split estratificado.
    
    Split según paper Zoph & Le (2017):
    - 45,000 imágenes de entrenamiento
    - 5,000 imágenes de validación (estratificado)
    - 10,000 imágenes de test
    
    Args:
        datasets_folder: Ruta a carpeta de datasets
        config: Configuración de transformaciones
    
    Returns:
        train_dataset: Subset con 45,000 imágenes
        val_dataset: Subset con 5,000 imágenes
        test_dataset: Dataset con 10,000 imágenes
        training_transformations: Transforms para train
        test_transformations: Transforms para val/test
    """
```

**Checklist:**
- [ ] Eliminar class Cifar101Dataset
- [ ] Eliminar función load_data()
- [ ] Eliminar función load_cifar101()
- [ ] Actualizar imports si es necesario
- [ ] Verificar que load_cifar10() funciona

**Comando de validación:**
```bash
python -c "
from app.src.load import load_cifar10
from app.src.pre_processed import TransformConfig
train, val, test, _, _ = load_cifar10('../datasets', TransformConfig())
print(f'✓ Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')
"
```

---

#### 1.4 Actualizar app/main.py

**Archivo:** `app/main.py`  
**Líneas a modificar:** 7, 49, 151-155  

**Cambios:**

```python
# Línea 7: Actualizar imports
# ANTES:
from src.arqui_cnn import BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR, NASCNN15

# DESPUÉS:
from src.arqui_cnn import NASCNN15

# Línea 49: Actualizar nombre de experimento
# ANTES:
experiment_name = "NASCNN_V13_OnlyCIFAR10"

# DESPUÉS:
experiment_name = "NASCNN15_Production"

# Líneas 151-155: Eliminar arquitecturas comentadas
# ELIMINAR:
#model = BaseModel()
#model = SimpleCNN()
#model = ImprovedCNN()
#model = ResNetCIFAR()

# MANTENER:
model = NASCNN15()
```

**Checklist:**
- [ ] Actualizar import statement
- [ ] Cambiar nombre de experimento
- [ ] Eliminar líneas comentadas
- [ ] Verificar que main() ejecuta

**Comando de validación:**
```bash
python app/main.py --dry-run  # Si implementamos este flag
```

---

#### 1.5 Actualizar app/src/auxiliares.py

**Archivo:** `app/src/auxiliares.py`  
**Función a modificar:** `compare_models()`  

**Estado actual:**
```python
def compare_models():
    """Compara todas las arquitecturas disponibles."""
    models = {
        'BaseModel': BaseModel(),
        'SimpleCNN': SimpleCNN(),
        'ImprovedCNN': ImprovedCNN(),
        'ResNetCIFAR': ResNetCIFAR(),
        'NASCNN15': NASCNN15()
    }
    # ... imprime comparación
```

**Estado objetivo:**
```python
def show_nascnn15_info():
    """Muestra información detallada de NASCNN15."""
    model = NASCNN15()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 70)
    print("ARQUITECTURA: NASCNN15")
    print("=" * 70)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Tamaño estimado: {total_params * 4 / (1024**2):.2f} MB")
    print()
    print("Características:")
    print("  - 15 capas convolucionales con skip connections")
    print("  - Kernels: 1x1, 3x3, 3x7, 5x5, 5x7, 7x1, 7x3, 7x5, 7x7")
    print("  - Filtros: 36 o 48 por capa")
    print("  - Accuracy esperado: 91%+ en CIFAR-10 test")
    print("=" * 70)
```

**Actualizar llamada en main.py:**
```python
# ANTES:
compare_models()

# DESPUÉS:
show_nascnn15_info()
```

**Checklist:**
- [ ] Renombrar función
- [ ] Eliminar modelos no usados
- [ ] Actualizar output
- [ ] Actualizar llamada en main.py

---

#### 1.6 Actualizar app/src/test.py

**Archivo:** `app/src/test.py`  
**Verificar función:** `run_cifar10_test_evaluation()`  

**Acción:**
- Verificar que NO hay función `run_cifar101_evaluation()`
- Si existe, eliminarla
- Asegurar que solo se evalúa en CIFAR-10 test set

**Checklist:**
- [ ] Verificar funciones en test.py
- [ ] Eliminar referencias a CIFAR-10.1 si existen
- [ ] Validar que run_cifar10_test_evaluation() funciona

---

#### 1.7 Actualizar Documentación

**Archivos a modificar:**
- `README.md` (raíz)
- `app/README.md`
- `app/src/README.md` (si existe)

**README.md principal:**
```markdown
# VC-ARN: Neural Architecture Search con NASCNN15

## Descripción

Implementación de NASCNN15, arquitectura de 15 capas descubierta mediante 
Neural Architecture Search con Reinforcement Learning (Zoph & Le, 2017).

## Características

- **Arquitectura**: NASCNN15 (15 capas convolucionales + skip connections)
- **Dataset**: CIFAR-10 (45k train / 5k val / 10k test)
- **Framework**: PyTorch 2.0+
- **Hardware**: CUDA / Apple Silicon (MPS) / CPU

## Quick Start

```bash
# Activar entorno
source .venv/bin/activate

# Entrenar NASCNN15
cd app
python main.py
```

## Estructura

```
VC-ARN/
├── app/
│   ├── main.py              # Punto de entrada
│   └── src/
│       ├── arqui_cnn.py     # NASCNN15 architecture
│       ├── load.py          # CIFAR-10 data loading
│       ├── train_pipeline.py # Training orchestration
│       └── ...
├── experiments/             # Training results
├── datasets/                # CIFAR-10 data
└── Monografia_NASCNN.md     # Documentación completa
```

## Accuracy Esperado

- Training: ~99%
- Validation: 91-93%
- Test: 91.5%

## Referencias

- Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. ICLR.
```

**Checklist:**
- [ ] Actualizar README.md principal
- [ ] Actualizar app/README.md
- [ ] Eliminar menciones a arquitecturas removidas
- [ ] Eliminar menciones a CIFAR-10.1

---

#### 1.8 Tests de Regresión

**Objetivo:** Verificar que el código refactorizado funciona correctamente

**Tests manuales:**

```bash
# 1. Importar arquitectura
python -c "from app.src.arqui_cnn import NASCNN15; print('✓ Import OK')"

# 2. Cargar datos
python -c "
from app.src.load import load_cifar10
from app.src.pre_processed import TransformConfig
train, val, test, _, _ = load_cifar10('../datasets', TransformConfig())
assert len(train) == 45000, 'Train size incorrect'
assert len(val) == 5000, 'Val size incorrect'
assert len(test) == 10000, 'Test size incorrect'
print('✓ Data loading OK')
"

# 3. Forward pass
python -c "
import torch
from app.src.arqui_cnn import NASCNN15
model = NASCNN15()
x = torch.randn(2, 3, 32, 32)
out = model(x)
assert out.shape == (2, 10), f'Output shape incorrect: {out.shape}'
print('✓ Forward pass OK')
"

# 4. Training step
python -c "
import torch
from app.src.arqui_cnn import NASCNN15
model = NASCNN15()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()

x = torch.randn(4, 3, 32, 32)
y = torch.randint(0, 10, (4,))

optimizer.zero_grad()
out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()
print('✓ Training step OK')
"
```

**Checklist:**
- [ ] Test 1: Import pasa
- [ ] Test 2: Data loading pasa
- [ ] Test 3: Forward pass pasa
- [ ] Test 4: Training step pasa

---

#### 1.9 Commit y Validación Final

**Commits sugeridos:**

```bash
# Commit 1: Limpiar arquitecturas
git add app/src/arqui_cnn.py
git commit -m "refactor: remove unused architectures, keep only NASCNN15"

# Commit 2: Limpiar data loading
git add app/src/load.py
git commit -m "refactor: remove CIFAR-10.1 support, keep only CIFAR-10"

# Commit 3: Actualizar main
git add app/main.py app/src/auxiliares.py
git commit -m "refactor: update main.py and auxiliares for NASCNN15 only"

# Commit 4: Actualizar documentación
git add README.md app/README.md
git commit -m "docs: update README files for NASCNN15-only branch"

# Commit 5: Tests
git add tests/
git commit -m "test: add regression tests for refactored code"
```

**Validación final:**

```bash
# Ejecutar entrenamiento de prueba (5 épocas)
cd app
python main.py --epochs 5 --experiment test_refactor
```

**Criterios de éxito Sprint 1:**
- ✓ Código compila sin errores
- ✓ NASCNN15 entrena correctamente
- ✓ No hay referencias a CIFAR-10.1
- ✓ No hay referencias a arquitecturas eliminadas
- ✓ Tests de regresión pasan
- ✓ Documentación actualizada

---

## 📋 Sprint 2: Migración NAS con RL

**Duración:** 2 semanas (29 Nov - 12 Dic)  
**Prioridad:** 🟡 Media  

### Objetivos

- Migrar NAS de TensorFlow 1.x a PyTorch
- Integrar con TrainingPipeline existente
- Implementar módulo `app/src/nas/`

### Estructura del Módulo NAS

```
app/src/nas/
├── __init__.py              # Exports principales
├── controller.py            # NASController (LSTM)
├── child_builder.py         # ChildNetworkBuilder
├── trainer.py               # NASTrainer (orchestrator)
├── reinforce.py             # REINFORCE optimizer
├── utils.py                 # DNA encoding/decoding
└── configs.py               # Configuración NAS
```


### Tareas Detalladas Sprint 2

#### 2.1 Crear estructura del módulo

```bash
mkdir -p app/src/nas
touch app/src/nas/__init__.py
touch app/src/nas/controller.py
touch app/src/nas/child_builder.py
touch app/src/nas/trainer.py
touch app/src/nas/reinforce.py
touch app/src/nas/utils.py
touch app/src/nas/configs.py
```

**Checklist:**
- [ ] Directorio creado
- [ ] Archivos inicializados
- [ ] __init__.py configurado

---

#### 2.2 Implementar controller.py

**Archivo:** `app/src/nas/controller.py`

```python
"""
NAS Controller: LSTM que genera arquitecturas de redes neuronales.

El Controller es un modelo de Reinforcement Learning que aprende a generar
arquitecturas CNN prometedoras maximizando la validation accuracy.
"""

import torch
import torch.nn as nn

class NASController(nn.Module):
    """
    Controller basado en LSTM para Neural Architecture Search.
    
    Genera arquitecturas codificadas como DNA:
        DNA = [[kernel, filters, stride, pool], ...]
    
    Args:
        num_layers: Número de capas a generar
        components_per_layer: Componentes por capa (default: 4)
        hidden_size: Tamaño del LSTM hidden state
        device: Device para el modelo
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
        
        # LSTM para generar secuencia
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # FC para mapear hidden state -> DNA
        self.fc = nn.Linear(hidden_size, self.num_outputs)
        
        # Inicializar bias a 0.01 (como en paper)
        self.fc.bias.data.fill_(0.01)
        
        self.to(device)
    
    def forward(self, prev_architecture: torch.Tensor):
        """
        Genera nueva arquitectura basada en la anterior.
        
        Args:
            prev_architecture: [batch, num_outputs] arquitectura previa
        
        Returns:
            architecture: [batch, num_outputs] nueva arquitectura
        """
        # Expandir dimensión para LSTM: [batch, num_outputs, 1]
        x = prev_architecture.unsqueeze(-1)
        
        # LSTM forward: [batch, num_outputs, hidden_size]
        lstm_out, _ = self.lstm(x)
        
        # Tomar último timestep: [batch, hidden_size]
        last_output = lstm_out[:, -1, :]
        
        # Generar DNA: [batch, num_outputs]
        architecture = self.fc(last_output)
        
        return architecture
    
    def sample(self, num_samples: int = 1):
        """
        Genera arquitecturas muestreando del controller.
        
        Args:
            num_samples: Número de arquitecturas a generar
        
        Returns:
            dna_list: Lista de DNAs generados
        """
        self.eval()
        with torch.no_grad():
            # Inicializar con arquitectura base
            prev_arch = torch.tensor(
                [[10.0, 128.0, 1.0, 1.0] * self.num_layers],
                dtype=torch.float32,
                device=self.device
            ).repeat(num_samples, 1)
            
            # Generar nueva arquitectura
            architecture = self.forward(prev_arch)
            
            # Convertir a DNA (escalar por 100 como en paper)
            dna = (architecture * 100).int()
            
        return dna.cpu().numpy()
    
    def get_policy_loss(self, architectures: torch.Tensor):
        """
        Calcula loss para policy gradient.
        
        Args:
            architectures: [batch, num_outputs] arquitecturas generadas
        
        Returns:
            loss: Scalar loss para backpropagation
        """
        # Forward pass
        predictions = self.forward(architectures / 100.0)
        
        # Softmax cross-entropy (como en paper)
        target = architectures / 100.0
        loss = nn.functional.mse_loss(predictions, target)
        
        return loss
```

**Checklist:**
- [ ] Clase NASController implementada
- [ ] Método forward funcional
- [ ] Método sample genera DNAs válidos
- [ ] Test unitario pasa

**Test:**
```python
def test_controller():
    controller = NASController(num_layers=3, device='cpu')
    dna = controller.sample(num_samples=5)
    assert dna.shape == (5, 12)  # 5 samples, 3 layers × 4 components
    print("✓ Controller test passed")
```

---

#### 2.3 Implementar child_builder.py

**Archivo:** `app/src/nas/child_builder.py`

```python
"""
Child Network Builder: Construye CNNs dinámicamente desde DNA.

El DNA codifica la arquitectura como lista de [kernel, filters, stride, pool].
"""

import torch
import torch.nn as nn
import numpy as np

class ChildNetworkBuilder:
    """Construye Child Networks desde especificación DNA."""
    
    @staticmethod
    def validate_dna(dna: np.ndarray) -> bool:
        """
        Valida que el DNA sea válido.
        
        Args:
            dna: Array [num_layers, 4] con arquitectura
        
        Returns:
            True si es válido, False otherwise
        """
        if dna.ndim != 2 or dna.shape[1] != 4:
            return False
        
        # Verificar que todos los valores sean positivos
        if np.any(dna <= 0):
            return False
        
        return True
    
    @staticmethod
    def build_from_dna(
        dna: np.ndarray,
        num_classes: int = 10,
        dropout_rate: float = 0.2
    ) -> nn.Module:
        """
        Construye CNN desde DNA.
        
        Args:
            dna: [num_layers, 4] array con [kernel, filters, stride, pool]
            num_classes: Número de clases de salida
            dropout_rate: Tasa de dropout
        
        Returns:
            model: nn.Module con la arquitectura especificada
        """
        if not ChildNetworkBuilder.validate_dna(dna):
            raise ValueError(f"Invalid DNA: {dna}")
        
        # Procesar DNA
        if dna.ndim == 1:
            # DNA plano: [k1,f1,s1,p1, k2,f2,s2,p2, ...]
            dna = dna.reshape(-1, 4)
        
        layers = []
        in_channels = 3
        
        for layer_idx, (kernel, filters, stride, pool) in enumerate(dna):
            # Convertir a int
            kernel = int(kernel)
            filters = int(filters)
            stride = int(stride)
            pool = int(pool)
            
            # Validar valores
            kernel = max(1, min(7, kernel))  # Clamp 1-7
            filters = max(32, min(512, filters))  # Clamp 32-512
            stride = max(1, min(2, stride))  # Clamp 1-2
            pool = max(1, min(3, pool))  # Clamp 1-3
            
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
            
            # MaxPool (si pool > 1)
            if pool > 1:
                layers.append(nn.MaxPool2d(
                    kernel_size=pool,
                    stride=1,
                    padding=pool // 2
                ))
            
            # Dropout
            layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = filters
        
        # Clasificador
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, num_classes))
        
        # Crear modelo
        model = nn.Sequential(*layers)
        
        return model
```

**Checklist:**
- [ ] Clase ChildNetworkBuilder implementada
- [ ] Validación de DNA funciona
- [ ] build_from_dna genera CNNs válidas
- [ ] Test pasa

**Test:**
```python
def test_child_builder():
    dna = np.array([[3, 128, 1, 1], [5, 256, 1, 2], [3, 512, 1, 1]])
    model = ChildNetworkBuilder.build_from_dna(dna)
    
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    print("✓ ChildBuilder test passed")
```

---

#### 2.4 Implementar reinforce.py

**Archivo:** `app/src/nas/reinforce.py`

```python
"""
REINFORCE: Policy Gradient optimizer para NAS Controller.

Implementa el algoritmo REINFORCE (Williams, 1992) con baseline EMA.
"""

import torch
import torch.optim as optim
import numpy as np

class REINFORCEOptimizer:
    """
    Optimizador REINFORCE con baseline EMA.
    
    El Controller se entrena para maximizar el expected reward (validation accuracy).
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
            lr: Learning rate inicial
            decay: Factor de decay
            decay_steps: Steps para aplicar decay
            beta: L2 regularization weight
            ema_alpha: Alpha para baseline EMA
        """
        self.controller = controller
        self.beta = beta
        self.ema_alpha = ema_alpha
        
        # Optimizer
        self.optimizer = optim.RMSprop(
            controller.parameters(),
            lr=lr
        )
        
        # LR Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=decay_steps,
            gamma=decay
        )
        
        # Baseline (EMA de rewards)
        self.baseline = 0.0
        self.reward_history = []
    
    def update_baseline(self, reward: float):
        """Actualiza baseline con EMA."""
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
        rewards: list[float]
    ):
        """
        Actualiza Controller con REINFORCE.
        
        Args:
            architectures: [batch, num_outputs] arquitecturas generadas
            rewards: Lista de rewards (validation accuracies)
        """
        self.optimizer.zero_grad()
        
        # Calcular policy loss
        policy_loss = self.controller.get_policy_loss(architectures)
        
        # L2 regularization
        l2_loss = sum(
            torch.sum(param ** 2)
            for param in self.controller.parameters()
        )
        total_loss = policy_loss + self.beta * l2_loss
        
        # Calcular advantage
        mean_reward = np.mean(rewards)
        self.update_baseline(mean_reward)
        advantage = mean_reward - self.baseline
        
        # REINFORCE: multiply gradient by advantage
        total_loss = total_loss * advantage
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(),
            max_norm=1.0
        )
        
        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': total_loss.item(),
            'reward': mean_reward,
            'baseline': self.baseline,
            'advantage': advantage
        }
```

**Checklist:**
- [ ] REINFORCEOptimizer implementado
- [ ] Baseline EMA funciona
- [ ] Gradient update correcto
- [ ] Test pasa

---

#### 2.5 Implementar trainer.py

**Archivo:** `app/src/nas/trainer.py`

```python
"""
NAS Trainer: Orquestador del proceso de búsqueda de arquitectura.

Integra Controller + Child Networks + REINFORCE para ejecutar NAS.
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from .controller import NASController
from .child_builder import ChildNetworkBuilder
from .reinforce import REINFORCEOptimizer
from ..train_pipeline import TrainingPipeline

class NASTrainer:
    """Orquestador de Neural Architecture Search."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Diccionario con configuración NAS
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
        
        # Historia
        self.reward_history = []
        self.architecture_history = []
        self.best_architecture = None
        self.best_reward = 0.0
        
        # Paths
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/nas'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _detect_device(self):
        """Detecta device disponible."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _train_child(
        self,
        dna: np.ndarray,
        child_id: str,
        train_loader,
        val_loader
    ) -> float:
        """
        Entrena Child Network y retorna reward.
        
        Args:
            dna: DNA de la arquitectura
            child_id: ID del child para logging
            train_loader: DataLoader de training
            val_loader: DataLoader de validation
        
        Returns:
            reward: Validation accuracy
        """
        # Construir modelo
        try:
            model = ChildNetworkBuilder.build_from_dna(dna)
        except ValueError as e:
            print(f"Invalid DNA {dna}: {e}")
            return 0.0  # Reward negativo para DNAs inválidos
        
        # Configuración de entrenamiento
        child_config = {
            'lr': self.config.get('child_lr', 3e-5),
            'epochs': self.config.get('child_epochs', 100),
            'batch_size': self.config.get('child_batch_size', 20),
            'checkpoint_dir': str(self.checkpoint_dir / child_id),
            'experiment_name': child_id
        }
        
        # Entrenar
        pipeline = TrainingPipeline(model, child_config)
        pipeline.train(train_loader, val_loader)
        
        # Obtener best validation accuracy
        reward = pipeline.best_val_acc
        
        return reward
    
    def search(
        self,
        train_loader,
        val_loader,
        num_episodes: int = 2000,
        children_per_episode: int = 10
    ):
        """
        Ejecuta búsqueda de arquitectura.
        
        Args:
            train_loader: DataLoader de training
            val_loader: DataLoader de validation
            num_episodes: Número de episodios
            children_per_episode: Children por episodio
        """
        print("=" * 70)
        print("INICIANDO NEURAL ARCHITECTURE SEARCH")
        print("=" * 70)
        print(f"Episodios: {num_episodes}")
        print(f"Children por episodio: {children_per_episode}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        for episode in tqdm(range(num_episodes), desc="NAS Episodes"):
            episode_rewards = []
            episode_architectures = []
            
            # Generar y evaluar children
            for child_idx in range(children_per_episode):
                # Sample arquitectura
                dna = self.controller.sample(num_samples=1)[0]
                dna = dna.reshape(-1, 4)
                
                # Entrenar child
                child_id = f"ep{episode}_child{child_idx}"
                reward = self._train_child(
                    dna, child_id, train_loader, val_loader
                )
                
                episode_rewards.append(reward)
                episode_architectures.append(dna)
                
                # Actualizar best
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_architecture = dna
                    print(f"\n✓ NEW BEST: {reward:.4f}")
            
            # Actualizar Controller con REINFORCE
            architectures_tensor = torch.tensor(
                np.vstack(episode_architectures),
                dtype=torch.float32,
                device=self.device
            )
            
            metrics = self.reinforce.step(architectures_tensor, episode_rewards)
            
            # Logging
            mean_reward = np.mean(episode_rewards)
            self.reward_history.append(mean_reward)
            
            if episode % 10 == 0:
                print(f"\nEpisode {episode}")
                print(f"  Mean Reward: {mean_reward:.4f}")
                print(f"  Baseline: {metrics['baseline']:.4f}")
                print(f"  Best Reward: {self.best_reward:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(f"nas_episode_{episode}.pth")
        
        print("\n" + "=" * 70)
        print("BÚSQUEDA COMPLETADA")
        print(f"Best Reward: {self.best_reward:.4f}")
        print("=" * 70)
    
    def save_checkpoint(self, filename: str):
        """Guarda checkpoint de NAS."""
        checkpoint = {
            'controller_state_dict': self.controller.state_dict(),
            'optimizer_state_dict': self.reinforce.optimizer.state_dict(),
            'reward_history': self.reward_history,
            'architecture_history': self.architecture_history,
            'best_architecture': self.best_architecture,
            'best_reward': self.best_reward,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint guardado: {path}")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Reanuda desde checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.reinforce.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.reward_history = checkpoint['reward_history']
        self.architecture_history = checkpoint.get('architecture_history', [])
        self.best_architecture = checkpoint.get('best_architecture')
        self.best_reward = checkpoint.get('best_reward', 0.0)
        
        print(f"✓ Checkpoint cargado: {checkpoint_path}")
        print(f"  Episodes completados: {len(self.reward_history)}")
        print(f"  Best reward: {self.best_reward:.4f}")
```

**Checklist:**
- [ ] NASTrainer implementado
- [ ] Método search funcional
- [ ] Checkpointing funciona
- [ ] Resume funciona

---

### Criterios de Éxito Sprint 2

- ✓ Módulo `app/src/nas/` completo
- ✓ Todos los componentes tienen tests
- ✓ NAS ejecuta end-to-end (aunque sea 10 episodios de prueba)
- ✓ Checkpoints se guardan y cargan correctamente

---

## 📋 Sprint 3: Checkpoint Support y CLI

**Duración:** 1 semana (13-19 Dic)  
**Prioridad:** 🟡 Media

### Objetivos

- Implementar sistema robusto de checkpoints
- CLI para iniciar/reanudar búsquedas
- Visualización de progreso NAS

### Tareas

#### 3.1 Mejorar sistema de checkpoints

- [ ] Formato de checkpoint con metadatos completos
- [ ] Compresión de checkpoints grandes
- [ ] Auto-save cada N episodios
- [ ] Cleanup de checkpoints antiguos

#### 3.2 CLI para NAS

```python
# app/nas_cli.py

import argparse
from src.nas.trainer import NASTrainer

def main():
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('--mode', choices=['search', 'resume'], required=True)
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for resume')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--config', type=str, default='configs/nas_default.json')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    # Load data
    train_loader, val_loader = load_cifar10_loaders(config)
    
    # Create trainer
    trainer = NASTrainer(config)
    
    if args.mode == 'resume':
        trainer.resume_from_checkpoint(args.checkpoint)
    
    # Start search
    trainer.search(train_loader, val_loader, num_episodes=args.episodes)

if __name__ == '__main__':
    main()
```

**Uso:**
```bash
# Nueva búsqueda
python app/nas_cli.py --mode search --episodes 2000

# Reanudar
python app/nas_cli.py --mode resume --checkpoint checkpoints/nas/nas_episode_500.pth
```

#### 3.3 Visualización de progreso

```python
# app/src/nas/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_nas_progress(reward_history, save_path='plots/nas_progress.png'):
    """Visualiza progreso de búsqueda NAS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Rewards over time
    ax1.plot(reward_history, alpha=0.6, label='Episode Reward')
    ax1.plot(np.convolve(reward_history, np.ones(10)/10, mode='valid'), 
             label='Moving Avg (10)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('NAS Progress: Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution
    sns.histplot(reward_history, bins=50, ax=ax2)
    ax2.set_xlabel('Reward')
    ax2.set_title('Reward Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {save_path}")
```

---

## 📋 Sprint 4: Testing y Documentación

**Duración:** 1 semana (20-26 Dic)  
**Prioridad:** 🟢 Baja

### Objetivos

- Tests completos
- Documentación final
- Release v1.0

### Tareas

#### 4.1 Suite de tests

```bash
tests/
├── test_arqui_cnn.py         # Test NASCNN15
├── test_load.py              # Test data loading
├── test_nas_controller.py    # Test Controller
├── test_child_builder.py     # Test ChildBuilder
├── test_reinforce.py         # Test REINFORCE
└── test_nas_trainer.py       # Test end-to-end
```

#### 4.2 Documentación

- [ ] Finalizar Monografia_NASCNN.md
- [ ] Actualizar README con instrucciones NAS
- [ ] Crear NASCNN_TRAINING_GUIDE.md
- [ ] Docstrings completos en todo el código

#### 4.3 Release

- [ ] Tag v1.0.0
- [ ] Release notes
- [ ] Merge a main

---

## 📊 Métricas de Éxito Global

### Técnicas
- ✓ NASCNN15 entrena sin errores
- ✓ Accuracy ≥ 91% en CIFAR-10 test
- ✓ NAS encuentra arquitecturas ≥ 85% accuracy
- ✓ Checkpoints funcionan correctamente
- ✓ Tests pasan (coverage ≥ 80%)

### Código
- ✓ Sin referencias a CIFAR-10.1
- ✓ Sin arquitecturas no utilizadas
- ✓ Código PyTorch puro (sin TensorFlow)
- ✓ Documentación completa

### Tiempo
- ✓ Sprint 1: 1 semana
- ✓ Sprint 2: 2 semanas
- ✓ Sprint 3: 1 semana
- ✓ Sprint 4: 1 semana
- **Total: 5 semanas**

---

## 🚀 Quick Commands

```bash
# Activar entorno
source .venv/bin/activate

# Sprint 1: Tests de regresión
python -m pytest tests/ -v

# Sprint 2: Test NAS
python -c "from app.src.nas import NASController; print('✓ NAS module OK')"

# Sprint 3: Búsqueda NAS
python app/nas_cli.py --mode search --episodes 100

# Sprint 4: Release
git tag -a v1.0.0 -m "NASCNN15 + NAS con RL"
git push origin v1.0.0
```

---

**Documento vivo - Actualizar conforme avance**

**Última actualización:** 21 Noviembre 2025  
**Versión:** 1.0  
**Estado:** Planificación completa

