# Monografía: NASCNN15 - Neural Architecture Search con Reinforcement Learning

**Proyecto:** VC-ARN - Visión Computacional con Aprendizaje por Refuerzo  
**Rama:** NASCNN15 Branch  
**Autor:** Esp. Joaquín S Tschopp  
**Fecha:** Noviembre 2025  
**Dataset:** CIFAR-10  

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Introducción](#introducción)
3. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
4. [Neural Architecture Search (NAS)](#neural-architecture-search-nas)
5. [NASCNN15: Arquitectura Detallada](#nascnn15-arquitectura-detallada)
6. [Estructura Actual del Proyecto](#estructura-actual-del-proyecto)
7. [Plan de Refactorización](#plan-de-refactorización)
8. [Migración de NAS con RL](#migración-de-nas-con-rl)
9. [Roadmap de Implementación](#roadmap-de-implementación)
10. [Referencias](#referencias)

---

## Resumen Ejecutivo

Este documento detalla la implementación de **Neural Architecture Search (NAS)** utilizando **Reinforcement Learning (RL)** en el proyecto VC-ARN. El objetivo principal es:

- **Consolidar la arquitectura NASCNN15** como modelo único del proyecto
- **Eliminar referencias a CIFAR-10.1**, manteniendo solo CIFAR-10 como dataset principal
- **Integrar NAS con RL** desde la carpeta `Neural-Architecture-Search-using-Reinforcement-Learning/` a `app/src/`
- **Implementar búsqueda de arquitectura desde checkpoint** para continuar entrenamientos

### Estado Actual
- ✅ Proyecto funcional con múltiples arquitecturas CNN
- ✅ NASCNN15 implementada en PyTorch
- ✅ Sistema de entrenamiento robusto con TrainingPipeline
- ⚠️ Referencias a CIFAR-10.1 dispersas en el código
- ⚠️ NAS con RL en carpeta separada (TensorFlow 1.x)

### Objetivos del Branch
1. **Arquitectura única**: Solo NASCNN15 disponible
2. **Dataset único**: Solo CIFAR-10 (train/val/test split)
3. **NAS integrado**: Controller de RL + Child Network en `src/`
4. **Checkpoint support**: Reanudar búsqueda de arquitectura

---

## Introducción

### Contexto del Proyecto

El proyecto **VC-ARN** (Visión Computacional - Arquitecturas de Redes Neuronales) implementa un sistema completo de clasificación de imágenes usando Deep Learning. La implementación actual incluye:

- **4 arquitecturas CNN**: BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR, NASCNN15
- **Pipeline OOP**: Sistema orientado a objetos para entrenamientos reproducibles
- **Multi-hardware**: Soporte automático para CUDA, MPS (Apple Silicon) y CPU
- **Experimentos tracking**: Sistema de checkpoints y registro de experimentos

### Motivación para NAS

**Neural Architecture Search** permite:
- Descubrir arquitecturas óptimas automáticamente
- Superar diseños manuales en accuracy
- Explorar espacios de búsqueda complejos

**NASCNN15** fue descubierta mediante NAS con RL en el paper seminal de Zoph & Le (2017).

---

## Arquitectura del Proyecto

### Estructura Actual de Directorios

```
VC-ARN/
├── Neural-Architecture-Search-using-Reinforcement-Learning/
│   ├── Controller.py              # RL Controller (TensorFlow)
│   ├── train.py                   # Script principal NAS
│   ├── Utils/
│   │   ├── child_network.py       # ChildCNN class
│   │   ├── cifar10_processor.py   # Data loading
│   │   ├── configs.py             # Hyperparameters
│   │   └── constants.py           # Paths
│   └── README.md
│
├── app/
│   ├── main.py                    # Punto de entrada
│   ├── src/
│   │   ├── arqui_cnn.py           # 5 arquitecturas (BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR, NASCNN15)
│   │   ├── load.py                # Data loaders (CIFAR-10 + CIFAR-10.1)
│   │   ├── train_pipeline.py      # TrainingPipeline class
│   │   ├── test.py                # Evaluation
│   │   ├── pre_processed.py       # Transformations
│   │   └── auxiliares.py          # Utilities
│   └── README.md
│
├── experiments/                   # Resultados de entrenamientos
├── datasets/                      # CIFAR-10 data
└── models/                        # Checkpoints
```

### Componentes Clave

#### 1. TrainingPipeline (`app/src/train_pipeline.py`)
- Clase principal para entrenamientos
- Métricas: loss, accuracy, confusion matrix
- Checkpointing automático
- Early stopping
- Visualizaciones con matplotlib/seaborn

#### 2. Arquitecturas CNN (`app/src/arqui_cnn.py`)
- **BaseModel**: Fully Connected baseline (~50% acc)
- **SimpleCNN**: 3 bloques Conv (65-70% acc)
- **ImprovedCNN**: 5 bloques Conv + BatchNorm (75-80% acc)
- **ResNetCIFAR**: Skip connections (80-85% acc)
- **NASCNN15**: Arquitectura descubierta por NAS (85%+ acc)

#### 3. Data Loading (`app/src/load.py`)
- `load_cifar10()`: CIFAR-10 con split 45k/5k/10k
- `load_cifar101()`: CIFAR-10.1 (dataset externo de evaluación)
- Transformaciones: normalización, augmentation

---

## Neural Architecture Search (NAS)

### Fundamentos

NAS automatiza el diseño de arquitecturas CNN mediante:
1. **Search Space**: Espacio de posibles arquitecturas
2. **Search Strategy**: Algoritmo de búsqueda (RL, EA, gradient-based)
3. **Performance Estimation**: Evaluación de arquitecturas candidatas

### NAS con Reinforcement Learning (Zoph & Le 2017)

#### Arquitectura del Sistema

```
┌─────────────────────────────────────────────────┐
│           CONTROLLER (RNN)                      │
│  - NASCell (LSTM modificado)                    │
│  - Genera arquitecturas como secuencias         │
│  - Entrenado con REINFORCE                      │
└──────────────────┬──────────────────────────────┘
                   │ DNA (arquitectura codificada)
                   ▼
┌─────────────────────────────────────────────────┐
│          CHILD NETWORK (CNN)                    │
│  - CNN construida según DNA                     │
│  - Entrenada en CIFAR-10                        │
│  - Retorna accuracy como reward                 │
└──────────────────┬──────────────────────────────┘
                   │ Validation Accuracy (reward)
                   ▼
┌─────────────────────────────────────────────────┐
│        POLICY GRADIENT UPDATE                   │
│  - Baseline: EMA de rewards                     │
│  - Actualiza Controller para max reward         │
└─────────────────────────────────────────────────┘
```

#### DNA Encoding

Cada capa se codifica como 4 valores:
```python
[kernel_size, num_filters, stride, max_pool_size]
```

Ejemplo para 3 capas:
```python
DNA = [
    [3, 128, 1, 1],  # Capa 1: kernel 3x3, 128 filtros
    [5, 256, 1, 2],  # Capa 2: kernel 5x5, 256 filtros
    [3, 512, 1, 1],  # Capa 3: kernel 3x3, 512 filtros
]
```


#### Algoritmo REINFORCE

El Controller se entrena con el algoritmo de Policy Gradient:

```
1. Sample arquitectura: a ~ π_θ(a)
2. Entrenar Child Network con arquitectura a
3. Obtener reward R (validation accuracy)
4. Actualizar θ: ∇_θ J(θ) = (R - baseline) * ∇_θ log π_θ(a)
5. Repetir hasta convergencia
```

**Baseline**: Exponential Moving Average (EMA) de rewards
**Learning Rate**: Decay exponencial (0.99 inicial, decay 0.96 cada 500 steps)

### Implementación Original (TensorFlow 1.x)

El código en `Neural-Architecture-Search-using-Reinforcement-Learning/` usa:
- **TensorFlow 1.x** (Session-based, deprecated)
- **NASCell** (tf.contrib.rnn, removido en TF 2.x)
- **3 capas máximo** por arquitectura
- **100 épocas** de entrenamiento por Child Network

---

## NASCNN15: Arquitectura Detallada

### Descripción General

**NASCNN15** es la arquitectura de 15 capas descubierta por NAS (Figura 7 del paper Zoph & Le).

#### Características Principales
- **15 capas convolucionales** con skip connections
- **Sin stride ni pooling** (resolución 32×32 constante)
- **Filtros de múltiples tamaños**: 1x1, 3x3, 3x7, 5x5, 5x7, 7x1, 7x3, 7x5, 7x7
- **Anchos variables**: N ∈ {36, 48} filtros por capa
- **Global Average Pooling** + FC al final
- **~2.5M parámetros** (relativamente compacta)

### Estructura Capa por Capa

```
Capa | Kernel | Filtros | Input Layers      | Output Shape
-----|--------|---------|-------------------|-------------
C1   | 3×3    | 36      | RGB               | [B, 36, 32, 32]
C2   | 3×3    | 48      | C1                | [B, 48, 32, 32]
C3   | 3×3    | 36      | C1, C2            | [B, 36, 32, 32]
C4   | 5×5    | 36      | C1, C2, C3        | [B, 36, 32, 32]
C5   | 3×7    | 48      | C3, C4            | [B, 48, 32, 32]
C6   | 7×7    | 48      | C2, C3, C4, C5    | [B, 48, 32, 32]
C7   | 7×7    | 48      | C2-C6             | [B, 48, 32, 32]
C8   | 7×3    | 36      | C1, C6, C7        | [B, 36, 32, 32]
C9   | 7×1    | 36      | C1, C5, C6, C8    | [B, 36, 32, 32]
C10  | 7×7    | 36      | C1, C3-C9         | [B, 36, 32, 32]
C11  | 5×7    | 36      | C1, C2, C5-C10    | [B, 36, 32, 32]
C12  | 7×7    | 48      | C1-C4, C6, C11    | [B, 48, 32, 32]
C13  | 7×5    | 48      | C1, C3, C6-C12    | [B, 48, 32, 32]
C14  | 7×5    | 48      | C3, C7, C12, C13  | [B, 48, 32, 32]
C15  | 7×5    | 48      | C6, C7, C11-C14   | [B, 48, 32, 32]
```

### Skip Connections

Las conexiones skip se implementan mediante **concatenación por canal**:

```python
# Ejemplo: C3 recibe C1 (36 canales) + C2 (48 canales)
x3_in = torch.cat([x1, x2], dim=1)  # [B, 84, 32, 32]
x3 = F.relu(self.bn3(self.conv3(x3_in)))  # [B, 36, 32, 32]
```

### Bloques de Construcción

Cada capa sigue el patrón:
```
Conv2d (bias=False) → BatchNorm2d → ReLU
```

**Regularización**:
- BatchNorm en todas las capas
- NO usa Dropout
- Weight decay: 1e-4

### Clasificador Final

```python
# Global Average Pooling
out = F.adaptive_avg_pool2d(x15, output_size=1)  # [B, 48, 1, 1]
out = out.view(out.size(0), -1)                  # [B, 48]
out = self.fc(out)                               # [B, 10] (logits)
```

**Nota**: El forward retorna logits (sin softmax). Se usa `nn.CrossEntropyLoss` que aplica softmax internamente.

### Hiperparámetros de Entrenamiento

Según paper original (NAS v1):
```python
optimizer = SGD(
    lr=0.1,              # Learning rate inicial
    momentum=0.9,        # Nesterov momentum
    weight_decay=1e-4,   # L2 regularization
    nesterov=True
)

lr_scheduler = ReduceLROnPlateau(
    patience=10,
    factor=0.5
)

epochs = 300
batch_size = 128
loss = nn.CrossEntropyLoss(label_smoothing=0.0)
```

### Accuracy Esperado

| Dataset | Split | Accuracy | Paper | Implementación Actual |
|---------|-------|----------|-------|----------------------|
| CIFAR-10 | Train | ~99% | ✓ | ✓ |
| CIFAR-10 | Val (5k) | 91-93% | ✓ | En progreso |
| CIFAR-10 | Test (10k) | 91.5% | ✓ | En progreso |

---

## Estructura Actual del Proyecto

### Análisis de Componentes

#### app/src/arqui_cnn.py

**Problemas identificados**:
1. ❌ **5 arquitecturas diferentes** (queremos solo NASCNN15)
2. ⚠️ Todas comparten mismo archivo
3. ✅ NASCNN15 bien implementada en PyTorch
4. ✅ Documentación exhaustiva

**Arquitecturas a eliminar**:
- BaseModel (baseline FC)
- SimpleCNN
- ImprovedCNN
- ResNetCIFAR

#### app/src/load.py

**Problemas identificados**:
1. ❌ **Referencias a CIFAR-10.1** en múltiples funciones:
   - `load_data()`: Descarga CIFAR-10.1 (líneas 46-77)
   - `load_cifar101()`: Loader para CIFAR-10.1 (líneas 160-204)
   - `Cifar101Dataset`: Custom dataset class (líneas 15-33)

2. ✅ `load_cifar10()` bien implementado con split estratificado

**Acción requerida**:
- Eliminar funciones relacionadas con CIFAR-10.1
- Mantener solo `load_cifar10()` con split 45k/5k/10k

#### app/main.py

**Problemas identificados**:
1. ❌ Líneas comentadas con todas las arquitecturas (151-155)
2. ❌ Nombre de experimento: "NASCNN_V13_OnlyCIFAR10" (línea 49)
3. ✅ Config bien definido para NASCNN15
4. ✅ Sistema de tracking de experimentos

#### Neural-Architecture-Search-using-Reinforcement-Learning/

**Estado**:
- ⚠️ Código en **TensorFlow 1.x** (incompatible con proyecto PyTorch)
- ✅ Lógica de NAS bien documentada
- ✅ Controller con REINFORCE implementado

**Necesita migración a**:
- PyTorch
- Integración con TrainingPipeline
- Soporte para checkpoints

---

## Plan de Refactorización

### Fase 1: Limpieza de Código Base (Prioridad Alta)

#### Objetivo
Eliminar todas las referencias a CIFAR-10.1 y arquitecturas no utilizadas.

#### Tareas

**1.1. Limpiar app/src/arqui_cnn.py**
```python
# ANTES (494 líneas)
class BaseModel(nn.Module): ...
class SimpleCNN(nn.Module): ...
class ImprovedCNN(nn.Module): ...
class ResNetCIFAR(nn.Module): ...
class NASCNN15(nn.Module): ...

# DESPUÉS (~150 líneas)
class NASCNN15(nn.Module): ...
```

**1.2. Limpiar app/src/load.py**
```python
# ELIMINAR:
- class Cifar101Dataset(Dataset)
- def load_data(datasets_folder)
- def load_cifar101(datasets_folder, ...)

# MANTENER:
- def load_cifar10(datasets_folder, config)
```

**1.3. Actualizar app/main.py**
```python
# ELIMINAR líneas comentadas:
#model = BaseModel()
#model = SimpleCNN()
#model = ImprovedCNN()
#model = ResNetCIFAR()

# MANTENER solo:
model = NASCNN15()
```

**1.4. Actualizar app/src/test.py**
- Eliminar función `run_cifar101_evaluation()` si existe
- Mantener solo `run_cifar10_test_evaluation()`

**1.5. Actualizar app/src/auxiliares.py**
```python
# Función compare_models():
# ANTES: Compara todas las arquitecturas
# DESPUÉS: Solo muestra info de NASCNN15
```

### Fase 2: Migración de NAS con RL (Prioridad Media)

#### Objetivo
Trasladar e integrar NAS desde carpeta separada a `app/src/`

#### Nuevos Archivos en app/src/

**2.1. app/src/nas_controller.py**
```python
"""
Neural Architecture Search Controller con Reinforcement Learning.

Componentes:
- NASController: Modelo RNN para generar arquitecturas
- REINFORCETrainer: Lógica de policy gradient
- ArchitectureEncoder: Convierte DNA → arquitectura PyTorch
"""

class NASController(nn.Module):
    """Controller basado en LSTM para generar arquitecturas."""
    
    def __init__(self, num_layers=3, components_per_layer=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(100, num_layers * components_per_layer)
    
    def forward(self, prev_architecture):
        """Genera nueva arquitectura basada en la anterior."""
        ...
```

**2.2. app/src/child_network_builder.py**
```python
"""
Construcción dinámica de Child Networks desde DNA.

DNA Format:
    [[kernel_size, num_filters, stride, max_pool], ...]
"""

class ChildNetworkBuilder:
    """Construye CNN desde especificación DNA."""
    
    @staticmethod
    def build_from_dna(dna: list[list[int]]) -> nn.Module:
        """
        Construye Child Network dinámicamente.
        
        Args:
            dna: Lista de [kernel, filters, stride, pool] por capa
        
        Returns:
            nn.Module: CNN construida
        """
        layers = []
        in_channels = 3
        
        for kernel, filters, stride, pool in dna:
            layers.append(nn.Conv2d(
                in_channels, filters,
                kernel_size=kernel,
                stride=stride,
                padding=kernel//2
            ))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU())
            
            if pool > 1:
                layers.append(nn.MaxPool2d(pool))
            
            in_channels = filters
        
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
```

**2.3. app/src/nas_trainer.py**
```python
"""
Entrenador de NAS que integra Controller + Child Networks.
"""

class NASTrainer:
    """Orquestador del proceso de búsqueda de arquitectura."""
    
    def __init__(self, config):
        self.controller = NASController(...)
        self.child_trainer = TrainingPipeline(...)  # Reusa pipeline existente
        self.reward_history = []
        self.architecture_history = []
    
    def search(self, num_episodes=2000, children_per_episode=10):
        """
        Ejecuta búsqueda de arquitectura.
        
        Process:
        1. Sample arquitectura del Controller
        2. Entrenar Child Network
        3. Obtener validation accuracy (reward)
        4. Actualizar Controller con REINFORCE
        """
        for episode in range(num_episodes):
            episode_rewards = []
            
            for child_id in range(children_per_episode):
                # 1. Generate architecture
                dna = self.controller.sample()
                
                # 2. Build and train child network
                child_model = ChildNetworkBuilder.build_from_dna(dna)
                reward = self._train_child(child_model, child_id)
                
                episode_rewards.append(reward)
            
            # 3. Update controller
            self._update_controller(episode_rewards)
            
            # 4. Save checkpoint
            if episode % 10 == 0:
                self.save_checkpoint(f"nas_episode_{episode}.pth")
    
    def resume_from_checkpoint(self, checkpoint_path):
        """Reanuda búsqueda desde checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.controller.load_state_dict(checkpoint['controller'])
        self.reward_history = checkpoint['reward_history']
        ...
```


#### Estructura de Archivos Propuesta

```
app/src/
├── arqui_cnn.py               # Solo NASCNN15
├── load.py                    # Solo load_cifar10()
├── train_pipeline.py          # Sin cambios
├── test.py                    # Solo CIFAR-10 test
├── pre_processed.py           # Sin cambios
├── auxiliares.py              # Actualizado para NASCNN15
│
├── nas/                       # NUEVO: Módulo NAS
│   ├── __init__.py
│   ├── controller.py          # NASController (LSTM)
│   ├── child_builder.py       # ChildNetworkBuilder
│   ├── trainer.py             # NASTrainer (orchestrator)
│   ├── reinforce.py           # REINFORCE algorithm
│   └── utils.py               # DNA encoding/decoding
```

### Fase 3: Integración y Testing (Prioridad Media)

#### 3.1. Actualizar main.py

```python
# app/main.py

from src.arqui_cnn import NASCNN15
from src.load import load_cifar10
from src.nas.trainer import NASTrainer  # NUEVO

def main():
    # Modo 1: Entrenar NASCNN15 directamente
    if config['mode'] == 'train_nascnn':
        model = NASCNN15()
        pipeline = TrainingPipeline(model, config)
        pipeline.train(train_dataloader, val_dataloader)
    
    # Modo 2: Ejecutar búsqueda NAS
    elif config['mode'] == 'nas_search':
        nas_trainer = NASTrainer(config)
        nas_trainer.search(
            num_episodes=2000,
            children_per_episode=10
        )
    
    # Modo 3: Reanudar búsqueda NAS
    elif config['mode'] == 'nas_resume':
        nas_trainer = NASTrainer(config)
        nas_trainer.resume_from_checkpoint(
            checkpoint_path='checkpoints/nas_episode_500.pth'
        )
```

#### 3.2. Tests Unitarios

```python
# tests/test_nas_controller.py

def test_controller_output_shape():
    controller = NASController(num_layers=3)
    dna = controller.sample()
    assert dna.shape == (3, 4)  # 3 capas, 4 componentes

def test_child_builder():
    dna = [[3, 128, 1, 1], [5, 256, 1, 2]]
    model = ChildNetworkBuilder.build_from_dna(dna)
    
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 256, 8, 8)  # Depende del pooling
```

### Fase 4: Documentación y Refinamiento (Prioridad Baja)

#### 4.1. Actualizar README.md principal

```markdown
# VC-ARN: Neural Architecture Search con NASCNN15

## Arquitectura
- **NASCNN15**: Arquitectura de 15 capas descubierta por NAS
- **Dataset**: CIFAR-10 (45k train / 5k val / 10k test)
- **NAS con RL**: Sistema completo de búsqueda de arquitectura

## Modos de Uso
1. Train NASCNN15: `python main.py --mode train_nascnn`
2. NAS Search: `python main.py --mode nas_search`
3. Resume NAS: `python main.py --mode nas_resume --checkpoint <path>`
```

#### 4.2. Crear NASCNN_TRAINING_GUIDE.md

Documentación detallada de:
- Hiperparámetros óptimos
- Curvas de entrenamiento esperadas
- Troubleshooting común
- Comparación con baselines

---

## Roadmap de Implementación

### Sprint 1: Limpieza (1 semana)

**Objetivos**:
- ✅ Eliminar arquitecturas no utilizadas
- ✅ Remover CIFAR-10.1
- ✅ Código base limpio y funcional

**Tareas**:
1. [ ] Crear branch `refactor/nascnn15-only`
2. [ ] Limpiar `arqui_cnn.py` (eliminar 4 clases)
3. [ ] Limpiar `load.py` (eliminar CIFAR-10.1)
4. [ ] Actualizar `main.py` (solo NASCNN15)
5. [ ] Actualizar `auxiliares.py`
6. [ ] Actualizar `test.py`
7. [ ] Ejecutar tests de regresión
8. [ ] Merge a `main` tras validación

**Criterios de Éxito**:
- ✓ Proyecto ejecuta sin errores
- ✓ NASCNN15 entrena correctamente
- ✓ No referencias a CIFAR-10.1
- ✓ README actualizado

### Sprint 2: Migración NAS (2 semanas)

**Objetivos**:
- ✅ NAS con RL integrado en PyTorch
- ✅ Compatibilidad con TrainingPipeline existente

**Tareas**:
1. [ ] Crear módulo `app/src/nas/`
2. [ ] Implementar `controller.py` (LSTM en PyTorch)
3. [ ] Implementar `child_builder.py` (DNA → CNN)
4. [ ] Implementar `reinforce.py` (policy gradient)
5. [ ] Implementar `trainer.py` (orchestrator)
6. [ ] Tests unitarios para cada componente
7. [ ] Documentación de API

**Criterios de Éxito**:
- ✓ Controller genera DNAs válidos
- ✓ ChildBuilder construye CNNs funcionales
- ✓ REINFORCE actualiza Controller correctamente
- ✓ NASTrainer ejecuta búsqueda end-to-end

### Sprint 3: Checkpoint Support (1 semana)

**Objetivos**:
- ✅ Reanudar búsqueda NAS desde cualquier episodio

**Tareas**:
1. [ ] Diseñar formato de checkpoint NAS
2. [ ] Implementar `save_checkpoint()` en NASTrainer
3. [ ] Implementar `resume_from_checkpoint()`
4. [ ] Tests de save/load
5. [ ] CLI interface para resume

**Formato de Checkpoint**:
```python
checkpoint = {
    'episode': 500,
    'controller_state_dict': controller.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'reward_history': [...],
    'architecture_history': [...],
    'best_architecture': {...},
    'best_reward': 0.915
}
```

**Criterios de Éxito**:
- ✓ Checkpoint guarda estado completo
- ✓ Resume continúa desde episodio correcto
- ✓ Reward history preservado
- ✓ Best architecture recuperable

### Sprint 4: Integración y Testing (1 semana)

**Objetivos**:
- ✅ Sistema completo probado end-to-end
- ✅ Documentación finalizada

**Tareas**:
1. [ ] Ejecutar búsqueda NAS completa (100 episodios)
2. [ ] Validar arquitecturas descubiertas
3. [ ] Comparar con NASCNN15 original
4. [ ] Crear visualizaciones de búsqueda
5. [ ] Finalizar documentación
6. [ ] Code review
7. [ ] Release v1.0

**Criterios de Éxito**:
- ✓ NAS encuentra arquitecturas >85% accuracy
- ✓ Documentación completa
- ✓ Tests pasan (100% coverage en NAS module)
- ✓ Performance comparable a paper original

---

## Migración de Código: TensorFlow → PyTorch

### Mapeo de Componentes

| TensorFlow 1.x | PyTorch | Notas |
|----------------|---------|-------|
| `tf.contrib.rnn.NASCell` | `nn.LSTM` | Usar LSTM estándar |
| `tf.Session()` | N/A | PyTorch es eager |
| `tf.placeholder` | Tensor input | No necesario |
| `tf.train.RMSPropOptimizer` | `optim.RMSprop` | API similar |
| `tf.nn.softmax_cross_entropy` | `nn.CrossEntropyLoss` | Incluye softmax |
| `tf.layers.conv2d` | `nn.Conv2d` | Sintaxis similar |
| `tf.layers.batch_normalization` | `nn.BatchNorm2d` | Comportamiento idéntico |

### Controller: TensorFlow vs PyTorch

**TensorFlow (original)**:
```python
class Controller:
    def network_generator(self, nas_cell_hidden_state):
        nas = tf.contrib.rnn.NASCell(self.num_cell_outputs)
        network_architecture, _ = tf.nn.dynamic_rnn(
            nas, 
            tf.expand_dims(nas_cell_hidden_state, -1), 
            dtype=tf.float32
        )
        bias = tf.Variable([0.01] * self.num_cell_outputs)
        return tf.nn.bias_add(network_architecture[:, -1, :], bias)
```

**PyTorch (migrado)**:
```python
class NASController(nn.Module):
    def __init__(self, num_outputs=12):  # 3 layers × 4 components
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, batch_first=True)
        self.fc = nn.Linear(100, num_outputs)
        self.fc.bias.data.fill_(0.01)
    
    def forward(self, prev_arch):
        # prev_arch: [batch, num_outputs]
        x = prev_arch.unsqueeze(-1)  # [batch, num_outputs, 1]
        lstm_out, _ = self.lstm(x)   # [batch, num_outputs, 100]
        output = self.fc(lstm_out[:, -1, :])  # [batch, num_outputs]
        return output
```

### REINFORCE: TensorFlow vs PyTorch

**TensorFlow (original)**:
```python
# Gradients con REINFORCE
for i, (grad, var) in enumerate(self.gradients):
    if grad is not None:
        self.gradients[i] = (grad * self.discounted_rewards, var)

self.train_op = self.optimizer.apply_gradients(
    self.gradients, 
    global_step=self.global_step
)
```

**PyTorch (migrado)**:
```python
class REINFORCEOptimizer:
    def __init__(self, controller, lr=0.99):
        self.optimizer = optim.RMSprop(controller.parameters(), lr=lr)
    
    def step(self, loss, reward, baseline):
        """Actualiza controller con policy gradient."""
        self.optimizer.zero_grad()
        
        # Policy gradient: (R - baseline) * ∇log π(a)
        advantage = reward - baseline
        policy_loss = loss * advantage
        
        policy_loss.backward()
        self.optimizer.step()
```

---

## Referencias

### Papers Fundamentales

1. **Neural Architecture Search with Reinforcement Learning**  
   Zoph, B., & Le, Q. V. (2017)  
   ICLR 2017  
   [arXiv:1611.01578](https://arxiv.org/abs/1611.01578)

2. **Learning Transferable Architectures for Scalable Image Recognition**  
   Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018)  
   CVPR 2018  
   [arXiv:1707.07012](https://arxiv.org/abs/1707.07012)

3. **CIFAR-10: Learning Multiple Layers of Features from Tiny Images**  
   Krizhevsky, A., & Hinton, G. (2009)  
   Technical Report, University of Toronto

### Implementaciones de Referencia

- **Implementación original TensorFlow**: [carpeta actual del proyecto]
- **AutoML Project (Google)**: https://github.com/google/automl
- **NAS-Bench**: https://github.com/google-research/nasbench

### Recursos Adicionales

- **PyTorch Documentation**: https://pytorch.org/docs/
- **CIFAR-10 Dataset**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Reinforcement Learning: An Introduction** (Sutton & Barto)

---

## Apéndices

### A. Hiperparámetros Completos

#### NASCNN15 Training
```python
config_nascnn15 = {
    # Optimizer
    'optimizer': 'SGD',
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'nesterov': True,
    
    # Scheduler
    'scheduler': 'ReduceLROnPlateau',
    'lr_patience': 10,
    'lr_factor': 0.5,
    
    # Training
    'epochs': 300,
    'batch_size': 128,
    'label_smoothing': 0.0,
    
    # Early Stopping
    'es_patience': 20,
    
    # Data Augmentation
    'use_augmentation': True,
    'horizontal_flip': True,
    'random_crop': True,
    'padding': 4,
    
    # Normalization
    'mean': [0.491, 0.482, 0.447],
    'std': [0.247, 0.243, 0.262]
}
```

#### NAS Controller
```python
config_nas_controller = {
    # Architecture
    'max_layers': 3,
    'components_per_layer': 4,  # [kernel, filters, stride, pool]
    'lstm_hidden_size': 100,
    
    # Training
    'max_episodes': 2000,
    'children_per_episode': 10,
    'child_epochs': 100,
    'child_batch_size': 20,
    'child_lr': 3e-5,
    
    # REINFORCE
    'controller_lr': 0.99,
    'lr_decay': 0.96,
    'lr_decay_steps': 500,
    'baseline_ema_alpha': 0.95,
    'beta': 1e-4,  # L2 weight decay
}
```

### B. Formato de DNA

```python
# Ejemplo de DNA válido
dna_example = [
    [3, 128, 1, 1],  # Layer 1: kernel 3×3, 128 filters, stride 1, no pool
    [5, 256, 1, 2],  # Layer 2: kernel 5×5, 256 filters, stride 1, pool 2×2
    [3, 512, 1, 1],  # Layer 3: kernel 3×3, 512 filters, stride 1, no pool
]

# Constraints:
# - kernel_size ∈ [1, 3, 5, 7]
# - num_filters ∈ [32, 64, 128, 256, 512]
# - stride ∈ [1, 2]
# - max_pool_size ∈ [1, 2, 3]  (1 = no pooling)
```

### C. Estructura de Experimentos

```
experiments/
├── NASCNN15_baseline/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── last_checkpoint.pth
│   ├── plots/
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── architecture.png
│   ├── artifacts/
│   │   └── experiment_config.json
│   └── logs/
│       └── training.log
│
├── NAS_search_001/
│   ├── checkpoints/
│   │   ├── nas_episode_0.pth
│   │   ├── nas_episode_10.pth
│   │   └── ...
│   ├── discovered_architectures/
│   │   ├── arch_ep10_reward0.87.json
│   │   ├── arch_ep50_reward0.91.json
│   │   └── best_architecture.pth
│   └── search_logs/
│       ├── reward_history.csv
│       └── architecture_history.json
```

### D. Troubleshooting

#### Problema: Out of Memory (OOM)

**Síntomas**: `RuntimeError: CUDA out of memory`

**Soluciones**:
1. Reducir `batch_size`: 128 → 64 → 32
2. Usar gradient accumulation
3. Mixed precision training (FP16)

```python
# Gradient Accumulation
accumulation_steps = 4
for i, (images, labels) in enumerate(dataloader):
    loss = model(images, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Problema: NAS no converge

**Síntomas**: Rewards no mejoran después de muchos episodios

**Diagnóstico**:
1. Verificar baseline calculation (EMA)
2. Revisar learning rate del controller
3. Aumentar `children_per_episode`

**Solución**:
```python
# Aumentar exploración
config['children_per_episode'] = 20  # era 10

# Ajustar baseline
baseline = ema(rewards, alpha=0.9)  # era 0.95

# Warm-up learning rate
if episode < 100:
    lr = 0.01
else:
    lr = 0.001
```

#### Problema: NASCNN15 accuracy bajo

**Síntomas**: Accuracy < 85% en validation

**Diagnóstico**:
1. Verificar normalización correcta
2. Revisar augmentation
3. Comprobar learning rate schedule

**Solución**:
```python
# Verificar stats
mean, std, _ = compute_dataset_stats(datasets_folder)
print(f"Mean: {mean}, Std: {std}")

# Debe ser aproximadamente:
# Mean: [0.491, 0.482, 0.447]
# Std: [0.247, 0.243, 0.262]
```

---

## Conclusiones

Este documento proporciona una guía completa para:

1. **Refactorizar** el proyecto eliminando componentes no esenciales (CIFAR-10.1, arquitecturas extras)
2. **Integrar NAS con RL** migrando de TensorFlow a PyTorch
3. **Mantener NASCNN15** como arquitectura única del branch
4. **Implementar checkpoint support** para búsquedas de arquitectura de larga duración

### Estado Actual vs. Estado Objetivo

| Aspecto | Actual | Objetivo |
|---------|--------|----------|
| **Arquitecturas** | 5 (BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR, NASCNN15) | 1 (NASCNN15) |
| **Datasets** | CIFAR-10 + CIFAR-10.1 | CIFAR-10 únicamente |
| **NAS** | Separado (TensorFlow) | Integrado (PyTorch) |
| **Checkpoints** | Solo training | Training + NAS search |
| **Documentación** | README básico | Monografía completa |

### Próximos Pasos

1. **Crear branch**: `refactor/nascnn15-only`
2. **Ejecutar Sprint 1**: Limpieza de código (1 semana)
3. **Review checkpoint**: Validar que NASCNN15 funciona correctamente
4. **Comenzar Sprint 2**: Migración NAS (2 semanas)

### Contribuciones Esperadas

- **Código más limpio**: Mantenibilidad mejorada
- **NAS integrado**: Búsqueda de arquitectura reproducible
- **Documentación exhaustiva**: Facilita colaboración y extensión
- **Checkpoint support**: Experimentos de larga duración viables

---

**Documento vivo**: Este documento se actualizará conforme avance la implementación.

**Última actualización**: 21 de Noviembre de 2025  
**Versión**: 1.0  
**Autor**: Esp. Joaquín S Tschopp

