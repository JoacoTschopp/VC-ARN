# Pipeline de CNN para Clasificación de Imágenes CIFAR-10

Proyecto de clasificación de imágenes utilizando redes neuronales convolucionales (CNN) sobre el dataset CIFAR-10 y evaluación en CIFAR-10.1.

## 📋 Descripción

Este proyecto implementa un pipeline completo de entrenamiento, validación y evaluación de modelos CNN para clasificación de imágenes. Incluye:

- 4 arquitecturas de red diferentes
- Pipeline de entrenamiento con early stopping
- Sistema de checkpoints automático
- Evaluación en CIFAR-10.1
- Visualizaciones profesionales de métricas

## 🏗️ Arquitecturas Disponibles

### 1. **BaseModel**
Modelo baseline con arquitectura fully connected simple.
- 2 capas densas (3072 → 512 → 10)
- Activación Tanh
- ~1.6M parámetros
- Accuracy esperado: ~45-50%

### 2. **SimpleCNN**
CNN básica con 3 bloques convolucionales.
- 3 bloques: Conv → ReLU → MaxPool
- Canales: 3 → 32 → 64 → 128
- 2 capas fully connected
- Dropout (0.5) para regularización
- ~850K parámetros
- Accuracy esperado: ~65-70%

### 3. **ImprovedCNN** ⭐ (Recomendada)
CNN mejorada con Batch Normalization.
- 5 bloques convolucionales con BatchNorm
- Canales: 3 → 64 → 128 → 256 → 256 → 512
- Dropout entre capas
- BatchNorm en capas convolucionales y fully connected
- ~6.5M parámetros
- Accuracy esperado: ~75-85%

### 4. **ResNetCIFAR**
Arquitectura tipo ResNet con skip connections.
- Bloques residuales con shortcuts
- 3 grupos de bloques (64, 128, 256 canales)
- Global Average Pooling
- BatchNorm en todas las capas convolucionales
- ~300K parámetros
- Accuracy esperado: ~80-88%

## 📦 Instalación

### Requisitos
- Python 3.8+
- pip o conda

### Instalar dependencias

```bash
# Opción 1: Usando pip
pip install -r requirements.txt

# Opción 2: Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
# .\.venv\Scripts\Activate.ps1  # En Windows
pip install -r requirements.txt
```

### Dependencias principales
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- torchview >= 0.2.6 (opcional, para visualización de arquitecturas)

## 🚀 Ejecución

### Estructura del proyecto

```
src/
├── main.py                 # Script principal de ejecución
├── arqui_cnn.py           # Definición de arquitecturas CNN
├── train_pipeline.py      # Pipeline de entrenamiento
├── load.py                # Carga de datasets
├── pre_processed.py       # Preprocesamiento y transformaciones
├── test.py                # Evaluación en CIFAR-10.1
├── auxiliares.py          # Funciones auxiliares
├── models.py              # Enumeradores y configuraciones
├── requirements.txt       # Dependencias
└── pyproject.toml         # Configuración de herramientas
```

### Ejecutar el proyecto completo

```bash
# Activar el entorno virtual (si se creó)
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows (PowerShell)

# Ejecutar el pipeline completo
python main.py
```

### Configuración de hiperparámetros

Edita la configuración en `main.py` (líneas 35-43):

```python
config = {
    'lr': 0.001,              # Learning rate
    'epochs': 100,            # Número de épocas
    'batch_size': 64,         # Tamaño del batch
    'patience': 10,           # Early stopping patience
    'checkpoint_dir': 'models/',  # Directorio de checkpoints
    'optimizer': 'AdamW',     # Optimizador: 'SGD', 'Adam', 'AdamW', 'RMSProp'
}
```

### Seleccionar arquitectura

En `main.py` (líneas 78-82), descomenta el modelo deseado:

```python
# model = BaseModel()         # Baseline simple
# model = SimpleCNN()         # CNN básica
model = ImprovedCNN()         # CNN mejorada (por defecto)
# model = ResNetCIFAR()       # ResNet adaptado
```

## 📊 Resultados y Checkpoints

Durante el entrenamiento, se generan automáticamente:

### Checkpoints
- `models/best_model.pth` - Mejor modelo según accuracy de validación
- `models/last_checkpoint.pth` - Checkpoint cada 5 épocas
- `models/interrupted_checkpoint.pth` - Si se interrumpe con Ctrl+C

### Visualizaciones
- Curvas de entrenamiento (Loss y Accuracy)
- Matriz de confusión en CIFAR-10.1
- Ejemplos de predicciones correctas/incorrectas
- Medida de overfitting

## 🔄 Reanudar Entrenamiento

Si el entrenamiento se interrumpe, puedes reanudarlo:

```python
# En main.py, descomenta la línea:
pipeline.resume_training('interrupted_checkpoint.pth', train_dataloader, validation_dataloader)
```

## 🎯 Evaluación

El pipeline incluye evaluación automática en CIFAR-10.1:
- Accuracy global
- Accuracy por clase
- Matriz de confusión
- Visualización de predicciones

## 🔧 Funciones Auxiliares

### Comparar arquitecturas
```python
from auxiliares import compare_models
compare_models()  # Muestra parámetros de todas las arquitecturas
```

### Detectar hardware disponible
```python
from auxiliares import que_fierro_tengo
que_fierro_tengo()  # Muestra GPU/CPU disponible
```

### Visualizar arquitectura
```python
from auxiliares import draw_model
from arqui_cnn import ImprovedCNN

model = ImprovedCNN()
draw_model(model)  # Requiere torchview instalado
```

## 📈 Mejores Prácticas

1. **Data Augmentation**: El preprocesamiento incluye:
   - Random horizontal flip
   - Random resized crop
   - Normalización con media y std de CIFAR-10

2. **Regularización**:
   - Dropout (0.5)
   - Batch Normalization
   - Label smoothing (0.05)
   - Early stopping

3. **Optimización**:
   - Soporte para múltiples optimizadores
   - Detección automática de GPU (CUDA/MPS)
   - Checkpoints automáticos

## 🐛 Troubleshooting

### Error: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error: "CUDA out of memory"
Reduce el batch_size en la configuración:
```python
config['batch_size'] = 32  # o 16
```

### Warning: "torchview no está instalado"
```bash
pip install torchview
```
Esto solo afecta la visualización de arquitecturas, el entrenamiento funcionará normalmente.

## 📚 Referencias

- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
- **CIFAR-10.1**: https://github.com/modestyachts/CIFAR-10.1
- **PyTorch**: https://pytorch.org/

## 👥 Autores

Proyecto desarrollado para la materia de Visión por Computadora - UBA

## 💾 Hacer Commits

El proyecto usa pre-commit hooks para validar código:

```bash
# Agregar todos los cambios
git add -A

# Hacer commit (usa --no-verify si pre-commit falla)
git commit -m "Tu mensaje"

# Si pre-commit modifica archivos, agregarlos y commitear nuevamente
git add -A
git commit -m "Tu mensaje"
```

**Nota**: Si pre-commit entra en conflicto, usa `git commit --no-verify` para saltarlo.

## 📄 Licencia

Este proyecto es de uso académico.
