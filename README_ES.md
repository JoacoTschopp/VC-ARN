# Búsqueda de Arquitecturas Neuronales con Aprendizaje por Refuerzo para CIFAR-10

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Implementación completa de **Neural Architecture Search (NAS)** utilizando el algoritmo **REINFORCE** para descubrir automáticamente arquitecturas CNN óptimas para la clasificación de CIFAR-10. Este proyecto migra el código original en TensorFlow 1.x a PyTorch moderno, incorporando mejoras orientadas a investigación y producción.

## 🎯 Resumen del Proyecto

El sistema NAS implementado utiliza una red recurrente (Controlador LSTM) para generar arquitecturas neuronales y la entrena con aprendizaje por refuerzo para maximizar la exactitud de validación. El sistema se valida en CIFAR-10 e incluye la arquitectura NASCNN15 descubierta mediante este proceso.

### Funcionalidades Clave

- ✅ **NAS con REINFORCE** para generar arquitecturas
- ✅ **Controlador LSTM** que produce secuencias de ADN arquitectónico
- ✅ **Constructor dinámico de CNN** a partir de ADN
- ✅ **Pipeline de entrenamiento** con early stopping y scheduler de LR
- ✅ **Checkpoints** con capacidad de reanudación
- ✅ **Logging narrativo** para documentación de investigaciones
- ✅ **CLI con dos modos** (búsqueda NAS + entrenamiento NASCNN15)
- ✅ **Código listo para producción** con pruebas exhaustivas (históricas)

## 📚 NAS Explicado

### ¿Qué es NAS?

**Neural Architecture Search** es una técnica de AutoML que explora automáticamente el espacio de arquitecturas para encontrar redes neuronales de alto desempeño sin intervención manual.

### ¿Cómo funciona?

Este proyecto sigue la propuesta de [&#34;Neural Architecture Search with Reinforcement Learning&#34;](https://arxiv.org/abs/1611.01578) (Zoph & Le, 2017):

![NAS Architecture Overview](https://miro.medium.com/max/656/1*hIif88uJ7Te8MJEhm40rbw.png)

**Flujo del proceso:**

1. **Controlador (LSTM)** genera descripciones de arquitectura (ADN)
2. **ADN** codifica la estructura de la red: `[kernel, filtros, stride, pool, ...]`
3. **Red hija** se construye dinámicamente y se entrena en CIFAR-10
4. **Exactitud de validación** se usa como recompensa
5. **REINFORCE** actualiza el Controlador para mejorar futuras arquitecturas

![NAS Training Process](https://i.ytimg.com/vi/CYUpDogeIL0/maxresdefault.jpg)

### Codificación ADN

```python
# ADN para una CNN de 3 capas:
DNA = [
    [3, 64,  1, 1],   # Capa 1: kernel 3x3, 64 filtros, stride=1, sin pooling
    [5, 128, 1, 2],   # Capa 2: kernel 5x5, 128 filtros, stride=1, pooling 2x2
    [3, 256, 1, 1]    # Capa 3: kernel 3x3, 256 filtros, stride=1, sin pooling
]
```

**Componentes:**

- **Tamaño de kernel:** 1-7
- **Número de filtros:** 32-512
- **Stride:** 1-2
- **Pool:** 1-3 (1 = sin pooling)

### Algoritmo REINFORCE

El Controlador se entrena con gradiente de políticas:

```
∇J(θ) = E[R × ∇log P(a|θ)]
```

donde **R** es la recompensa (exactitud de validación).

### Arquitectura NASCNN15

Arquitectura de 15 capas descubierta para CIFAR-10:

```
Entrada (3×32×32)
  ↓
Conv 1: 3×3, 36 filtros
  ↓
Conv 2: 3×3, 48 filtros
  ↓
...
Conv 15: 7×5, 48 filtros
  ↓
Global Average Pooling
  ↓
Capa totalmente conectada (10 clases)
```

**Características:**

- Sin stride ni pooling (resolución fija 32×32)
- Conexiones densas por concatenación
- Kernels variados (1×1 a 7×7)
- BatchNorm + ReLU tras cada convolución

## 🚀 Inicio Rápido

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/VC-ARN.git
   cd VC-ARN
   ```
2. Crear entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### CLI (dos modos)

#### Modo 1: Búsqueda NAS

- **Prueba rápida (5-10 min):**
  ```bash
  cd app
  python main.py --mode nas --config fast --episodes 5 --children 2
  ```
- **Búsqueda corta (experimental):**
  ```bash
  python main.py --mode nas --config fast --episodes 50 --children 5
  ```
- **Búsqueda completa (producción):**
  ```bash
  python main.py --mode nas --config default
  ```
- **Reanudar desde checkpoint:**
  ```bash
  python main.py --mode nas --resume checkpoints/nas/nas_episode_50.pth
  ```

#### Modo 2: Entrenar NASCNN15

```bash
python main.py --mode train
```

#### Ayuda del CLI

```bash
python main.py --help
```

### Configuraciones

| Configuración | Episodios | Hijos/episodio | Épocas por hijo | Arquitecturas totales | Tiempo aprox. |
| -------------- | --------- | -------------- | ---------------- | --------------------- | ------------- |
| `fast`       | 100       | 5              | 20               | 500                   | 2-3 horas     |
| `default`    | 2,000     | 10             | 100              | 20,000                | 100-150 horas |
| `thorough`   | 5,000     | 15             | 150              | 75,000                | 500-600 horas |

## 📁 Estructura del Proyecto

```
VC-ARN/
├── app/
│   ├── main.py                          # CLI (NAS + entrenamiento)
│   └── src/
│       ├── arqui_cnn.py                 # Arquitectura NASCNN15
│       ├── load.py                      # Carga y split de CIFAR-10
│       ├── train_pipeline.py            # Orquestador de entrenamiento
│       ├── auxiliares.py                # Funciones auxiliares
│       ├── pre_processed.py             # Utilidades de preprocesamiento
│       └── nas/                         # Módulo NAS
│           ├── __init__.py
│           ├── configs.py               # Configuraciones NAS
│           ├── utils.py                 # Utilidades ADN
│           ├── controller.py            # Controlador LSTM
│           ├── child_builder.py         # Constructor de CNN
│           ├── reinforce.py             # Optimizador REINFORCE
│           └── trainer.py               # Orquestador NAS
│
├── datasets/                            # Cache de CIFAR-10 (generado)
├── experiments/                         # Salidas y checkpoints
├── Salidas_Experimentos/                # Exportaciones legacy
├── README.md                            # Referencia en inglés
└── README_ES.md                         # Este archivo
```

## 🔧 Detalles Técnicos

### Componentes del módulo NAS

1. **Controller (`controller.py`)**
   - LSTM de 11K parámetros
   - Genera secuencias ADN
2. **Child Builder (`child_builder.py`)**
   - Construye CNN a partir del ADN
   - Normaliza y valida rangos
3. **REINFORCE (`reinforce.py`)**
   - Gradiente de políticas con baseline EMA
   - Regularización L2 y clipping de gradientes
4. **NAS Trainer (`trainer.py`)**
   - Orquesta el ciclo completo
   - Logging narrativo + checkpoints + reanudación
5. **Utilities (`utils.py`)**
   - Encode/decode ADN
   - ADN aleatorio y representaciones legibles

### Pipeline de entrenamiento

`TrainingPipeline` ofrece:

- Early stopping configurable
- ReduceLROnPlateau y otros schedulers
- Guardado de checkpoints y métricas

### Sistema de logging

Niveles jerárquicos con íconos:

- 📋 INFO, ✅ SUCCESS, 🔹 STEP, 📊 METRIC
- ❌ ERROR, 🏗️ ARCHITECTURE, 🎯 TRAINING, 🏆 REWARD

## 📊 CIFAR-10

- 60,000 imágenes 32×32 (color)
- 10 clases balanceadas
- 50k train / 10k test
- Desafíos: tamaño pequeño, variaciones de vista y luz, occlusiones

## 📈 Resultados

### NASCNN15 (baseline)

| Métrica                | Valor     |
| ----------------------- | --------- |
| Parámetros             | ~1.9M     |
| Exactitud test          | ~92.5%    |
| Tiempo de entrenamiento | 4-6 horas |

### Salidas de NAS

- `checkpoints/nas/nas_final.pth`
- `checkpoints/nas/best_architecture.json`
- `logs/nas/nas_search_*.log`

## 🧪 Pruebas

Las pruebas automatizadas históricas se eliminaron junto con `test_nas_module.py`. Actualmente se recomienda validar mediante:

- Carga de CIFAR-10 (`load.py`)
- Sampleo del Controller (`controller.py`)
- Entrenamiento corto de NASCNN15 (`main.py --mode train`)

## 📖 Documentación

- `README.md`: Referencia principal en inglés
- `README_ES.md`: Resumen en español
- Docstrings detallados en cada módulo del paquete `src/`

## 🛠️ Tecnologías

- PyTorch 2.0+, TorchVision, NumPy
- Matplotlib/Seaborn, scikit-learn
- Optimizadores: SGD, Adam, RMSprop
- Schedulers: StepLR, ReduceLROnPlateau, OneCycleLR

## 🎓 Aplicaciones Académicas

Ideal para:

- Investigación en NAS y AutoML
- Estudios comparativos de arquitecturas
- Cursos avanzados de Deep Learning
- Proyectos de tesis/monografías

### Para publicaciones

Los logs narrativos facilitan documentar:

- Metodología completa
- Evolución de recompensas
- Descubrimiento de arquitecturas

## 🔬 Extensiones

- Ajustar límites ADN en `configs.py`
- Agregar nuevos bloques en `child_builder.py`
- Experimentar con PPO/A2C reemplazando `reinforce.py`

## 📝 Citas

```
@misc{nascnn2025,
  author = {Tschopp, Joaquín S.},
  title = {Neural Architecture Search with Reinforcement Learning for CIFAR-10},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tu-usuario/VC-ARN}
}
```

## 👨‍💻 Autor

**Esp. Joaquín S. Tschopp**

Especialista en Data Scientist

## 📄 Licencia

Proyecto bajo licencia MIT.

## 🙏 Agradecimientos

- Paper original de Zoph & Le
- Equipo PyTorch
- Creadores del dataset CIFAR-10
- Comunidad open source

## 🔗 Recursos

- [Paper original](https://arxiv.org/abs/1611.01578)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Documentación PyTorch](https://pytorch.org/docs/)
- [Video REINFORCE](https://www.youtube.com/watch?v=CYUpDogeIL0)

**Estado del proyecto:** 🟢 Listo para producción
**Última actualización:** 22 de noviembre de 2025
