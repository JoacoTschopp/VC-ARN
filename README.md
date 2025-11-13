# Clasificación de Imágenes con Redes Neuronales Convolucionales (CNN)

## 🎯 Finalidad del Proyecto

Este proyecto implementa un **sistema de clasificación de imágenes** utilizando Redes Neuronales Convolucionales (CNN) avanzadas. El objetivo principal es demostrar la aplicación práctica de técnicas de Visión por Computadora para la clasificación precisa de imágenes en tiempo real, utilizando como caso de estudio el reconocimiento de objetos en el dataset CIFAR-10.

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### Instalación

1. **Clonar el repositorio** (o descargar como ZIP):

   ```bash
   git clone https://github.com/tu-usuario/VC-ARN.git
   cd VC-ARN
   ```
2. **Crear y activar un entorno virtual** (recomendado):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```
3. **Instalar dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

### Ejecución

1. **Preparar los datos**:

   ```bash
   python -m src.load
   ```
2. **Entrenar un modelo**:

   ```bash
   python main.py
   ```

## 🛠️ Herramientas Aplicadas

### Bibliotecas Principales

- **PyTorch**: Framework de aprendizaje profundo para la implementación de redes neuronales
- **TorchVision**: Para cargar y transformar conjuntos de datos de visión por computadora
- **NumPy**: Para operaciones numéricas eficientes
- **Matplotlib/Seaborn**: Para visualización de datos y resultados
- **scikit-learn**: Para métricas de evaluación y utilidades

### Técnicas de Programación

- **Pipeline con Programación Orientada a Objetos**:
  - Abstracciones para datasets, modelos y etapas del pipeline
  - Reutilización de componentes entre experimentos
  - Configuración declarativa de hiperparámetros

### Optimizadores Implementados

- **SGD** (con momentum y Nesterov)
- **Adam** (optimización adaptativa de gradientes)
- **RMSprop** (ajuste dinámico por parámetro)

### Programadores de Learning Rate

- **StepLR** (reducción por pasos)
- **ReduceLROnPlateau** (tasa adaptada al desempeño)
- **OneCycleLR** y **CyclicLR** (curvas cíclicas controladas)

### Arquitecturas Implementadas

1. **SimpleCNN**: Una red neuronal convolucional básica
2. **ImprovedCNN**: Versión mejorada con capas adicionales y regularización
3. **ResNetCIFAR**: Implementación de ResNet adaptada para CIFAR-10

## 📊 Sobre el Dataset CIFAR-10

### Características Principales

- **60,000 imágenes** a color de 32x32 píxeles
- **10 clases** diferentes de objetos
- División estándar: 50,000 para entrenamiento y 10,000 para prueba
- Clases balanceadas (6,000 imágenes por clase)

### Categorías

El dataset incluye las siguientes 10 categorías de objetos:

| ID | Categoría    | Ejemplos                       |
| -- | ------------- | ------------------------------ |
| 0  | ✈️ Avión   | Aviones, jets, avionetas       |
| 1  | 🚗 Automóvil | Coches, camionetas, furgonetas |
| 2  | 🐦 Pájaro    | Aves de diferentes especies    |
| 3  | 🐱 Gato       | Gatos domésticos              |
| 4  | 🦌 Ciervo     | Venados, corzos                |
| 5  | 🐕 Perro      | Perros de diferentes razas     |
| 6  | 🐸 Rana       | Ranas y sapos                  |
| 7  | 🐎 Caballo    | Caballos, ponis                |
| 8  | 🚢 Barco      | Barcos, botes, veleros         |
| 9  | 🚜 Camión    | Camiones, tráilers            |

### Desafíos

- Imágenes pequeñas (32x32 píxeles)
- Objetos en diferentes posiciones y ángulos
- Variaciones en la iluminación y el fondo
- Oclusión parcial en algunos casos

## 📁 Estructura del Proyecto

```
VC-ARN/
├── TP-FINAL/               # Código principal del proyecto
│   ├── datasets/           # Datasets descargados
│   ├── models/             # Modelos guardados
│   ├── src/                # Código fuente
│   │   ├── arqui_cnn.py    # Arquitecturas de redes
│   │   ├── auxiliares.py   # Funciones auxiliares
│   │   ├── load.py         # Carga de datos
│   │   ├── test.py         # Evaluación de modelos
│   │   └── train_pipeline.py # Pipeline de entrenamiento
│   ├── main.py             # Punto de entrada principal
│   └── VCBRNA-grupo-3.ipynb # Notebook de análisis
└── README.md               # Este archivo
```

## 👨‍💻 Autor

**Esp. Joaquín S Tschopp**

---

*Este proyecto demuestra la aplicación práctica de Redes Neuronales Convolucionales para la clasificación de imágenes, utilizando técnicas avanzadas de Deep Learning y Visión por Computadora.*
