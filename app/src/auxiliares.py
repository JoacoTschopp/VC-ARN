"""
Utilidades auxiliares para el proyecto NASCNN15

Funciones para visualización de arquitecturas, detección de hardware,
y despliegue de información del modelo.
"""

from pathlib import Path

import torch
import torch.nn as nn
from .arqui_cnn import NASCNN15

try:
    from torchview import draw_graph
    TORCHVIEW_AVAILABLE = True
except ImportError:
    TORCHVIEW_AVAILABLE = False


# ==============================================================================
# FUNCIÓN: Información de NASCNN15
# ==============================================================================
def show_nascnn15_info():
    """Muestra información detallada de la arquitectura NASCNN15."""
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
    print("  - Kernels: 1×1, 3×3, 3×7, 5×5, 5×7, 7×1, 7×3, 7×5, 7×7")
    print("  - Filtros: 36 o 48 por capa")
    print("  - Resolución constante: 32×32 (sin stride/pooling)")
    print("  - Accuracy esperado: 91%+ en CIFAR-10 test")
    print()
    print("Referencia:")
    print("  Zoph, B., & Le, Q. V. (2017)")
    print("  Neural Architecture Search with Reinforcement Learning. ICLR.")
    print("=" * 70)


def draw_model(model: nn.Module, output_dir=None):
    """Dibuja la arquitectura de un modelo"""
    if not TORCHVIEW_AVAILABLE:
        print("! torchview no está instalado. Instalar con: pip install torchview")
        return None
    model_graph = draw_graph(model, input_size=(1, 3, 32, 32), expand_nested=True)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "model.png"
        model_graph.visual_graph.save(file_path)
        print(f"✓ Arquitectura guardada en {file_path}")
    return model_graph.visual_graph


def que_fierro_tengo():
    """Detecta y muestra el dispositivo disponible (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU NVIDIA disponible: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ GPU Apple Silicon (MPS) disponible")
    else:
        device = torch.device("cpu")
        print("✓ Usando CPU")

    # Crear un tensor de prueba
    test_tensor = torch.randn(10, 10).to(device)
    print(f"✓ Tensor de prueba creado en: {test_tensor.device}")
    return device
