from pathlib import Path

import torch
import torch.nn as nn
from .arqui_cnn import BaseModel, ImprovedCNN, ResNetCIFAR, SimpleCNN, NASCNN15

try:
    from torchview import draw_graph

    TORCHVIEW_AVAILABLE = True
except ImportError:
    TORCHVIEW_AVAILABLE = False


# ==============================================================================
# HELPER FUNCTION: Compare architectures
# ==============================================================================
def compare_models():
    """Compare the 4 available architectures"""
    models = {
        "BaseModel": BaseModel(),
        "SimpleCNN": SimpleCNN(),
        "ImprovedCNN": ImprovedCNN(),
        "ResNetCIFAR": ResNetCIFAR(),
        "NASCNN15": NASCNN15(),
    }

    print("=" * 70)
    print("COMPARACIÓN DE ARQUITECTURAS")
    print("=" * 70)

    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{name}")
        print(f"  Parámetros totales: {total_params:,}")
        print(f"  Parámetros entrenables: {trainable_params:,}")

        # Calculate size in MB
        param_size = total_params * 4 / (1024**2)  # 4 bytes per parameter (float32)
        print(f"  Tamaño estimado: {param_size:.2f} MB")

    print("\n" + "=" * 70)
    print("RECOMENDACIÓN: ImprovedCNN para mejor balance complejidad/rendimiento")
    print("=" * 70)

    print("✓ Arquitecturas CNN cargadas exitosamente")
    print("\nPara comparar modelos ejecuta: compare_models()")


def draw_model(model: nn.Module, output_dir=None):
    """Draw model architecture"""
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
    """Detect and display available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU NVIDIA disponible: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ GPU Apple Silicon (MPS) disponible")
    else:
        device = torch.device("cpu")
        print("✓ Usando CPU")

    # Create test tensor
    test_tensor = torch.randn(10, 10).to(device)
    print(f"✓ Tensor de prueba creado en: {test_tensor.device}")
    return device
