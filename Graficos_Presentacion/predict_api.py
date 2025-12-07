from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
from PIL import Image

MODEL_PATH = Path(__file__).with_name("best_model.pth")
DEVICE = torch.device("mps")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES: List[str] = [
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
]

CLASS_EMOJIS: Dict[str, str] = {
    "avión": "✈️",
    "automóvil": "🚗",
    "pájaro": "🐦",
    "gato": "🐱",
    "ciervo": "🦌",
    "perro": "🐶",
    "rana": "🐸",
    "caballo": "🐴",
    "barco": "🚢",
    "camión": "🚚",
}

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)


class PredictionRequest(BaseModel):

    image: str


class PredictionResponse(BaseModel):
    label: str
    emoji: str
    confidence: float
    probabilities: List[float]

class ResidualBlock(nn.Module):
    """Bloque residual básico con skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection con ajuste de dimensión
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection
        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet adaptado para CIFAR-10 con skip connections

    Arquitectura:
    - Capa inicial convolucional
    - 3 grupos de bloques residuales
    - Global Average Pooling
    - Fully connected final

    Parámetros:
    - num_blocks: Lista con el número de bloques residuales por grupo (por defecto [2, 2, 2])
    """

    def __init__(self, num_blocks=[2, 2, 2]):
        super(ResNetCIFAR, self).__init__()

        # Capa inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Grupos de bloques residuales
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)

        # Clasificador
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

        # Activación final
        self.final_activation = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # Primer bloque puede cambiar dimensiones
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Bloques subsecuentes mantienen dimensiones
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Capa inicial
        x = F.relu(self.bn1(self.conv1(x)))

        # Bloques residuales
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Average Pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        # Clasificador
        x = self.fc(x)

        return x

    def predict(self, x):
        return self.final_activation(self.forward(x))


def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("El checkpoint no contiene 'model_state_dict'.")

    model = ResNetCIFAR(num_blocks=[9, 9, 9])
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def decode_image(data: str) -> Image.Image:
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(data)
    except Exception as exc:  # pragma: no cover - FastAPI maneja respuesta
        raise HTTPException(status_code=400, detail="Imagen base64 inválida") from exc

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen") from exc


model = load_model()
app = FastAPI(title="CIFAR10 Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/classify", response_model=PredictionResponse)
def classify_image(payload: PredictionRequest):
    image = decode_image(payload.image)
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    confidence, pred_idx = torch.max(probabilities, dim=0)
    label = CIFAR10_CLASSES[pred_idx.item()]
    emoji = CLASS_EMOJIS[label]

    return PredictionResponse(
        label=label,
        emoji=emoji,
        confidence=float(confidence.item()),
        probabilities=[float(p) for p in probabilities.tolist()],
    )
