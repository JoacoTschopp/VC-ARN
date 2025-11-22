"""
Arquitecturas CNN para Clasificación de Imágenes CIFAR-10

Este módulo contiene la implementación de NASCNN15, una arquitectura de 15 capas
descubierta mediante Neural Architecture Search con Reinforcement Learning.

Referencia:
    Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. ICLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# NAS-CNN v1 (Zoph & Le, 2017) - 15 capas sin stride ni pooling (Figura 7)
# Anchos por capa (N) en {36,48} y kernels FH×FW tal cual la imagen.
# Skips: se mantienen exactamente como en tu forward.
# ==============================================================================

class NASCNN15(nn.Module):
    """
    Reimplementación aproximada de la arquitectura de 15 capas para CIFAR-10
    (NAS v1, sin stride ni pooling, Figura 7 del paper).

    - Bloque por capa: Conv2d (bias=False) -> BatchNorm -> ReLU.
    - Resolución fija 32×32 (stride=1). El padding se elige para conservar tamaño:
      padding = ((FH-1)//2, (FW-1)//2) en kernels impares (incluye casos asimétricos).
    - Las capas con múltiples predecesoras concatenan sus entradas por el eje de canales.
    - Clasificador: Global Average Pooling (1×1) + Linear (con bias) → logits.
      Softmax solo en .predict() (para entrenamiento usar CrossEntropyLoss sobre logits).

    Detalle por capa (FH = filter height, FW = filter width, N = #filtros), tal cual Figura 7:
      C1 : FH=3, FW=3, N=36      ; in = imagen RGB  
      C2 : FH=3, FW=3, N=48      ; in = C1
      C3 : FH=3, FW=3, N=36      ; in = concat(C1, C2)
      C4 : FH=5, FW=5, N=36      ; in = concat(C1, C2, C3)
      C5 : FH=3, FW=7, N=48      ; in = concat(C3, C4)
      C6 : FH=7, FW=7, N=48      ; in = concat(C2, C3, C4, C5)
      C7 : FH=7, FW=7, N=48      ; in = concat(C2, C3, C4, C5, C6)
      C8 : FH=7, FW=3, N=36      ; in = concat(C1, C6, C7)
      C9 : FH=7, FW=1, N=36      ; in = concat(C1, C5, C6, C8)
      C10: FH=7, FW=7, N=36      ; in = concat(C1, C3, C4, C5, C6, C7, C8, C9)
      C11: FH=5, FW=7, N=36      ; in = concat(C1, C2, C5, C6, C7, C8, C9, C10)
      C12: FH=7, FW=7, N=48      ; in = concat(C1, C2, C3, C4, C6, C11)
      C13: FH=7, FW=5, N=48      ; in = concat(C1, C3, C6, C7, C8, C9, C10, C11, C12)
      C14: FH=7, FW=5, N=48      ; in = concat(C3, C7, C12, C13)
      C15: FH=7, FW=5, N=48      ; in = concat(C6, C7, C11, C12, C13, C14)

    Entrenamiento (setup fiel al paper NAS v1 para CIFAR-10):
      - Optimizador: SGD con Nesterov, lr inicial 0.1, momentum 0.9, weight decay 1e-4.
      - Pérdida: nn.CrossEntropyLoss (sin label smoothing).
      - Épocas: ~200–300 con scheduler que reduce lr.
      - Augment básico recomendado (flip horizontal, crop con padding).
    """


    def __init__(self, num_classes: int = 10):
        super().__init__()

        # N (out_channels) por capa según Figura 7:
        # 1:36, 2:48, 3:36, 4:36, 5:48, 6:48, 7:48, 8:36, 9:36,
        # 10:36, 11:36, 12:48, 13:48, 14:48, 15:48

        # C1: in=3,   out=36, k=3x3
        self.conv1 = nn.Conv2d(3, 36, kernel_size=(3,3), padding=(1,1), bias=False)
        self.bn1   = nn.BatchNorm2d(36)

        # C2: in=36,  out=48, k=3x3
        self.conv2 = nn.Conv2d(36, 48, kernel_size=(3,3), padding=(1,1), bias=False)
        self.bn2   = nn.BatchNorm2d(48)

        # C3: in=[x1(36), x2(48)] = 84,  out=36, k=3x3
        self.conv3 = nn.Conv2d(84, 36, kernel_size=(3,3), padding=(1,1), bias=False)
        self.bn3   = nn.BatchNorm2d(36)

        # C4: in=[x1(36), x2(48), x3(36)] = 120, out=36, k=5x5
        self.conv4 = nn.Conv2d(120, 36, kernel_size=(5,5), padding=(2,2), bias=False)
        self.bn4   = nn.BatchNorm2d(36)

        # C5: in=[x3(36), x4(36)] = 72,  out=48, k=3x7
        self.conv5 = nn.Conv2d(72, 48, kernel_size=(3,7), padding=(1,3), bias=False)
        self.bn5   = nn.BatchNorm2d(48)

        # C6: in=[x2(48), x3(36), x4(36), x5(48)] = 168, out=48, k=7x7
        self.conv6 = nn.Conv2d(168, 48, kernel_size=(7,7), padding=(3,3), bias=False)
        self.bn6   = nn.BatchNorm2d(48)

        # C7: in=[x2(48), x3(36), x4(36), x5(48), x6(48)] = 216, out=48, k=7x7
        self.conv7 = nn.Conv2d(216, 48, kernel_size=(7,7), padding=(3,3), bias=False)
        self.bn7   = nn.BatchNorm2d(48)

        # C8: in=[x1(36), x6(48), x7(48)] = 132, out=36, k=7x3
        self.conv8 = nn.Conv2d(132, 36, kernel_size=(7,3), padding=(3,1), bias=False)
        self.bn8   = nn.BatchNorm2d(36)

        # C9: in=[x1(36), x5(48), x6(48), x8(36)] = 168, out=36, k=7x1
        self.conv9 = nn.Conv2d(168, 36, kernel_size=(7,1), padding=(3,0), bias=False)
        self.bn9   = nn.BatchNorm2d(36)

        # C10: in=[x1(36), x3(36), x4(36), x5(48), x6(48), x7(48), x8(36), x9(36)] = 324,
        #      out=36, k=7x7
        self.conv10 = nn.Conv2d(324, 36, kernel_size=(7,7), padding=(3,3), bias=False)
        self.bn10   = nn.BatchNorm2d(36)

        # C11: in=[x1(36), x2(48), x5(48), x6(48), x7(48), x8(36), x9(36), x10(36)] = 336,
        #      out=36, k=5x7
        self.conv11 = nn.Conv2d(336, 36, kernel_size=(5,7), padding=(2,3), bias=False)
        self.bn11   = nn.BatchNorm2d(36)

        # C12: in=[x1(36), x2(48), x3(36), x4(36), x6(48), x11(36)] = 240,
        #      out=48, k=7x7
        self.conv12 = nn.Conv2d(240, 48, kernel_size=(7,7), padding=(3,3), bias=False)
        self.bn12   = nn.BatchNorm2d(48)

        # C13: in=[x1(36), x3(36), x6(48), x7(48), x8(36), x9(36), x10(36), x11(36), x12(48)] = 360,
        #      out=48, k=7x5
        self.conv13 = nn.Conv2d(360, 48, kernel_size=(7,5), padding=(3,2), bias=False)
        self.bn13   = nn.BatchNorm2d(48)

        # C14: in=[x3(36), x7(48), x12(48), x13(48)] = 180,
        #      out=48, k=7x5
        self.conv14 = nn.Conv2d(180, 48, kernel_size=(7,5), padding=(3,2), bias=False)
        self.bn14   = nn.BatchNorm2d(48)

        # C15: in=[x6(48), x7(48), x11(36), x12(48), x13(48), x14(48)] = 276,
        #      out=48, k=7x5
        self.conv15 = nn.Conv2d(276, 48, kernel_size=(7,5), padding=(3,2), bias=False)
        self.bn15   = nn.BatchNorm2d(48)

        # Clasificador: GAP + FC (logits). FC con bias (no hay BN después).
        self.fc = nn.Linear(48, num_classes)

        # Softmax sólo para .predict()
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # C1, C2
        x1 = F.relu(self.bn1(self.conv1(x)))         # [B, 36, 32, 32]
        x2 = F.relu(self.bn2(self.conv2(x1)))        # [B, 48, 32, 32]

        # C3
        x3_in = torch.cat([x1, x2], dim=1)
        x3 = F.relu(self.bn3(self.conv3(x3_in)))     # [B, 36, 32, 32]

        # C4
        x4_in = torch.cat([x1, x2, x3], dim=1)
        x4 = F.relu(self.bn4(self.conv4(x4_in)))     # [B, 36, 32, 32]

        # C5
        x5_in = torch.cat([x3, x4], dim=1)
        x5 = F.relu(self.bn5(self.conv5(x5_in)))     # [B, 48, 32, 32]

        # C6
        x6_in = torch.cat([x2, x3, x4, x5], dim=1)
        x6 = F.relu(self.bn6(self.conv6(x6_in)))     # [B, 48, 32, 32]

        # C7
        x7_in = torch.cat([x2, x3, x4, x5, x6], dim=1)
        x7 = F.relu(self.bn7(self.conv7(x7_in)))     # [B, 48, 32, 32]

        # C8
        x8_in = torch.cat([x1, x6, x7], dim=1)
        x8 = F.relu(self.bn8(self.conv8(x8_in)))     # [B, 36, 32, 32]

        # C9
        x9_in = torch.cat([x1, x5, x6, x8], dim=1)
        x9 = F.relu(self.bn9(self.conv9(x9_in)))     # [B, 36, 32, 32]

        # C10
        x10_in = torch.cat([x1, x3, x4, x5, x6, x7, x8, x9], dim=1)
        x10 = F.relu(self.bn10(self.conv10(x10_in))) # [B, 36, 32, 32]

        # C11
        x11_in = torch.cat([x1, x2, x5, x6, x7, x8, x9, x10], dim=1)
        x11 = F.relu(self.bn11(self.conv11(x11_in))) # [B, 36, 32, 32]

        # C12
        x12_in = torch.cat([x1, x2, x3, x4, x6, x11], dim=1)
        x12 = F.relu(self.bn12(self.conv12(x12_in))) # [B, 48, 32, 32]

        # C13
        x13_in = torch.cat([x1, x3, x6, x7, x8, x9, x10, x11, x12], dim=1)
        x13 = F.relu(self.bn13(self.conv13(x13_in))) # [B, 48, 32, 32]

        # C14
        x14_in = torch.cat([x3, x7, x12, x13], dim=1)
        x14 = F.relu(self.bn14(self.conv14(x14_in))) # [B, 48, 32, 32]

        # C15
        x15_in = torch.cat([x6, x7, x11, x12, x13, x14], dim=1)
        x15 = F.relu(self.bn15(self.conv15(x15_in))) # [B, 48, 32, 32]

        # Clasificador
        out = F.adaptive_avg_pool2d(x15, output_size=1)  # [B, 48, 1, 1]
        out = out.view(out.size(0), -1)                  # [B, 48]
        out = self.fc(out)                               # [B, num_classes] (logits)
        return out

    def predict(self, x):
        logits = self.forward(x)
        return self.final_activation(logits)