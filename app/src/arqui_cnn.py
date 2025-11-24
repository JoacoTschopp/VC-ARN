import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# ARQUITECTURA DEL MODELO 1.0
# ==============================================================================


class BaseModel(nn.Module):
    """
    Modelo base con arquitectura fully connected de 2 capas.
    Para clasificación de imágenes en 10 clases usando datos de CIFAR10
    """

    def __init__(self):
        super(BaseModel, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 10)
        )
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        """Forward pass"""
        return self.model(inputs)

    def predict(self, inputs):
        """Predicción con softmax"""
        return self.final_activation(self.model(inputs))


# ==============================================================================


# ==============================================================================
# OPCIÓN 1: CNN Simple
# ==============================================================================
class SimpleCNN(nn.Module):
    """
    CNN básica con 3 bloques convolucionales

    Arquitectura:
    - 3 bloques Conv -> ReLU -> MaxPool
    - 2 capas fully connected
    - Dropout para regularización

    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Bloque 1: 3 -> 32 canales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Bloque 2: 32 -> 64 canales
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Bloque 3: 64 -> 128 canales
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4

        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        # Regularización
        self.dropout = nn.Dropout(0.5)

        # Activación final
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # Bloque 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Bloque 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Bloque 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(-1, 128 * 4 * 4)

        # FC con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        return self.final_activation(self.forward(x))


# ==============================================================================
# OPCIÓN 2: CNN Mejorada con Batch Normalization
# ==============================================================================
class ImprovedCNN(nn.Module):
    """
    CNN con Batch Normalization y arquitectura más profunda

    Arquitectura:
    - 4 bloques Conv -> BatchNorm -> ReLU -> MaxPool
    - 2 capas fully connected con BatchNorm
    - Dropout para regularización

    Parámetros: ~340K
    Accuracy esperado: ~75-80%

    """

    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Bloque 1: 3 -> 64 canales
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Bloque 2: 64 -> 128 canales
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Bloque 3: 128 -> 256 canales
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Bloque 4: 256 -> 256 canales
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Bloque 5: 256 -> 512 canales
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4

        # Fully connected con BatchNorm
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

        # Regularización
        self.dropout = nn.Dropout(0.5)

        # Activación final
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # Bloque 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Bloque 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # Bloque 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Bloque 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout(x)

        # Bloque 5
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(-1, 512 * 4 * 4)

        # FC con BatchNorm y Dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        return self.final_activation(self.forward(x))


# ==============================================================================
# OPCIÓN 3: ResNet-like con Skip Connections
# ==============================================================================
class ResidualBlock(nn.Module):
    """Bloque residual básico con skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection con ajuste de dimensión
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
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

    """

    def __init__(self, num_blocks=[2, 2, 2]):
        super(ResNetCIFAR, self).__init__()

        # Capa inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

        # Global Average Pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        # Clasificador
        x = self.fc(x)

        return x

    def predict(self, x):
        return self.final_activation(self.forward(x))


##################################################################################################

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

# ==============================================================================
# OPCIÓN 2: CNN Mejorada con Batch Normalization
# ==============================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, p_dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout2d(p_dropout) if use_dropout and p_dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedTwoCNN(nn.Module):
    def __init__(self, num_classes=10, base_width=64, p_dropout=0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        # 32x32, canales = base_width
        self.layer1 = nn.Sequential(
            BasicBlock(base_width, base_width, stride=1, use_dropout=False),
            BasicBlock(base_width, base_width, stride=1, use_dropout=False),
        )

        # 16x16, canales = 2*base_width
        self.layer2 = nn.Sequential(
            BasicBlock(base_width, base_width * 2, stride=2, use_dropout=True, p_dropout=p_dropout),
            BasicBlock(base_width * 2, base_width * 2, stride=1, use_dropout=False),
        )

        # 8x8, canales = 4*base_width
        self.layer3 = nn.Sequential(
            BasicBlock(base_width * 2, base_width * 4, stride=2, use_dropout=True, p_dropout=p_dropout),
            BasicBlock(base_width * 4, base_width * 4, stride=1, use_dropout=False),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 4, num_classes)
        
        # Softmax sólo para .predict()
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def predict(self, x):
        return self.final_activation(self.forward(x))