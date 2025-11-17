from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import Transform
from torchvision import datasets


@dataclass
class TransformConfig:
    img_size: int = 32

    # Upsampling previo
    use_upsample: bool = False
    upsample_size: int = 36

    # Geométricas básicas
    use_random_resized_crop: bool = False
    use_random_crop_with_padding: bool = False
    crop_padding: int = 4

    use_random_horizontal_flip: bool = False
    random_horizontal_flip_prob: float = 0.5

    use_random_rotation: bool = False
    rotation_degrees: float = 15.0

    # Políticas avanzadas
    use_autoaugment: bool = False
    use_trivial_augment: bool = False

    # Fotométricas simples
    use_color_jitter: bool = False
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.1

    # Regularización en imagen
    use_random_erasing: bool = False
    random_erasing_p: float = 0.25

    normalize: bool = True
    use_whitening: bool = False
    whitening_eps: float = 1e-6


class ZCAWhitening(Transform):
    """Aplica ZCA Whitening utilizando estadísticas precomputadas del dataset."""

    def __init__(self, mean: torch.Tensor, whitening_matrix: torch.Tensor):
        super().__init__()
        if mean.dim() != 1:
            raise ValueError("El vector de media debe ser 1D")
        if whitening_matrix.dim() != 2:
            raise ValueError("La matriz de whitening debe ser 2D")
        if whitening_matrix.size(0) != whitening_matrix.size(1):
            raise ValueError("La matriz de whitening debe ser cuadrada")
        if whitening_matrix.size(0) != mean.numel():
            raise ValueError("Dimensiones incompatibles entre media y matriz ZCA")

        self.register_buffer("mean", mean.float())
        self.register_buffer("whitening_matrix", whitening_matrix.float())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError("ZCAWhitening espera tensores de tipo torch.Tensor")

        original_shape = image.shape
        flat = image.reshape(-1).float()
        centered = flat - self.mean
        whitened = torch.matmul(self.whitening_matrix, centered)
        return whitened.reshape(original_shape).to(image.dtype)


def compute_dataset_stats(dataset_root: str, compute_zca: bool = False, eps: float = 1e-5):
    cifar10_training = datasets.CIFAR10(dataset_root, train=True, download=True)
    data = cifar10_training.data  # (50000, 32, 32, 3), uint8

    mean = np.mean(data, axis=(0, 1, 2)) / 255.0
    std = np.std(data, axis=(0, 1, 2)) / 255.0

    zca_params = None
    if compute_zca:
        data_float = data.astype(np.float32) / 255.0  # NHWC
        data_chw = np.transpose(data_float, (0, 3, 1, 2))  # -> NCHW (C,H,W)
        num_samples = data_chw.shape[0]
        flat = data_chw.reshape(num_samples, -1)

        zca_mean = flat.mean(axis=0)
        flat_centered = flat - zca_mean

        covariance = np.matmul(flat_centered.T, flat_centered) / num_samples
        eigvals, eigvecs = np.linalg.eigh(covariance)
        whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T

        zca_params = {
            "mean": zca_mean.astype(np.float32),
            "matrix": whitening_matrix.astype(np.float32),
        }

    return mean.tolist(), std.tolist(), zca_params


def build_transforms(mean, std, config: TransformConfig, zca_params=None):
    """
    Construye transformaciones usando la nueva API torchvision.transforms.v2.
    Pensado para CIFAR10 (32x32).
    """
    train_transforms = []

    # -------------------------
    # 1) Geométricas iniciales
    # -------------------------
    if config.use_random_resized_crop:
        train_transforms.append(T.RandomResizedCrop(config.img_size))
    else:
        # Upsampling previo opcional
        base_size = config.upsample_size if config.use_upsample else config.img_size
        train_transforms.append(T.Resize((base_size, base_size)))

        if config.use_random_crop_with_padding:
            train_transforms.append(
                T.RandomCrop(config.img_size, padding=config.crop_padding)
            )
        elif config.use_upsample and base_size > config.img_size:
            train_transforms.append(T.RandomCrop(config.img_size))
        elif base_size != config.img_size:
            train_transforms.append(T.Resize((config.img_size, config.img_size)))

    if config.use_random_horizontal_flip:
        train_transforms.append(
            T.RandomHorizontalFlip(config.random_horizontal_flip_prob)
        )

    if config.use_random_rotation:
        train_transforms.append(
            T.RandomRotation(degrees=config.rotation_degrees)
        )

    # -------------------------
    # 2) Políticas de AutoAugment
    #    (incluyen color jitter, etc.)
    # -------------------------
    if config.use_autoaugment:
        train_transforms.append(
            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10)
        )
    elif config.use_trivial_augment:
        train_transforms.append(T.TrivialAugmentWide())

    # -------------------------
    # 3) Fotométricas simples
    #    (solo si NO usamos una política automática)
    # -------------------------
    if config.use_color_jitter and not (config.use_autoaugment or config.use_trivial_augment):
        train_transforms.append(
            T.ColorJitter(
                brightness=config.jitter_brightness,
                contrast=config.jitter_contrast,
                saturation=config.jitter_saturation,
                hue=config.jitter_hue,
            )
        )

    # -------------------------
    # 4) Conversión a tensor + normalización
    # -------------------------
    train_transforms.extend([
        T.ToImage(),                              # PIL/ndarray -> Tensor (C,H,W)
        T.ToDtype(torch.float32, scale=True),     # Escala a [0,1] y dtype float32
    ])

    if config.use_whitening:
        if zca_params is None:
            raise ValueError("Se solicitó ZCA Whitening pero no se proporcionaron parámetros precomputados")
        mean_tensor = torch.from_numpy(zca_params["mean"]).clone()
        matrix_tensor = torch.from_numpy(zca_params["matrix"]).clone()
        train_transforms.append(ZCAWhitening(mean_tensor, matrix_tensor))

    if config.normalize:
        train_transforms.append(T.Normalize(mean, std))

    # -------------------------
    # 5) Regularización final
    # -------------------------
    if config.use_random_erasing:
        train_transforms.append(
            T.RandomErasing(p=config.random_erasing_p)
        )

    train_transform = T.Compose(train_transforms)

    # Transformaciones de test: sin augmentations pesadas
    test_transform = T.Compose([
        T.Resize((config.upsample_size if config.use_upsample else config.img_size,
                  config.upsample_size if config.use_upsample else config.img_size)),
        T.CenterCrop(config.img_size) if config.use_upsample else T.Identity(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        (ZCAWhitening(torch.from_numpy(zca_params["mean"]).clone(),
                      torch.from_numpy(zca_params["matrix"]).clone())
         if config.use_whitening else T.Identity()),
        T.Normalize(mean, std) if config.normalize else T.Identity(),
    ])

    return train_transform, test_transform


# ==============================================================================
# VARIANTES DE DATA AUGMENTATION
# ==============================================================================

class config_augmentation:
    def __init__(self):
        self.config_sin_augmentation = TransformConfig() # Por defecto solo normaliza

        # Crop+Padding y Horizontal Flip
        self.config_light = TransformConfig(
            img_size=32,
            use_random_resized_crop=False,
            use_random_crop_with_padding=True,
            crop_padding=4,
            use_random_horizontal_flip=True,
            random_horizontal_flip_prob=0.5,
            use_random_rotation=False,
            use_autoaugment=False,
            use_trivial_augment=False,
            use_color_jitter=False,
            use_random_erasing=False,
            normalize=True,
        )

        # AutoAugment + RandomErasing
        self.config_autoaugment = TransformConfig(
            img_size=32,
            use_random_resized_crop=False,
            use_random_crop_with_padding=True,
            crop_padding=4,
            use_random_horizontal_flip=True,
            random_horizontal_flip_prob=0.5,
            use_random_rotation=False,
            use_autoaugment=True,
            use_trivial_augment=False,
            use_color_jitter=False,    # lo hace AutoAugment
            use_random_erasing=True,
            random_erasing_p=0.25,
            normalize=True,
        )

        # Variante geométrica+color jitter
        self.config_geometrica = TransformConfig(
            img_size=32,
            use_random_resized_crop=False,
            use_random_crop_with_padding=True,
            crop_padding=4,
            use_random_horizontal_flip=True,
            random_horizontal_flip_prob=0.5,
            use_random_rotation=True,
            rotation_degrees=15,
            use_autoaugment=False,
            use_trivial_augment=False,
            use_color_jitter=True,
            jitter_brightness=0.3,
            jitter_contrast=0.3,
            jitter_saturation=0.3,
            jitter_hue=0.1,
            use_random_erasing=True,
            random_erasing_p=0.15,
            normalize=True,
        )

        # NAS-CNN: upsample + random crop + flip + whitening
        self.config_cnn_nas = TransformConfig(
            img_size=32,
            use_upsample=True,
            upsample_size=40,
            use_random_resized_crop=False,
            use_random_crop_with_padding=True,
            crop_padding=4,
            use_random_horizontal_flip=True,
            random_horizontal_flip_prob=0.5,
            use_random_rotation=False,
            use_autoaugment=False,
            use_trivial_augment=False,
            use_color_jitter=False,
            use_random_erasing=False,
            normalize=False,
            use_whitening=True,
            whitening_eps=1e-6,
        )