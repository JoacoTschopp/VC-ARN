from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision import datasets


@dataclass
class TransformConfig:
    img_size: int = 32
    use_random_horizontal_flip: bool = False
    use_random_resized_crop: bool = False
    use_color_jitter: bool = False
    jitter_brightness: float = 0.0
    jitter_contrast: float = 0.0
    jitter_saturation: float = 0.0
    jitter_hue: float = 0.0
    normalize: bool = True


def compute_dataset_stats(dataset_root: str) -> Tuple[list[float], list[float]]:
    """Calcula media y desvío estándar canalizados para CIFAR-10."""

    cifar10_training = datasets.CIFAR10(dataset_root, train=True, download=True)
    data = cifar10_training.data.astype(np.float32)
    mean = np.mean(data, axis=(0, 1, 2)) / 255.0
    std = np.std(data, axis=(0, 1, 2)) / 255.0
    return mean.tolist(), std.tolist()


def build_transforms(
    mean: list[float],
    std: list[float],
    config: TransformConfig,
) -> Tuple[T.Compose, T.Compose]:
    """Construye transformaciones de entrenamiento y validación/test."""

    train_transforms: list = []

    if config.use_random_resized_crop:
        train_transforms.append(T.RandomResizedCrop(config.img_size))
    else:
        train_transforms.append(T.Resize((config.img_size, config.img_size)))

    if config.use_random_horizontal_flip:
        train_transforms.append(T.RandomHorizontalFlip())

    if config.use_color_jitter:
        train_transforms.append(
            T.ColorJitter(
                brightness=config.jitter_brightness,
                contrast=config.jitter_contrast,
                saturation=config.jitter_saturation,
                hue=config.jitter_hue,
            )
        )

    train_transforms.extend(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    if config.normalize:
        train_transforms.append(T.Normalize(mean, std))

    test_transforms = [
        T.Resize((config.img_size, config.img_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]

    if config.normalize:
        test_transforms.append(T.Normalize(mean, std))

    return T.Compose(train_transforms), T.Compose(test_transforms)
