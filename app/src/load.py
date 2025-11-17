# Configuración de carpetas para uso local
import os
import urllib.request
from os import makedirs, path

import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from .pre_processed import config_augmentation

from .pre_processed import TransformConfig, build_transforms, compute_dataset_stats


class Cifar101Dataset(Dataset):
    """Dataset wrapper para CIFAR-10.1 almacenado en archivos .npy."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Descargar archivos si no existen
def download_file(url, filename):
    if not path.exists(filename):
        print(f"Descargando {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ {filename} descargado")
    else:
        print(f"✓ {filename} ya existe")


def load_data(datasets_folder: str | None = None) -> str:
    """
    Descarga los datos de CIFAR10.1 si no existen
    """

    # Carpeta local donde van a guardar los datos
    if datasets_folder is None:
        datasets_folder = "../datasets"
    makedirs(datasets_folder, exist_ok=True)

    # Rutas de los archivos
    data_file = path.join(datasets_folder, "cifar10.1_v4_data.npy")
    labels_file = path.join(datasets_folder, "cifar10.1_v4_labels.npy")

    # URLs de descarga
    data_url = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy"
    labels_url = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy"

    download_file(data_url, data_file)
    download_file(labels_url, labels_file)

    # Listar archivos en la carpeta
    print(f"\nArchivos en {datasets_folder}:")
    for item in os.listdir(datasets_folder):
        item_path = path.join(datasets_folder, item)
    if path.isfile(item_path):
        size_mb = path.getsize(item_path) / (1024 * 1024)
        print(f"  - {item} ({size_mb:.2f} MB)")
    else:
        print(f"  - {item}/ (directorio)")

    return datasets_folder


def load_cifar10(
    datasets_folder: str, config: TransformConfig | None = None
) -> tuple[datasets.CIFAR10, datasets.CIFAR10, dict]:
    """Carga CIFAR-10, transformaciones y estadísticas asociadas."""

    config = config or TransformConfig()

    mean, std, zca_params = compute_dataset_stats(
        datasets_folder, compute_zca=config.use_whitening
    )
    training_transformations, test_transformations = build_transforms(
        mean, std, config, zca_params=zca_params
    )

    train_dataset = datasets.CIFAR10(
        datasets_folder, train=True, download=True, transform=training_transformations
    )
    val_dataset = datasets.CIFAR10(
        datasets_folder, train=False, download=True, transform=test_transformations
    )

    return train_dataset, val_dataset, training_transformations, test_transformations


def load_cifar101(
    datasets_folder: str,
    batch_size: int = 64,
    shuffle: bool = False,
    config: TransformConfig | None = None,
):
    """Carga CIFAR-10.1 y construye un `DataLoader` listo para usar.

    Retorna un diccionario con los objetos más utilizados (`dataloader`, `dataset`,
    `images`, `labels`, `transform`, `mean`, `std`).
    """

    data_file = path.join(datasets_folder, "cifar10.1_v4_data.npy")
    labels_file = path.join(datasets_folder, "cifar10.1_v4_labels.npy")

    images = np.load(data_file)
    labels = np.load(labels_file)

    print("\n" + "=" * 70)
    print("CIFAR-10.1 DATASET")
    print("=" * 70)
    print(f"Shape de imágenes: {images.shape}")
    print(f"Shape de labels: {labels.shape}")
    print(f"Total de imágenes de test: {len(images)}")
    print("=" * 70)

    config = config or TransformConfig()
    mean, std, zca_params = compute_dataset_stats(
        datasets_folder, compute_zca=config.use_whitening
    )
    _, test_transform = build_transforms(mean, std, config, zca_params=zca_params)

    dataset = Cifar101Dataset(images, labels, test_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return {
        "dataloader": dataloader,
        "dataset": dataset,
        "images": images,
        "labels": labels,
        "transform": test_transform,
        "mean": mean,
        "std": std,
    }
