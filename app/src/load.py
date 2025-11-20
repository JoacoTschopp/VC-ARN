# Configuración de carpetas para uso local
import os
import urllib.request
from os import makedirs, path

import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
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
) -> tuple[Subset, Subset, datasets.CIFAR10, dict, dict]:
    """
    Carga CIFAR-10 siguiendo la metodología del paper de Zoph:
    - 50,000 imágenes de entrenamiento total
    - 5,000 imágenes para validación (muestra estratificada: 500 por clase)
    - 45,000 imágenes para entrenamiento
    - 10,000 imágenes para test (conjunto de test oficial de CIFAR-10)
    
    Returns:
        train_dataset: Subset con 45,000 imágenes de entrenamiento
        val_dataset: Subset con 5,000 imágenes de validación
        test_dataset: Dataset con 10,000 imágenes de test
        training_transformations: Transformaciones para entrenamiento
        test_transformations: Transformaciones para validación y test
    """
    
    config = config or TransformConfig()

    mean, std, zca_params = compute_dataset_stats(
        datasets_folder, compute_zca=config.use_whitening
    )
    training_transformations, test_transformations = build_transforms(
        mean, std, config, zca_params=zca_params
    )

    # Cargar el dataset completo de entrenamiento sin transformaciones primero
    # para obtener los labels y hacer el split estratificado
    temp_dataset = datasets.CIFAR10(
        datasets_folder, train=True, download=True, transform=None
    )
    
    # Obtener todos los labels del dataset de entrenamiento
    all_labels = np.array([label for _, label in temp_dataset])
    all_indices = np.arange(len(temp_dataset))
    
    # Realizar split estratificado: 45,000 train / 5,000 val
    # test_size=5000 nos da 5,000 para validación
    # random_state=811219 es la semilla para reproducibilidad
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=5000,
        stratify=all_labels,
        random_state=811219
    )
    
    print("\n" + "="*70)
    print("SPLIT DE DATOS CIFAR-10 (según paper de Zoph)")
    print("="*70)
    print(f"Total de imágenes en train original: {len(temp_dataset)}")
    print(f"Imágenes para entrenamiento: {len(train_indices)}")
    print(f"Imágenes para validación: {len(val_indices)}")
    
    # Verificar distribución estratificada en validación
    val_labels = all_labels[val_indices]
    unique, counts = np.unique(val_labels, return_counts=True)
    print("\nDistribución de clases en validación:")
    for label, count in zip(unique, counts):
        print(f"  Clase {label}: {count} imágenes")
    print("="*70)
    
    # Cargar datasets con las transformaciones apropiadas
    train_full_dataset = datasets.CIFAR10(
        datasets_folder, train=True, download=True, transform=training_transformations
    )
    val_full_dataset = datasets.CIFAR10(
        datasets_folder, train=True, download=True, transform=test_transformations
    )
    test_dataset = datasets.CIFAR10(
        datasets_folder, train=False, download=True, transform=test_transformations
    )
    
    # Crear subsets usando los índices estratificados
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    return train_dataset, val_dataset, test_dataset, training_transformations, test_transformations


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
