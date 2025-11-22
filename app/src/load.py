"""
Carga y preparación de datos para CIFAR-10

Este módulo maneja la carga del dataset CIFAR-10 con split estratificado
según la metodología del paper de Zoph & Le (2017).

Split utilizado:
    - 45,000 imágenes de entrenamiento
    - 5,000 imágenes de validación (estratificado por clase)
    - 10,000 imágenes de test
"""

from os import makedirs

import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from .pre_processed import TransformConfig, build_transforms, compute_dataset_stats


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
