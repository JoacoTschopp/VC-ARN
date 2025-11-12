import numpy as np

from .load import load_cifar101

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def run_cifar101_evaluation(
    pipeline,
    datasets_folder: str,
    batch_size: int = 64,
    shuffle: bool = False,
):
    """Ejecuta la evaluación completa sobre CIFAR-10.1."""

    data = load_cifar101(
        datasets_folder=datasets_folder,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    dataloader = data["dataloader"]
    labels = data["labels"]
    mean = np.array(data["mean"], dtype=np.float32)
    std = np.array(data["std"], dtype=np.float32)

    print(f"✓ {len(data['dataset'])} imágenes preprocesadas")

    results = pipeline.evaluate(dataloader, dataset_name="CIFAR-10.1")

    print("\n" + "=" * 70)
    print("ACCURACY POR CLASE")
    print("=" * 70)
    for idx, class_name in enumerate(CLASS_NAMES):
        mask = labels == idx
        if mask.sum() > 0:
            class_acc = (
                results["predictions"][mask] == labels[mask]
            ).sum() / mask.sum()
            print(f"  {class_name:12s}: {class_acc:6.2%}  ({mask.sum():4d} samples)")
    print("=" * 70)

    print("\n4. Generando matriz de confusión...")
    pipeline.plot_confusion_matrix(
        results["predictions"], results["labels"], CLASS_NAMES
    )

    print("\n5. Mostrando ejemplos de predicciones...")
    images = data["images"].astype(np.float32) / 255.0
    images = (images - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    images = np.transpose(images, (0, 3, 1, 2))

    pipeline.plot_examples(
        images,
        results["predictions"],
        results["labels"],
        CLASS_NAMES,
        mean,
        std,
        n_correct=10,
        n_incorrect=10,
    )

    print("\n" + "=" * 70)
    print("EVALUACIÓN COMPLETADA")
    print("=" * 70)

    return results
