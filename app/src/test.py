from datetime import datetime
from pathlib import Path

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
    config=None,
):
    """Ejecuta la evaluación completa sobre CIFAR-10.1."""

    data = load_cifar101(
        datasets_folder=datasets_folder,
        batch_size=batch_size,
        shuffle=shuffle,
        config=config,
    )

    dataloader = data["dataloader"]
    labels = data["labels"]
    mean = np.array(data["mean"], dtype=np.float32)
    std = np.array(data["std"], dtype=np.float32)

    print(f"✓ {len(data['dataset'])} imágenes preprocesadas")

    results = pipeline.evaluate(dataloader, dataset_name="CIFAR-10.1")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = getattr(pipeline, "artifacts_dir", None)
    base_dir = Path(base_dir) if base_dir is not None else Path("experiments")
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / f"cifar101_evaluation_{timestamp}.txt"

    total_samples = len(results["labels"])
    correct_predictions = int((results["predictions"] == results["labels"]).sum())

    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(message: str = ""):
            print(message)
            log_file.write(message + "\n")

        log("\n" + "=" * 70)
        log("EVALUACIÓN CIFAR-10.1")
        log("=" * 70)
        log(f"Accuracy global: {results['accuracy']:.2%}")
        log(f"Correctas: {correct_predictions}/{total_samples}")

        log("\nACCURACY POR CLASE")
        log("=" * 70)
        for idx, class_name in enumerate(CLASS_NAMES):
            mask = labels == idx
            if mask.sum() > 0:
                class_acc = (
                    results["predictions"][mask] == labels[mask]
                ).sum() / mask.sum()
                log(
                    f"  {class_name:12s}: {class_acc:6.2%}  ({mask.sum():4d} samples)"
                )
        log("=" * 70)

        log("\n4. Generando matriz de confusión...")
        pipeline.plot_confusion_matrix(
            results["predictions"],
            results["labels"],
            CLASS_NAMES,
        )

        log("\n5. Mostrando ejemplos de predicciones...")
        images = data["images"].astype(np.float32) / 255.0  # (N, H, W, C)
        images = np.transpose(images, (0, 3, 1, 2))          # (N, C, H, W)
        mean_broadcast = mean.reshape(1, 3, 1, 1)
        std_broadcast = std.reshape(1, 3, 1, 1)
        images = (images - mean_broadcast) / std_broadcast

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

        log("\n" + "=" * 70)
        log("EVALUACIÓN COMPLETADA")
        log("=" * 70)
        log(f"Reporte guardado en: {log_path}")

    results["report_path"] = str(log_path)
    return results
