from pathlib import Path
from datetime import datetime
import time

import torch

from src.arqui_cnn import BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR, NASCNN15
from src.auxiliares import compare_models, draw_model, que_fierro_tengo
from src.load import load_cifar10, load_data
from src.pre_processed import config_augmentation
from src.test import run_cifar10_test_evaluation
from src.train_pipeline import TrainingPipeline


def format_elapsed_time(elapsed_seconds: float) -> str:
    """Formatea tiempo transcurrido en formato legible."""
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"





def main():
    # ============================================================================
    # INICIO DEL EXPERIMENTO
    # ============================================================================
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print("EXPERIMENTO INICIADO")
    print("="*70)
    print(f"Fecha y hora: {start_datetime}")
    print("="*70 + "\n")

    #Que fierro tengo??
    que_fierro_tengo()

    # Experimento Nombre y rutas de salida
    experiment_name = "NASCNN_V13_OnlyCIFAR10"
    experiments_root = Path("../experiments")
    experiment_dir = experiments_root / experiment_name

    checkpoints_dir = experiment_dir / "checkpoints"
    plots_dir = experiment_dir / "plots"
    artifacts_dir = experiment_dir / "artifacts"

    for directory in (experiment_dir, checkpoints_dir, plots_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Crear carpeta datasets
    datasets = Path("../datasets")
    
    # 1. Cargar datos
    augmentation_configs = config_augmentation()
    datasets_folder = load_data(datasets_folder=str(datasets))

    # Comparar arquitecturas
    compare_models()

    # Dibujar arquitectura
    draw_model(NASCNN15(), output_dir=artifacts_dir)

    # 2. Preprocesamiento de Datos
    # Cargamos los datos de entrenamiento, calculamos media y desvio para normalizar
    augmentation_configs = config_augmentation()
    cifar10_training, cifar10_validation, cifar10_test, training_transformations, test_transformations = load_cifar10(
        datasets_folder, config=augmentation_configs.config_cnn_nas
    )
    
    elapsed_prep = time.time() - start_time
    print("\n" + "="*70)
    print(f"✓ PREPROCESAMIENTO COMPLETADO")
    print(f"  Tiempo transcurrido: {format_elapsed_time(elapsed_prep)}")
    print("="*70 + "\n")

    # 3. Entrenamiento
    # ==============================================================================
    # CONFIGURACIÓN DE HIPERPARÁMETROS
    # ==============================================================================

    config = {
        'experiment_name': experiment_name,
        'lr': 0.1,
        'epochs': 300,
        'batch_size': 128,
        'es_patience': 20,
        'lr_scheduler': True,
        'lr_patience': 10,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'nesterov': True,
        'use_scheduler': True,
        'label_smoothing': 0.0,
        'optimizer': 'SGD',
        'base_dir': str(experiment_dir),
        'checkpoint_dir': str(checkpoints_dir),
        'experiment_dir': str(experiment_dir),
        'plots_dir': str(plots_dir),
        'artifacts_dir': str(artifacts_dir),
        'show_plots': False,          # o True para mostrarlos
        'plot_display_time': 5,       # opcional si quieres autocierre en segundos
    }

    # Actualizar variables globales para compatibilidad
    LR = config['lr']
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']

    print("="*70)
    print("CONFIGURACIÓN")
    print("="*70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)


    # ==============================================================================
    # PREPARACIÓN DE DATOS Y MODELO
    # ==============================================================================

    # Crear DataLoaders
    train_dataloader = torch.utils.data.DataLoader(
        cifar10_training, 
        batch_size=config['batch_size'], 
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        cifar10_validation, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    print("="*70)
    print("DATASET")
    print("="*70)
    print(f"✓ Train set: {len(train_dataloader.dataset)} imágenes")
    print(f"✓ Validation set: {len(validation_dataloader.dataset)} imágenes")
    print(f"✓ Test set: {len(cifar10_test)} imágenes")

    # Crear modelo
    #model = BaseModel()
    #model = SimpleCNN()
    #model = ImprovedCNN()
    #model = ResNetCIFAR()
    model = NASCNN15()

    print("="*70)
    print("SELECCIÓN DE MODELO")
    print("="*70)
    print(" ")
    print(f"✓ {model.__class__.__name__}")
    print("="*70)



    # Crear pipeline de entrenamiento
    pipeline = TrainingPipeline(model, config)

    print(f"✓ Pipeline inicializado")
    print(f"✓ Total de parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # ==============================================================================
    # ENTRENAMIENTO
    # ==============================================================================

    pipeline.train(train_dataloader, validation_dataloader)
    
    elapsed_train = time.time() - start_time
    print("\n" + "="*70)
    print(f"✓ ENTRENAMIENTO COMPLETADO")
    print(f"  Tiempo transcurrido: {format_elapsed_time(elapsed_train)}")
    print("="*70 + "\n")
    
    # ==============================================================================
    # REANUDAR ENTRENAMIENTO (si fue interrumpido)
    # ==============================================================================

    # Descomenta y ejecuta si fue interrumpido:
    # pipeline.resume_training('interrupted_checkpoint.pth', train_dataloader, validation_dataloader)

    print("! Para reanudar, descomenta la línea anterior y ejecuta esta celda")

    # ==============================================================================
    # VISUALIZACIÓN DE PREPROCESAMIENTOS E HIPERPARÁMETROS DEL ENTRENAMIENTO
    # ==============================================================================
    pipeline.register_experiment(training_transformations, test_transformations)

    # ==============================================================================
    # VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
    # ==============================================================================

    pipeline.plot_training_curves()

    # ==============================================================================
    # SUMARIZACIÓN DE EXPERIMENTOS
    # ==============================================================================
    summary = pipeline.summarize_experiments(sort_by="results.best_val_acc", top_k=5)
    
    elapsed_viz = time.time() - start_time
    print("\n" + "="*70)
    print(f"✓ VISUALIZACIONES COMPLETADAS")
    print(f"  Tiempo transcurrido: {format_elapsed_time(elapsed_viz)}")
    print("="*70 + "\n")

    # ==============================================================================
    # TEST EN CIFAR-10 TEST SET
    # ==============================================================================
    # Obtener mean y std del config de transformaciones
    from src.pre_processed import compute_dataset_stats
    mean, std, _ = compute_dataset_stats(
        datasets_folder, compute_zca=augmentation_configs.config_cnn_nas.use_whitening
    )
    run_cifar10_test_evaluation(
        pipeline, 
        cifar10_test, 
        mean=mean, 
        std=std, 
        batch_size=config['batch_size']
    )
    
    elapsed_total = time.time() - start_time
    print("\n" + "="*70)
    print(f"✓ TEST COMPLETADO")
    print(f"  Tiempo transcurrido: {format_elapsed_time(elapsed_total)}")
    print("="*70 + "\n")
    
    # ============================================================================
    # RESUMEN FINAL
    # ============================================================================
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("="*70)
    print("EXPERIMENTO FINALIZADO")
    print("="*70)
    print(f"Fecha y hora final: {end_datetime}")
    print(f"Tiempo total: {format_elapsed_time(elapsed_total)}")
    print("="*70 + "\n")
    
if __name__ == "__main__":
    main()
