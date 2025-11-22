"""
Sistema de Entrenamiento y Búsqueda de Arquitecturas para CIFAR-10

Este script proporciona dos modos de operación:
    1. NAS Search: Búsqueda automática de arquitecturas con RL
    2. NASCNN15 Training: Entrenamiento completo de NASCNN15

Uso:
    # Búsqueda NAS (modo rápido para pruebas)
    python main.py --mode nas --config fast --episodes 10 --children 3
    
    # Búsqueda NAS (modo completo)
    python main.py --mode nas --config default
    
    # Entrenamiento NASCNN15
    python main.py --mode train
    
    # Reanudar búsqueda NAS
    python main.py --mode nas --resume checkpoints/nas/nas_episode_50.pth
"""

from pathlib import Path
from datetime import datetime
import time
import argparse

import torch

from src.arqui_cnn import NASCNN15
from src.auxiliares import draw_model, que_fierro_tengo, show_nascnn15_info
from src.load import load_cifar10
from src.pre_processed import config_augmentation, TransformConfig
from src.test import run_cifar10_test_evaluation
from src.train_pipeline import TrainingPipeline
from src.nas import NASTrainer, get_nas_config


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





def run_nas_search(args):
    """
    Ejecuta búsqueda de arquitecturas con NAS.
    
    Args:
        args: Argumentos de línea de comandos
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print("🔍 MODO: NEURAL ARCHITECTURE SEARCH")
    print("="*70)
    print(f"Configuración: {args.config}")
    print(f"Episodios: {args.episodes or 'default'}")
    print(f"Children/episodio: {args.children or 'default'}")
    print("="*70 + "\n")
    
    # Hardware info
    que_fierro_tengo()
    
    # Crear carpeta datasets
    datasets_folder = Path("../datasets")
    datasets_folder.mkdir(parents=True, exist_ok=True)
    
    # Cargar configuración NAS
    config = get_nas_config(args.config)
    
    # Cargar datos
    print("\n" + "="*70)
    print("CARGANDO DATOS")
    print("="*70)
    
    train_dataset, val_dataset, _, _, _ = load_cifar10(
        str(datasets_folder),
        config=TransformConfig()
    )
    
    # DataLoaders
    batch_size = config.get('child_batch_size', 20)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✓ Train: {len(train_dataset)} imágenes")
    print(f"✓ Val: {len(val_dataset)} imágenes")
    print(f"✓ Batch size: {batch_size}")
    print("="*70 + "\n")
    
    # Crear NAS Trainer
    trainer = NASTrainer(config)
    
    # Reanudar si se especificó checkpoint
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    
    # Ejecutar búsqueda
    trainer.search(
        train_loader,
        val_loader,
        num_episodes=args.episodes,
        children_per_episode=args.children
    )
    
    # Resumen final
    elapsed_total = time.time() - start_time
    print("\n" + "="*70)
    print("🏁 BÚSQUEDA NAS FINALIZADA")
    print("="*70)
    print(f"Tiempo total: {format_elapsed_time(elapsed_total)}")
    print(f"Mejor reward: {trainer.best_reward:.4f}")
    print(f"Checkpoints: {trainer.checkpoint_dir}")
    print(f"Logs: {trainer.log_file}")
    print("="*70 + "\n")


def run_nascnn15_training(args):
    """
    Ejecuta entrenamiento completo de NASCNN15.
    
    Args:
        args: Argumentos de línea de comandos
    """
    # ============================================================================
    # INICIO DEL EXPERIMENTO
    # ============================================================================
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print("🎯 MODO: ENTRENAMIENTO NASCNN15")
    print("="*70)
    print(f"Fecha y hora: {start_datetime}")
    print("="*70 + "\n")

    # Que fierro tengo??
    que_fierro_tengo()

    # Experimento Nombre y rutas (simplificadas)
    experiment_name = "NASCNN15_Production"
    experiments_root = Path("../experiments")
    experiment_dir = experiments_root / experiment_name

    # Solo crear las carpetas que realmente se usan
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Crear carpeta datasets
    datasets_folder = Path("../datasets")
    datasets_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Información de la arquitectura
    show_nascnn15_info()

    # Dibujar arquitectura (guardará en experiment_dir)
    draw_model(NASCNN15(), output_dir=experiment_dir)

    # 2. Preprocesamiento de Datos
    # Cargamos los datos de entrenamiento, calculamos media y desvio para normalizar
    augmentation_configs = config_augmentation()
    cifar10_training, cifar10_validation, cifar10_test, training_transformations, test_transformations = load_cifar10(
        str(datasets_folder), config=augmentation_configs.config_cnn_nas
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
        'checkpoint_dir': str(experiment_dir),
        'experiment_dir': str(experiment_dir),
        'show_plots': False,
        'plot_display_time': 5,
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
    
def main():
    """Punto de entrada principal con CLI."""
    parser = argparse.ArgumentParser(
        description='Sistema de Entrenamiento y Búsqueda de Arquitecturas para CIFAR-10',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            Ejemplos:
            # Búsqueda NAS rápida (pruebas)
            python main.py --mode nas --config fast --episodes 10 --children 3
            
            # Búsqueda NAS completa
            python main.py --mode nas --config default
            
            # Entrenamiento NASCNN15
            python main.py --mode train
            
            # Reanudar búsqueda NAS
            python main.py --mode nas --resume checkpoints/nas/nas_episode_50.pth
        '''
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['nas', 'train'],
        help='Modo de operación: "nas" para búsqueda de arquitecturas, "train" para entrenar NASCNN15'
    )
    
    # Argumentos específicos de NAS
    nas_group = parser.add_argument_group('Argumentos para modo NAS')
    nas_group.add_argument(
        '--config',
        type=str,
        default='fast',
        choices=['default', 'fast', 'thorough'],
        help='Configuración de NAS (default: fast)'
    )
    nas_group.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Número de episodios (override config)'
    )
    nas_group.add_argument(
        '--children',
        type=int,
        default=None,
        help='Arquitecturas por episodio (override config)'
    )
    nas_group.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path a checkpoint para reanudar NAS'
    )
    
    args = parser.parse_args()
    
    # Ejecutar modo seleccionado
    if args.mode == 'nas':
        run_nas_search(args)
    elif args.mode == 'train':
        run_nascnn15_training(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
