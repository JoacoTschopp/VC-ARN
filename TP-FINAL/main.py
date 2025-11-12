import torch

from src.arqui_cnn import BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR
from src.auxiliares import compare_models, draw_model, que_fierro_tengo
from src.load import load_cifar10, load_data
from src.pre_processed import TransformConfig
from src.test import run_cifar101_evaluation
from src.train_pipeline import TrainingPipeline

def main():
    print("Iniciando el proyecto")

    #Que fierro tengo??
    que_fierro_tengo()

    # 1. Cargar datos
    datasets_folder = load_data()


    # Comparar arquitecturas
    compare_models()

    # Dibujar arquitectura
    draw_model(SimpleCNN())

    # 2. Preprocesamiento de Datos

    # Cargamos los datos de entrenamiento, calculamos media y desvio para normalizar
    cifar10_training, cifar10_validation = load_cifar10(datasets_folder)

    # 3. Entrenamiento
    # ==============================================================================
    # CONFIGURACIÓN DE HIPERPARÁMETROS
    # ==============================================================================

    config = {
        'lr': 0.001,
        'epochs': 100,
        'batch_size': 64,
        'patience': 10,
        #'momentum': 0.9, #Solo para SGD
        'checkpoint_dir': 'models/',
        'optimizer': 'AdamW',
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

    print(f"✓ Train set: {len(train_dataloader.dataset)} imágenes")
    print(f"✓ Validation set: {len(validation_dataloader.dataset)} imágenes")

    # Crear modelo
    #model = BaseModel()
    #model = SimpleCNN()
    model = ImprovedCNN()
    #model = ResNetCIFAR()

    # Crear pipeline de entrenamiento
    pipeline = TrainingPipeline(model, config)

    print(f"✓ Pipeline inicializado")
    print(f"✓ Total de parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # ==============================================================================
    # ENTRENAMIENTO
    # ==============================================================================

    pipeline.train(train_dataloader, validation_dataloader)
    
    # ==============================================================================
    # REANUDAR ENTRENAMIENTO (si fue interrumpido)
    # ==============================================================================

    # Descomenta y ejecuta si fue interrumpido:
    # pipeline.resume_training('interrupted_checkpoint.pth', train_dataloader, validation_dataloader)

    print("! Para reanudar, descomenta la línea anterior y ejecuta esta celda")

    # ==============================================================================
    # VISUALIZACIÓN DE PREPROCESAMIENTOS E HIPERPARÁMETROS DEL ENTRENAMIENTO
    # ==============================================================================

    # pipeline.describe_pipeline(cifar10_training.transform, cifar10_validation.transform)

    # ==============================================================================
    # VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
    # ==============================================================================

    pipeline.plot_training_curves()


    # 4. Test
    run_cifar101_evaluation(pipeline, datasets_folder)
    
if __name__ == "__main__":
    main()
