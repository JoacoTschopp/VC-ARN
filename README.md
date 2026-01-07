# CIFAR-10 Image Classification with Deep Learning

A proof-of-concept demonstrating advanced Convolutional Neural Network (CNN) architectures for image classification, featuring multiple model variants, comprehensive training pipeline, and a full-stack web application for real-time inference.

## 🎯 Project Objective

This project implements an end-to-end **deep learning system for image classification** using state-of-the-art CNN architectures. The main objective is to demonstrate practical application of Computer Vision techniques for accurate image classification on the CIFAR-10 benchmark dataset, achieving high accuracy through advanced training techniques including:

- Custom CNN architectures (SimpleCNN, ImprovedCNN, ResNet-based models)
- Advanced data augmentation strategies
- Training optimization with multiple schedulers and regularization techniques
- Production-ready deployment with FastAPI backend and interactive frontend

## 🚀 Technologies

### Backend & Deep Learning
- **PyTorch 2.0+** - Deep learning framework
- **TorchVision** - Computer vision utilities and datasets
- **FastAPI** - Modern, high-performance web framework for ML APIs
- **Uvicorn** - ASGI server for production deployment
- **NumPy** - Numerical computing
- **scikit-learn** - ML metrics and utilities

### Frontend & Visualization
- **D3.js** - Interactive data visualizations
- **HTML5/CSS3/JavaScript** - Modern web interface
- **Matplotlib/Seaborn** - Training curve visualizations

### Development Tools
- **Python 3.8+** - Programming language
- **Git** - Version control

## 📁 Project Structure

```
VC-ARN/
├── app/                           # Main application
│   ├── main.py                    # Training entry point
│   └── src/                       # Source code
│       ├── arqui_cnn.py          # CNN architectures
│       ├── train_pipeline.py     # Training pipeline
│       ├── load.py               # Data loading utilities
│       ├── pre_processed.py      # Data augmentation configs
│       ├── test.py               # Evaluation on CIFAR-10.1
│       └── auxiliares.py         # Helper functions
├── Graficos_Presentacion/        # Web application
│   ├── index.html                # Training metrics dashboard
│   ├── ejemplos.html             # Image classification demo
│   ├── app.js                    # Frontend logic
│   ├── predict_api.py            # FastAPI inference server
│   └── best_model.pth            # Trained model checkpoint
└── README.md                     # This file
```

## 🏗️ CNN Architectures

### 1. BaseModel
Simple fully-connected baseline
- 2 dense layers (3072 → 512 → 10)
- Tanh activation
- ~1.6M parameters
- Expected accuracy: ~45-50%

### 2. SimpleCNN
Basic convolutional network
- 3 convolutional blocks (Conv → ReLU → MaxPool)
- Channels: 3 → 32 → 64 → 128
- Dropout regularization (0.5)
- ~850K parameters
- Expected accuracy: ~65-70%

### 3. ImprovedCNN ⭐
Enhanced CNN with Batch Normalization
- 5 convolutional blocks with BatchNorm
- Channels: 3 → 64 → 128 → 256 → 256 → 512
- Dropout + BatchNorm regularization
- ~6.5M parameters
- Expected accuracy: ~75-85%

### 4. ResNetCIFAR
ResNet-inspired architecture with skip connections
- 3 groups of residual blocks
- Global Average Pooling
- BatchNorm in all convolutional layers
- Configurable depth
- ~300K parameters (default config)
- Expected accuracy: ~80-88%

## 📊 CIFAR-10 Dataset

- **60,000 color images** (32×32 pixels)
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Split**: 50,000 training / 10,000 test images
- **Balanced classes**: 6,000 images per class

### Challenges
- Small image resolution (32×32 pixels)
- Varied object orientations and scales
- Lighting and background variations
- Partial occlusions

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/VC-ARN.git
   cd VC-ARN
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎓 Training

### Quick Start

1. **Navigate to training directory**:
   ```bash
   cd app
   ```

2. **Run training**:
   ```bash
   python main.py
   ```

### Configuration

Edit hyperparameters in `app/main.py` (lines 92-115):

```python
config = {
    'lr': 0.1,                    # Learning rate
    'epochs': 200,                # Number of epochs
    'batch_size': 128,            # Batch size
    'es_patience': 15,            # Early stopping patience
    'optimizer': 'SGD',           # Optimizer type
    'momentum': 0.9,              # SGD momentum
    'weight_decay': 1e-4,         # L2 regularization
    # ... more options
}
```

### Select Model Architecture

In `app/main.py` (lines 158-163), uncomment desired model:

```python
# model = BaseModel()         # Baseline
# model = SimpleCNN()         # Basic CNN
model = ImprovedTwoCNN()      # Default: Enhanced CNN
# model = ResNetCIFAR()       # ResNet variant
```

### Training Outputs

- **Checkpoints**: Saved in `experiments/{experiment_name}/checkpoints/`
  - `best_model.pth` - Best validation accuracy
  - `last_checkpoint.pth` - Periodic checkpoints
  - `interrupted_checkpoint.pth` - Recovery checkpoint
- **Visualizations**: Training curves, confusion matrices in `plots/`
- **Logs**: Experiment tracking in `experiments_log.jsonl`

## 🧪 Testing & Evaluation

Automatic evaluation on CIFAR-10.1 test set runs after training, generating:
- Global accuracy metrics
- Per-class accuracy breakdown
- Confusion matrix
- Correct/incorrect prediction examples

## 🌐 Web Application (Inference Demo)

### Backend API (FastAPI)

1. **Copy trained model**:
   ```bash
   cp experiments/{experiment_name}/checkpoints/best_model.pth Graficos_Presentacion/
   ```

2. **Start API server**:
   ```bash
   cd Graficos_Presentacion
   source ../.venv/bin/activate  # Activate venv
   uvicorn predict_api:app --reload --port 8002
   ```

3. **API endpoints**:
   - `GET /health` - Health check
   - `POST /classify` - Image classification (base64 input)

### Frontend Dashboard

1. **Start metrics dashboard**:
   ```bash
   cd Graficos_Presentacion
   python -m http.server 8001
   ```

2. **Access dashboards**:
   - Training metrics: [http://localhost:8001/index.html](http://localhost:8001/index.html)
   - Image classifier: [http://localhost:8001/ejemplos.html](http://localhost:8001/ejemplos.html)

### Features
- **Metrics Dashboard**: Interactive D3.js visualizations of training curves
- **Image Classifier**: Upload custom images for real-time classification
- **Experiment Comparison**: Compare multiple training runs

## 📈 Training Optimizations

### Data Augmentation
- Random horizontal flip
- Random resized crop
- Color jitter
- AutoAugment policies
- ZCA whitening (optional)

### Regularization
- Dropout (0.3-0.5)
- Batch Normalization
- Label smoothing
- Early stopping
- L2 weight decay

### Optimizers & Schedulers
- **SGD** with Nesterov momentum
- **Adam**, **AdamW**, **RMSprop**
- **ReduceLROnPlateau** - Adaptive learning rate
- Automatic device detection (CUDA/MPS/CPU)

## 🔄 Resume Training

If training is interrupted:

```python
# In app/main.py, uncomment:
pipeline.resume_training('interrupted_checkpoint.pth', train_dataloader, validation_dataloader)
```

## 📝 Experiment Tracking

All experiments are logged to `experiments_log.jsonl` with:
- Model architecture and parameters
- Hyperparameter configuration
- Training curves and metrics
- Best validation accuracy
- Timestamp and experiment ID

## 🎯 Expected Results

| Model | Parameters | Validation Acc | Test Acc (CIFAR-10.1) |
|-------|-----------|----------------|------------------------|
| BaseModel | 1.6M | ~45-50% | ~40-45% |
| SimpleCNN | 850K | ~65-70% | ~60-65% |
| ImprovedCNN | 6.5M | ~75-85% | ~70-80% |
| ResNetCIFAR | 300K-2M | ~80-88% | ~75-85% |

## 🐛 Troubleshooting

### CUDA out of memory
```python
config['batch_size'] = 32  # Reduce batch size
```

### Missing dependencies
```bash
pip install torch torchvision fastapi uvicorn
```

### Port already in use
```bash
# Change port in uvicorn command
uvicorn predict_api:app --port 8003
```

## 📚 References

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10.1: https://github.com/modestyachts/CIFAR-10.1
- PyTorch: https://pytorch.org/
- FastAPI: https://fastapi.tiangolo.com/

---

**Note**: This is a proof-of-concept for educational and portfolio purposes, demonstrating end-to-end deep learning workflow from training to deployment.
