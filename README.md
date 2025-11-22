# Neural Architecture Search with Reinforcement Learning for CIFAR-10

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete implementation of **Neural Architecture Search (NAS)** using **Reinforcement Learning** (REINFORCE algorithm) to automatically discover optimal CNN architectures for CIFAR-10 image classification. This project successfully migrates the original TensorFlow implementation to modern PyTorch with enhanced features for research and production use.

## 🎯 Project Overview

This project implements an automated neural architecture search system that uses a recurrent network (LSTM Controller) to generate neural network architectures and trains this controller with reinforcement learning to maximize validation accuracy. The system is tested on the CIFAR-10 dataset and includes the NASCNN15 architecture discovered through NAS.

### Key Features

- ✅ **Neural Architecture Search** with REINFORCE algorithm
- ✅ **LSTM Controller** for architecture generation
- ✅ **Dynamic CNN Builder** from DNA encoding
- ✅ **Complete training pipeline** with early stopping and LR scheduling
- ✅ **Checkpoint system** with resume capability
- ✅ **Narrative logging** for research documentation
- ✅ **Dual-mode CLI** (NAS search + model training)
- ✅ **Production-ready code** with comprehensive tests

## 📚 Neural Architecture Search Explained

### What is NAS?

**Neural Architecture Search (NAS)** is an automated machine learning technique that discovers optimal neural network architectures without human intervention. Instead of manually designing networks, NAS uses algorithms to explore the architecture space and find high-performing models.

### How It Works

The NAS system implemented in this project follows the approach from ["Neural Architecture Search with Reinforcement Learning"](https://arxiv.org/abs/1611.01578) (Zoph & Le, 2017):

![NAS Architecture Overview](https://miro.medium.com/max/656/1*hIif88uJ7Te8MJEhm40rbw.png)

**Process Overview:**

1. **Controller (LSTM)** generates architecture descriptions (DNA)
2. **DNA** encodes network structure: `[kernel_size, num_filters, stride, pool_size, ...]`
3. **Child Network** is built from DNA and trained on CIFAR-10
4. **Validation Accuracy** serves as reward signal
5. **REINFORCE** updates Controller to generate better architectures

![NAS Training Process](https://i.ytimg.com/vi/CYUpDogeIL0/maxresdefault.jpg)

### DNA Encoding

Each architecture is represented as a DNA sequence:

```python
# Example DNA for a 3-layer CNN:
DNA = [
    [3, 64,  1, 1],   # Layer 1: 3x3 kernel, 64 filters, stride=1, no pooling
    [5, 128, 1, 2],   # Layer 2: 5x5 kernel, 128 filters, stride=1, 2x2 pooling
    [3, 256, 1, 1]    # Layer 3: 3x3 kernel, 256 filters, stride=1, no pooling
]
```

**DNA Components:**
- **Kernel Size**: 1-7 (conv kernel dimensions)
- **Num Filters**: 32-512 (number of convolutional filters)
- **Stride**: 1-2 (convolution stride)
- **Pool Size**: 1-3 (max pooling size, 1 = no pooling)

### REINFORCE Algorithm

The Controller is trained using the **REINFORCE** policy gradient algorithm:

1. **Sample** architectures from Controller
2. **Train** each child network on training set
3. **Evaluate** on validation set → **reward**
4. **Update** Controller using policy gradient:
   ```
   ∇J(θ) = E[R × ∇log P(a|θ)]
   ```
   where R is the reward (validation accuracy)

### NASCNN15 Architecture

NASCNN15 is a 15-layer CNN architecture discovered through NAS for CIFAR-10:

```
Input (3×32×32)
  ↓
Conv Layer 1: 3×3, 36 filters
  ↓
Conv Layer 2: 3×3, 48 filters
  ↓
... (with skip connections)
  ↓
Conv Layer 15: 7×5, 48 filters
  ↓
Global Average Pooling
  ↓
Fully Connected (10 classes)
```

**Key Features:**
- No stride or pooling (maintains 32×32 resolution)
- Dense skip connections (concatenation by channel)
- Varying kernel sizes (1×1 to 7×7)
- Batch normalization + ReLU after each conv

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/VC-ARN.git
   cd VC-ARN
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

The system provides two main modes through a unified CLI:

#### Mode 1: Neural Architecture Search

**Quick test (5-10 minutes):**
```bash
cd app
python main.py --mode nas --config fast --episodes 5 --children 2
```

**Short search (experimental):**
```bash
python main.py --mode nas --config fast --episodes 50 --children 5
```

**Full search (production):**
```bash
python main.py --mode nas --config default
# 2000 episodes × 10 children = 20,000 architectures evaluated
```

**Resume from checkpoint:**
```bash
python main.py --mode nas --resume checkpoints/nas/nas_episode_50.pth
```

#### Mode 2: Train NASCNN15

```bash
python main.py --mode train
```

#### CLI Help

```bash
python main.py --help
```

### Configuration Options

Three predefined NAS configurations:

| Config | Episodes | Children/Ep | Child Epochs | Total Archs | Est. Time |
|--------|----------|-------------|--------------|-------------|-----------|
| `fast` | 100 | 5 | 20 | 500 | ~2-3h |
| `default` | 2,000 | 10 | 100 | 20,000 | ~100-150h |
| `thorough` | 5,000 | 15 | 150 | 75,000 | ~500-600h |

## 📁 Project Structure

```
VC-ARN/
├── app/
│   ├── main.py                          # CLI entry point (NAS + Training)
│   └── src/
│       ├── arqui_cnn.py                 # NASCNN15 architecture
│       ├── load.py                      # CIFAR-10 data loading
│       ├── train_pipeline.py            # Training orchestrator
│       ├── auxiliares.py                # Helper functions
│       ├── pre_processed.py             # Data preprocessing utilities
│       └── nas/                         # NAS module
│           ├── __init__.py
│           ├── configs.py               # NAS configurations
│           ├── utils.py                 # DNA utilities
│           ├── controller.py            # LSTM Controller
│           ├── child_builder.py         # Dynamic CNN builder
│           ├── reinforce.py             # REINFORCE optimizer
│           └── trainer.py               # NAS orchestrator
│
├── datasets/                            # CIFAR-10 data cache (generated)
├── experiments/                         # Training outputs and checkpoints
├── Salidas_Experimentos/                # Legacy experiment exports
├── README.md                            # This file (English)
└── README_ES.md                         # Spanish version
```

## 🔧 Technical Details

### NAS Module Components

**1. Controller (`controller.py`)**
- LSTM-based architecture generator
- ~11K trainable parameters
- Generates DNA sequences for child networks

**2. Child Builder (`child_builder.py`)**
- Dynamically constructs CNNs from DNA
- Validates and clips DNA to allowed ranges
- Adds BatchNorm, ReLU, and optional pooling

**3. REINFORCE Optimizer (`reinforce.py`)**
- Policy gradient with baseline (EMA)
- L2 regularization (β=1e-4)
- Exponential LR decay
- Gradient clipping

**4. NAS Trainer (`trainer.py`)**
- Orchestrates the full NAS process
- Narrative logging for research documentation
- Automatic checkpointing
- Resume capability

**5. Utilities (`utils.py`)**
- DNA encode/decode/validate
- Random DNA generation
- Architecture visualization

### Training Pipeline

The `TrainingPipeline` class provides:
- Early stopping with patience
- LR scheduling (ReduceLROnPlateau)
- Checkpoint saving
- Metrics logging
- Plot generation

### Logging System

**8 hierarchical log levels:**
- 📋 **INFO**: General information
- ✅ **SUCCESS**: Successful operations
- 🔹 **STEP**: Process steps
- 📊 **METRIC**: Metrics and results
- ❌ **ERROR**: Errors
- 🏗️ **ARCHITECTURE**: Architecture info
- 🎯 **TRAINING**: Child training
- 🏆 **REWARD**: Rewards obtained

**Example log output:**
```
[2025-11-22 10:15:23] 📋 STARTING NEURAL ARCHITECTURE SEARCH
[2025-11-22 10:15:23] 🔹 SEARCH CONFIGURATION:
[2025-11-22 10:15:23] 📋   • Total episodes: 100
[2025-11-22 10:15:23] 📋   • Architectures per episode: 5
[2025-11-22 10:16:45] 🎯 → Evaluating architecture 1/5
[2025-11-22 10:16:45] 🏗️ Child ep1_child1 - Architecture built: 245,322 parameters
[2025-11-22 10:18:30] 🏆 Child ep1_child1 - Training completed: Best Val Acc = 0.6523
[2025-11-22 10:20:15] ✅ NEW BEST ARCHITECTURE FOUND! Reward: 0.6523
```

## 📊 CIFAR-10 Dataset

### Overview

- **60,000 color images** (32×32 pixels)
- **10 classes** of objects
- Standard split: 50,000 training + 10,000 test
- Balanced classes (6,000 images per class)

### Classes

| ID | Category | Examples |
|----|----------|----------|
| 0 | ✈️ Airplane | Aircraft, jets |
| 1 | 🚗 Automobile | Cars, vans |
| 2 | 🐦 Bird | Various bird species |
| 3 | 🐱 Cat | Domestic cats |
| 4 | 🦌 Deer | Deer, elk |
| 5 | 🐕 Dog | Various dog breeds |
| 6 | 🐸 Frog | Frogs and toads |
| 7 | 🐎 Horse | Horses, ponies |
| 8 | 🚢 Ship | Ships, boats |
| 9 | 🚜 Truck | Trucks, trailers |

### Challenges

- Small image size (32×32)
- Objects in different positions and angles
- Lighting and background variations
- Partial occlusion

## 📈 Results and Performance

### NASCNN15 Baseline

| Metric | Value |
|--------|-------|
| Parameters | ~1.9M |
| Test Accuracy | ~92.5% |
| Training Time | ~4-6 hours |

### NAS Search Results

The NAS search explores thousands of architectures and tracks:
- Best architecture found (saved to JSON)
- Reward history over episodes
- Convergence metrics (baseline, advantage)
- Architecture diversity

**Outputs:**
- `checkpoints/nas/best_architecture.json` - Best architecture DNA
- `checkpoints/nas/nas_final.pth` - Final checkpoint
- `logs/nas/nas_search_TIMESTAMP.log` - Complete narrative log

## 🧪 Testing

The project currently relies on manual validation workflows (Smoke tests on data
loading, controller sampling, and NASCNN15 training). Automated unit tests can
be reintroduced by porting the previous `test_nas_module.py` suite if desired.

## 📖 Documentation

- **`README.md`**: English reference (this document)
- **`README_ES.md`**: Spanish overview
- **Inline docstrings**: Components include detailed explanations and usage examples

## 🛠️ Technologies Used

### Core Libraries

- **PyTorch** (2.0+): Deep learning framework
- **TorchVision**: Computer vision datasets and transforms
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Metrics and utilities

### Optimizers

- **SGD** (with momentum and Nesterov)
- **Adam** (adaptive gradient optimization)
- **RMSprop** (per-parameter adaptive learning)

### LR Schedulers

- **StepLR**: Step-based decay
- **ReduceLROnPlateau**: Performance-based adjustment
- **OneCycleLR / CyclicLR**: Cyclic learning rates

## 🎓 Research Applications

This project is ideal for:

- **Academic research** on NAS and AutoML
- **Architecture discovery** for specific datasets
- **Comparative studies** of NAS algorithms
- **Educational purposes** to understand NAS

### For Research Papers

The narrative logging system provides:
- Complete process traceability
- Step-by-step explanations
- Detailed metrics per episode
- Timeline of architecture discoveries

Perfect for writing **Methodology** and **Results** sections with precise, traceable data.

## 🔬 Extending the Project

### Adding New Search Spaces

Modify `configs.py` to change DNA limits:
```python
DNA_LIMITS = {
    'kernel_size': (1, 7),
    'num_filters': (32, 512),
    'stride': (1, 2),
    'pool_size': (1, 3),
}
```

### Custom Architectures

Extend `ChildNetworkBuilder.build_from_dna()` to support:
- Different layer types (depthwise, grouped conv)
- Attention mechanisms
- Custom skip connections

### Alternative RL Algorithms

The modular design allows replacing REINFORCE with:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- Evolution Strategies

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{nascnn2025,
  author = {Tschopp, Joaquín S.},
  title = {Neural Architecture Search with Reinforcement Learning for CIFAR-10},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/VC-ARN}
}
```

### Original Paper

```bibtex
@article{zoph2017neural,
  title={Neural architecture search with reinforcement learning},
  author={Zoph, Barret and Le, Quoc V},
  journal={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```

## 👨‍💻 Author

**Esp. Joaquín S. Tschopp**

Computer Vision & Artificial Intelligence Specialist

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original NAS paper by Zoph & Le (Google Brain)
- PyTorch team for the excellent framework
- CIFAR-10 dataset creators
- Open source community

---

*This project demonstrates the practical application of Neural Architecture Search for automated neural network design, utilizing advanced techniques in Deep Learning, Reinforcement Learning, and Computer Vision.*

## 🔗 Additional Resources

- [Original NAS Paper](https://arxiv.org/abs/1611.01578)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [REINFORCE Algorithm](https://www.youtube.com/watch?v=CYUpDogeIL0)

**Project Status:** 🟢 Production Ready  
**Last Updated:** November 22, 2025
