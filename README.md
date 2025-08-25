# CasCAM: Cascaded Class Activation Mapping

A research implementation of Cascaded Class Activation Mapping (CasCAM) for computer vision model interpretability. This project provides an iterative approach to generating and refining class activation maps through cascaded model training.

## Overview

CasCAM extends traditional Class Activation Mapping (CAM) techniques by using a cascaded iterative process:

1. **Initial Training**: Train a model on original images
2. **CAM Generation**: Generate activation maps from the trained model
3. **Image Weighting**: Apply CAM-based weighting to reduce attention on high-activation regions
4. **Iterative Refinement**: Retrain on weighted images and combine results

The final CasCAM result combines activation maps from multiple iterations using exponentially decaying weights.

## Features

- **Multiple CAM Methods**: Supports 9 different CAM variants for comparison (GradCAM, HiResCAM, ScoreCAM, etc.)
- **Configurable Parameters**: Adjustable theta (weighting strength), lambda (decay rate), and iteration count
- **Automated Pipeline**: Complete end-to-end analysis with minimal configuration
- **Visualization**: Automatic generation of comparison figures
- **Pet Dataset Ready**: Pre-configured for cat/dog classification tasks

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone <repository-url>
cd CasCAM
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd CasCAM

# Create virtual environment (recommended)
python -m venv cascam_env
source cascam_env/bin/activate  # On Windows: cascam_env\Scripts\activate

# Install dependencies
pip install -e .

# For GPU support (optional)
pip install -e .[gpu]

# For Jupyter notebook support (optional)
pip install -e .[jupyter]

# For development (optional)
pip install -e .[dev]
```

### Using conda

```bash
# Clone the repository
git clone <repository-url>
cd CasCAM

# Create and activate environment
conda env create -f environment.yml
conda activate cascam
```

## Usage

### Basic Usage

```bash
# Run with default configuration (3CasCAM)
python run.py

# Run 5CasCAM
python run.py --num_iter 5

# All options with default values explicitly shown
python run.py --num_iter 3 --theta 0.1 --lambda_vals 0.3 --data_path "./data/pet/"
```

### Common Examples

```bash
# 3CasCAM with custom base theta
python run.py --num_iter 3 --theta 0.15

# 5CasCAM with different parameters
python run.py --num_iter 5 --theta 0.2 --lambda_vals 0.5

# Use different dataset
python run.py --num_iter 3 --data_path "./data/my_images/"

# Multiple lambda values for comparison (efficient - models train only once)
python run.py --num_iter 3 --lambda_vals 0.01 0.05 0.1 0.2 0.3 0.5 1.0

# Compare different theta and lambda combinations
python run.py --num_iter 3 --theta 0.12 --lambda_vals 0.1 0.3 0.5
```

### Adaptive Theta Usage Examples

```bash
# Default adaptive theta (recommended)
python run.py --num_iter 3 --theta 0.1

# Custom base theta with adaptive adjustment
python run.py --num_iter 3 --theta 0.2

# Stronger base theta for more aggressive weighting
python run.py --num_iter 3 --theta 0.3

# Multiple base theta values for comparison
python run.py --num_iter 3 --theta 0.05 --lambda_vals 0.1 0.3 0.5
python run.py --num_iter 3 --theta 0.1 --lambda_vals 0.1 0.3 0.5
python run.py --num_iter 3 --theta 0.2 --lambda_vals 0.1 0.3 0.5
```

**Note:** The adaptive theta system automatically adjusts the actual theta used for each image based on its CAM clarity. The `--theta` parameter sets the base value for this adaptive calculation.

### Custom Configuration

Create a custom configuration file:

```python
# custom_config.py
from core import CasCAMConfig

def get_config():
    return CasCAMConfig(
        num_iter=5,
        theta=2.0,
        lambda_vals=[0.3],
        data_path="./data/my_dataset/",
        random_seed=12345
    )
```

```bash
python run.py --config custom_config.py
```

## Data Structure

The expected data structure for your dataset:

```
data/
├── pet/                          # Original dataset
│   ├── Abyssinian_1.jpg         # Cat images (uppercase = cat)
│   ├── bengal_1.jpg              # Dog images (lowercase = dog)
│   └── ...
results/
└── run_20250820_143052/          # Timestamp-based run ID
    ├── config.yaml               # Experiment configuration
    ├── training_metrics.json         # Consolidated training metrics for all iterations
    └── artifacts/
        ├── removed1/             # First iteration processed images
        ├── removed2/             # Second iteration processed images
        ├── removed3/             # Third iteration processed images
        └── comparisons/          # Comparison results for different lambda values
            ├── lambda_0.01/      # Comparison with λ=0.01
            ├── lambda_0.05/      # Comparison with λ=0.05
            ├── lambda_0.1/       # Comparison with λ=0.1
            ├── lambda_0.2/       # Comparison with λ=0.2
            └── lambda_0.3/       # Comparison with λ=0.3
```

## Configuration Parameters

### Core Parameters
- **num_iter**: Number of cascaded iterations (default: 3)
- **theta (θ)**: Base theta value for CAM-based weighting strength (default: 0.1)
- **lambda_vals (λ)**: Decay rate for combining iteration weights (default: [0.3])
- **data_path**: Path to original dataset (default: "./data/pet/")
- **random_seed**: Random seed for reproducibility (default: 43052)

### Adaptive Theta Parameters
- **use_adaptive**: Enable adaptive theta adjustment based on CAM clarity (default: True)
- **clarity_factor**: Controls sensitivity of adaptive theta to CAM clarity (default: 2.0)

The adaptive theta system automatically adjusts the weighting strength for each image based on its CAM characteristics:
- **Higher clarity CAMs** → Lower theta (gentler weighting, preserve clear features)
- **Lower clarity CAMs** → Higher theta (aggressive weighting, remove unclear features)

**Adaptive Theta Formula:**
```
adaptive_theta = base_theta * (clarity_factor - clarity_normalized * (clarity_factor - 0.1))
```

Where `clarity` combines variance, contrast, and sharpness of the CAM.

## Algorithm Details

### Weight Calculation

CasCAM weights are calculated using exponential decay:

```
w_k = exp(-λ * k) / Σ exp(-λ * i)
```

Where k is the iteration index (0-indexed).

### Image Processing

For each iteration, images are weighted by their activation maps using adaptive theta:

```
adaptive_theta = base_theta * (clarity_factor - clarity_normalized * (clarity_factor - 0.1))
I_weighted = I * exp(-adaptive_theta * CAM)
```

Where `clarity` is calculated from CAM variance, contrast, and sharpness. The weighted image is then normalized to maintain proper intensity ranges.

## Output

The analysis generates:

- **Processed Images**: Weighted images for each iteration in `results/run_<timestamp>/artifacts/`
- **Comparison Figures**: PDF files comparing CasCAM with other CAM methods in `results/run_<timestamp>/artifacts/comparisons/lambda_<value>/`
- **Training Metrics**: Consolidated JSON file with training history for all iterations in `results/run_<timestamp>/training_metrics.json`
- **Configuration**: YAML file with all parameters used for the analysis
- **Analysis Data**: Internal data structures for further processing

## Performance Optimizations

- **Efficient Lambda Processing**: When multiple lambda values are specified, models train only once and lambda weights are applied afterward
- **Early Stopping**: Training stops when validation loss doesn't improve (patience=1 by default)
- **Weight Reset**: Iterations after the first start with fresh weights for independent training
- **Silent Mode**: Warnings and progress bars are suppressed for cleaner output

## Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **FastAI**: High-level deep learning library
- **pytorch-grad-cam**: CAM implementation library
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- ~8GB RAM minimum
- ~5GB disk space for pet dataset analysis

## Research Context

This implementation is part of ongoing research in explainable AI and computer vision interpretability. The cascaded approach aims to provide more focused and refined activation maps compared to traditional single-pass CAM methods.

## Authors

- **Seoyeon Choi***¹ - Department of Statistics, Jeonbuk National University
- **Hayoung Kim***² - KT Corporation  
- **Guebin Choi**†¹ - Department of Statistics, Jeonbuk National University

*Equal contribution, †Corresponding author

¹Jeonbuk National University, ²KT Corporation

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cascam2025,
  title={CasCAM: Cascaded Class Activation Mapping for Enhanced Model Interpretability},
  author={Seoyeon Choi and Hayoung Kim and Guebin Choi},
  year={2025},
  note={Research implementation},
  institution={Jeonbuk National University, KT Corporation}
}
```

## License

This project is intended for research purposes. Please refer to individual library licenses for their respective terms.

## Contributing

This is a research implementation. For questions or collaboration inquiries, please open an issue in the repository.