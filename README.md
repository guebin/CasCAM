# CasCAM: Cascaded Class Activation Mapping

A research implementation of Cascaded Class Activation Mapping (CasCAM) for computer vision model interpretability.

## Overview

CasCAM extends traditional Class Activation Mapping (CAM) techniques through an iterative cascaded process:

1. **Initial Training**: Train model on original images
2. **CAM Generation**: Generate activation maps from trained model
3. **Image Weighting**: Apply CAM-based weighting to reduce attention on high-activation regions
4. **Iterative Refinement**: Retrain on weighted images and combine results

The final CasCAM combines activation maps from multiple iterations using exponentially decaying weights.

## Key Features

- **9 CAM Methods**: Compares against GradCAM, HiResCAM, ScoreCAM, GradCAM++, AblationCAM, XGradCAM, FullGrad, EigenGradCAM, LayerCAM
- **Automated Evaluation**: IoU metrics and computation time analysis
- **Automated Pipeline**: End-to-end analysis with minimal configuration
- **Configurable Parameters**: Adjustable theta (weighting strength), lambda (decay rate), and iteration count
- **Ready-to-Use**: Pre-configured for Oxford-IIIT Pet dataset

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository-url>
cd CasCAM
uv sync
```

### Using pip

```bash
git clone <repository-url>
cd CasCAM
python -m venv cascam_env
source cascam_env/bin/activate  # Windows: cascam_env\Scripts\activate
pip install -e .

# Optional extras
pip install -e .[gpu]      # GPU support
pip install -e .[jupyter]  # Jupyter support
pip install -e .[dev]      # Development tools
```

### Using conda

```bash
git clone <repository-url>
cd CasCAM
conda env create -f environment.yml
conda activate cascam
```

## How to Run

### Command Line Parameters

#### Core Parameters
- `--num_iter`: Number of training iterations (default: 3)
- `--theta`: Weighting parameter (default: 0.3)
- `--lambda_vals`: Lambda values for combining iterations (default: [0.1])
- `--data_path`: Path to dataset (default: "./data/pet_randombox/")
- `--threshold_method`: CAM thresholding method - `top_k` or `ebayesthresh` (omit for no thresholding)
- `--max_comparison_images`: Maximum images for comparison (omit this option to process all images)
- `--random_seed`: Random seed for reproducibility (default: 43052)
- `--annotation_dir`: Path to annotation directory for IoU evaluation (default: "./data/oxford_pets/annotations")
- `--no_iou`: Skip IoU evaluation (include this flag to disable)

#### Threshold Method Parameters
*These parameters depend on the selected `--threshold_method`*

**When using `--threshold_method top_k`:**
- `--top_k`: Percentage of top values to keep (default: 10)

**When using `--threshold_method ebayesthresh`:**
- `--ebayesthresh_method`: Algorithm - `sure` or `bayes` (default: sure)
- `--ebayesthresh_prior`: Prior - `laplace` or `cauchy` (default: laplace)
- `--ebayesthresh_a`: Regularization parameter (default: 0.5)

### Basic Usage
```bash
python run.py
```

### Usage Examples

**1. No Threshold Method**
```bash
python run.py \
  --num_iter 5 \
  --theta 0.3 \
  --lambda_vals 0.1 0.2 \
  --data_path "./data/pet_randombox/" \
  --max_comparison_images 10
```

**2. Top-k Method (Default)**
```bash
python run.py \
  --num_iter 5 \
  --theta 0.3 \
  --lambda_vals 0.1 0.2 \
  --data_path "./data/pet_randombox/" \
  --max_comparison_images 10 \
  --threshold_method top_k \
  --top_k 15
```

**3. EBayesThresh Method**
```bash
python run.py \
  --num_iter 5 \
  --theta 0.3 \
  --lambda_vals 0.1 0.2 \
  --data_path "./data/pet_randombox/" \
  --max_comparison_images 10 \
  --threshold_method ebayesthresh \
  --ebayesthresh_method sure \
  --ebayesthresh_prior laplace \
  --ebayesthresh_a 0.5
```

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

```
data/
├── pet/                          # Original dataset
│   ├── Abyssinian_1.jpg         # Cat images (uppercase = cat)
│   ├── bengal_1.jpg              # Dog images (lowercase = dog)
│   └── ...

results/
└── run_20250820_143052/          # Timestamp-based run ID
    ├── config.yaml               # Experiment configuration
    ├── training_metrics.json     # Training metrics for all iterations
    └── artifacts/
        ├── removed1/             # First iteration processed images
        ├── removed2/             # Second iteration processed images
        ├── removed3/             # Third iteration processed images
        └── comparisons/          # Comparison results
            ├── lambda_0.01/
            ├── lambda_0.05/
            ├── lambda_0.1/
            ├── lambda_0.2/
            └── lambda_0.3/
```

## Output

The analysis generates:
- **Processed Images**: Weighted images for each iteration
- **Comparison Figures**: PDF files comparing CasCAM with other CAM methods
- **Training Metrics**: Consolidated JSON with training history
- **Configuration**: YAML file with all experiment parameters
- **IoU Evaluation**: Three CSV files with detailed IoU metrics
  - `iou_detailed.csv`: Per-image IoU scores for each method
  - `iou_summary.csv`: Mean, std, median, min, max IoU per method
  - `iou_comparison.csv`: Comparison vs baseline (CAM) method
- **Computation Time Analysis**:
  - `computation_times.csv`: Total computation time for each method
  - `timing_breakdown.json`: Detailed breakdown of CasCAM timing
    - (1) Training time per iteration
    - (2) Thresholding time
    - (3) Preprocessing time (dataloader + image processing)
    - (4) CAM generation time per method

## Performance Optimizations

- **Efficient Lambda Processing**: Models train once, lambda weights applied afterward
- **Early Stopping**: Training stops when validation loss plateaus (patience=1)

## Authors

- **Seoyeon Choi***¹ - Department of Statistics, Jeonbuk National University
- **Hayoung Kim***² - Core Network Control Department, KT Corporation  
- **Guebin Choi**†¹ - Department of Statistics, Jeonbuk National University

*Equal contribution, †Corresponding author

## Citation

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