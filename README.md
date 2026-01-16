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
- **Comprehensive Evaluation Metrics**:
  - Basic segmentation: IoU, Dice/F1, Precision, Recall
  - Curve-based: PR Curve (AP), ROC Curve (AUC)
  - Localization: Top-k Precision, Pointing Game, Centroid Distance
  - Boundary quality: Boundary F1, Chamfer Distance, Hausdorff Distance
  - Artifact detection: Artifact FPR, Distribution analysis (Gini, Entropy)
- **Automated Evaluation**: IoU metrics and computation time analysis
- **Automated Pipeline**: End-to-end analysis with minimal configuration
- **Configurable Parameters**: Adjustable theta (weighting strength), lambda (decay rate), and iteration count
- **Ready-to-Use**: Pre-configured for Oxford-IIIT Pet dataset

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/guebin/CasCAM.git
cd CasCAM
uv sync
```

### Using pip

```bash
git clone https://github.com/guebin/CasCAM.git
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
git clone https://github.com/guebin/CasCAM.git
cd CasCAM
conda env create -f environment.yml
conda activate cascam
```

## How to Run

### Command Line Parameters

#### Core Parameters
- `--num_iter`: Number of CasCAM iterations (default: 3)
- `--theta`: Masking strength parameter (default: 0.3)
- `--lambda_vals`: Lambda values for exponential decay weighting (default: [0.1])
- `--data_path`: Path to dataset (default: "./data/oxford-pets-cascam/with_artifact/")
- `--random_seed`: Random seed for reproducibility (default: 43052)
- `--max_comparison_images`: Limit number of images to process (default: all)

#### Thresholding Parameters
- `--threshold_method`: CAM thresholding method - `top_k` or `ebayesthresh` (omit for no thresholding)

**When using `--threshold_method top_k`:**
- `--top_k`: Percentage of top activation values to keep (default: 10)

**When using `--threshold_method ebayesthresh`:**
- `--ebayesthresh_method`: Algorithm - `sure` or `bayes` (default: sure)
- `--ebayesthresh_prior`: Prior distribution - `laplace` or `cauchy` (default: laplace)
- `--ebayesthresh_a`: Regularization parameter (default: 0.5)

#### Evaluation Parameters
- `--annotation_dir`: Path to trimap annotations for IoU evaluation (default: "./data/oxford-pets-cascam/annotations")
- `--no_iou`: Skip IoU evaluation (training + CAM generation only)

### Basic Usage
```bash
python run.py
```

### Usage Examples

**1. Standard Run with Top-k Thresholding**
```bash
python run.py \
  --num_iter 10 \
  --theta 0.3 \
  --lambda_vals 0.1 0.2 \
  --threshold_method top_k \
  --top_k 10
```

**2. Multiple Lambda Values**
```bash
python run.py \
  --num_iter 5 \
  --theta 0.3 \
  --lambda_vals 0.1 0.2 0.3
```

**3. Training + CAM Generation Only (Skip Evaluation)**
```bash
python run.py \
  --num_iter 3 \
  --theta 0.3 \
  --lambda_vals 0.1 \
  --no_iou
```

**4. EBayesThresh Thresholding**
```bash
python run.py \
  --num_iter 10 \
  --theta 0.1 \
  --lambda_vals 0.1 \
  --threshold_method ebayesthresh \
  --ebayesthresh_method sure \
  --ebayesthresh_prior laplace \
  --ebayesthresh_a 0.5
```

**5. No Thresholding (Raw CAM)**
```bash
python run.py \
  --num_iter 10 \
  --theta 0.3 \
  --lambda_vals 0.1
```

### Custom Configuration

```python
# custom_config.py
from config import CasCAMConfig

def get_config():
    return CasCAMConfig(
        num_iter=5,
        theta=0.3,
        lambda_vals=[0.1, 0.2],
        data_path="./data/my_dataset/",
        random_seed=12345,
        threshold_method='top_k',
        threshold_params={'k': 10}
    )
```

```bash
python run.py --config custom_config.py
```

## Data Structure

```
data/
└── oxford-pets-cascam/
    ├── with_artifact/           # Training images (with artifacts)
    │   ├── Abyssinian_1.jpg     # Cat images (uppercase = cat)
    │   ├── beagle_1.jpg         # Dog images (lowercase = dog)
    │   └── ...
    └── annotations/             # Ground truth annotations
        └── trimaps/
            ├── Abyssinian_1.png
            └── ...
```

## Output Structure

```
results/
└── run_YYYYMMDD_HHMMSS/         # Timestamp-based run ID
    ├── config.yaml              # Experiment configuration
    ├── training/
    │   ├── checkpoints/         # Model checkpoints per iteration
    │   │   ├── iter_1/
    │   │   ├── iter_2/
    │   │   └── ...
    │   ├── removed1/            # Masked images after iteration 1
    │   ├── removed2/            # Masked images after iteration 2
    │   └── ...
    ├── cams/                    # Generated CAM files
    │   └── lambda_0.1/
    │       ├── CasCAM_Abyssinian_108.npy
    │       ├── GradCAM_Abyssinian_108.npy
    │       └── ...
    ├── timing/                  # Computation time statistics
    │   ├── computation_times.csv
    │   ├── timing_per_image_cam.csv
    │   └── timing_per_iteration.csv
    └── evaluation/
        └── lambda_0.1/
            └── comparison_figures/  # Side-by-side CAM visualizations
```

## Output

The analysis generates:
- **Model Checkpoints**: Trained models for each iteration (`training/checkpoints/iter_K/`)
- **Masked Images**: Images with high-activation regions suppressed (`training/removedK/`)
- **CAM Files**: Activation maps as NumPy arrays (`cams/lambda_X/{Method}_{ImageName}.npy`)
- **Timing Statistics**: Per-method and per-iteration computation times
- **Comparison Figures**: Side-by-side visualizations of all CAM methods

## Performance Optimizations

- **Efficient Lambda Processing**: Models train once, lambda weights applied afterward
- **Early Stopping**: Training stops when validation loss plateaus (patience=1)

## Authors

- **Seoyeon Choi***¹ - Department of Statistics, Jeonbuk National University
- **Hayoung Kim***² - Core Network Control Department, KT Corporation  
- **Guebin Choi**†¹ - Department of Statistics, Jeonbuk National University

*Equal contribution, †Corresponding author

## Related Resources

- **Interactive Results Visualization**: [https://guebin.github.io/cascam-results/](https://guebin.github.io/cascam-results/)
- **MS-COCO Dataset (with artifacts)**: [https://github.com/guebin/coco-catdog-cascam](https://github.com/guebin/coco-catdog-cascam)
- **Oxford-IIIT Pet Dataset (with artifacts)**: [https://github.com/guebin/oxford-pets-cascam](https://github.com/guebin/oxford-pets-cascam)

## Citation

```bibtex
@article{cascam2025,
  title={Cascading Class Activation Mapping: A Counterfactual Reasoning-Based Explainable Method for Comprehensive Feature Discovery},
  author={Seoyeon Choi and Hayoung Kim and Guebin Choi},
  journal={Computer Modeling in Engineering \& Sciences},
  year={2025},
  institution={Jeonbuk National University, KT Corporation}
}
```

## License

This project is intended for research purposes. Please refer to individual library licenses for their respective terms.

## Contributing

This is a research implementation. For questions or collaboration inquiries, please open an issue in the repository.