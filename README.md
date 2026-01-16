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
- `--num_iter`: Number of training iterations (default: 3)
- `--theta`: Weighting parameter (default: 0.3)
- `--lambda_vals`: Lambda values for combining iterations (default: [0.1])
- `--data_path`: Path to dataset (default: "./data/oxford-pets-cascam/with_artifact/")
- `--threshold_method`: CAM thresholding method - `top_k` or `ebayesthresh` (omit for no thresholding)
- `--max_comparison_images`: Maximum images for comparison (omit this option to process all images)
- `--random_seed`: Random seed for reproducibility (default: 43052)

#### Evaluation Parameters
- `--annotation_dir`: Path to annotation directory for evaluation (default: "./data/oxford-pets-cascam/annotations")
- `--artifact_masks_dir`: Path to artifact masks directory (optional, enables artifact analysis)
- `--no_iou`: Skip all evaluation (training + CAM generation only)
- `--no_advanced`: Skip advanced metrics evaluation (run basic IoU only)
- `--eval_only_basic`: Run only basic IoU evaluation (faster, skip advanced metrics)

#### Threshold Method Parameters
*These parameters depend on the selected `--threshold_method`*

**When using `--threshold_method top_k`:**
- `--top_k`: Percentage of top values to keep (default: 10)

**When using `--threshold_method ebayesthresh`:**
- `--ebayesthresh_method`: Algorithm - `sure` or `bayes` (default: sure)
- `--ebayesthresh_prior`: Prior - `laplace` or `cauchy` (default: laplace)
- `--ebayesthresh_a`: Regularization parameter (default: 0.5)

#### Training Parameters
- `--patience`: Early stopping patience (default: 1)
- `--max_epochs`: Maximum training epochs (default: 10)

### Basic Usage
```bash
python run.py
```

### Usage Examples

**1. Full Evaluation with All Metrics (Recommended)**
```bash
python run.py \
  --num_iter 3 \
  --theta 0.3 \
  --lambda_vals 0.1 \
  --data_path "./data/oxford-pets-cascam/with_artifact/" \
  --annotation_dir "./data/oxford-pets-cascam/annotations" \
  --artifact_masks_dir "./data/oxford-pets-cascam/artifact_boxes/" \
  --threshold_method top_k \
  --top_k 10 \
  --eval_use_topk \
  --eval_k_percent 0.10
```
This runs:
- Object Localization evaluation (vs GT annotations)
- Artifact Detection evaluation (vs artifact masks)
- **Cross-Analysis evaluation** (object-artifact relationship) - NEW!
- Advanced metrics with all distribution statistics

**2. Basic IoU Evaluation Only (Faster)**
```bash
python run.py \
  --num_iter 3 \
  --theta 0.3 \
  --lambda_vals 0.1 \
  --data_path "./data/oxford-pets-cascam/with_artifact/" \
  --annotation_dir "./data/oxford-pets-cascam/annotations" \
  --eval_only_basic
```

**3. Training + CAM Generation Only (No Evaluation)**
```bash
python run.py \
  --num_iter 3 \
  --theta 0.3 \
  --lambda_vals 0.1 0.2 \
  --data_path "./data/oxford-pets-cascam/with_artifact/" \
  --no_iou
```

**4. Multiple Lambda Values with Advanced Metrics**
```bash
python run.py \
  --num_iter 5 \
  --theta 0.3 \
  --lambda_vals 0.01 0.05 0.1 0.2 0.3 \
  --data_path "./data/oxford-pets-cascam/with_artifact/" \
  --annotation_dir "./data/oxford-pets-cascam/annotations" \
  --artifact_masks_dir "./data/oxford-pets-cascam/artifact_boxes/"
```

**5. EBayesThresh Method with Evaluation**
```bash
python run.py \
  --num_iter 5 \
  --theta 0.3 \
  --lambda_vals 0.1 \
  --data_path "./data/oxford-pets-cascam/with_artifact/" \
  --annotation_dir "./data/oxford-pets-cascam/annotations" \
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
├── oxford-pets-cascam/
│   ├── with_artifact/           # Training images (with artifacts)
│   │   ├── Abyssinian_1.jpg     # Cat images (uppercase = cat)
│   │   ├── beagle_1.jpg         # Dog images (lowercase = dog)
│   │   └── ...
│   ├── annotations/             # Ground truth annotations
│   │   └── trimaps/
│   │       ├── Abyssinian_1.png
│   │       └── ...
│   └── artifact_boxes/          # Artifact masks (optional)
│       ├── Abyssinian_1_artifact.png
│       └── ...

results/
└── run_20251111_120000/         # Timestamp-based run ID
    ├── config.yaml              # Experiment configuration
    ├── training/
    │   ├── checkpoints/         # Model checkpoints per iteration
    │   │   ├── iter_1/
    │   │   ├── iter_2/
    │   │   └── iter_3/
    │   ├── removed1/            # First iteration processed images
    │   ├── removed2/            # Second iteration processed images
    │   └── removed3/            # Third iteration processed images
    ├── cams/                    # Generated CAM files
    │   ├── lambda_0.1/
    │   │   ├── CasCAM_image1.npy
    │   │   ├── GradCAM_image1.npy
    │   │   └── ...
    │   └── lambda_0.2/
    │       └── ...
    ├── timing/                  # Computation time statistics
    │   ├── computation_times.csv
    │   └── timing_breakdown.json
    └── evaluation/              # Evaluation results
        ├── lambda_0.1/
        └── lambda_0.2/
```

## Output

The analysis generates:
- **Model Checkpoints**: Trained models for each iteration
- **Processed Images**: Weighted images for each iteration
- **CAM Files**: Generated activation maps saved as .npy files
- **Timing Information**: Computation time statistics and breakdowns
- **Configuration**: YAML file with all experiment parameters

### Evaluation Results Structure

```
evaluation/lambda_X/
├── object_localization/              # Object finding performance (vs GT annotation)
│   ├── detailed.csv                  # Per-image metrics (all 11 methods)
│   ├── summary.csv                   # Statistical summary
│   └── vs_baseline.csv               # Comparison vs CAM
│
├── artifact_detection/               # Artifact finding performance (vs artifact masks)
│   ├── detailed.csv                  # Per-image metrics
│   ├── summary.csv                   # Statistical summary
│   └── vs_baseline.csv               # Comparison vs CAM
│
├── cross_analysis/                   # Object-Artifact relationship analysis
│   ├── per_image.csv                 # Cross-metrics per image
│   ├── summary.csv                   # Statistical summary
│   └── vs_CAM.csv                    # Comparison vs baseline
│
├── detailed_results.csv              # Comprehensive metrics (all in one)
├── summary_report.csv                # Overall summary report
└── comparison_figures/               # Visualization PDFs
```

### Object Localization Evaluation
Evaluates how well CAM methods locate the target object (e.g., cat/dog).
- **Metrics**: IoU, Dice, Precision, Recall, F1, AP, AUC
- **Localization**: Top-15% Precision, Pointing Game, Centroid Distance
- **Boundary**: Boundary F1, Chamfer Distance, Hausdorff Distance
- **Higher values = better object localization**

### Artifact Detection Evaluation (if artifact masks provided)
Evaluates how well CAM methods detect artifacts (watermarks, logos, text).
- **Same metrics as object localization**
- **Purpose**: Understand if model relies on artifacts for predictions
- **Higher IoU = model found artifacts (shows dependency)**

### Cross-Analysis Evaluation (NEW!)
Analyzes the relationship between object and artifact activations.

**Key Metrics**:
- `artifact_fpr`: False positive rate in artifact regions (객체 찾으려다 아티팩트 오탐)
  - **Lower is better** - indicates less confusion with artifacts

- `clean_object_precision`: Object precision excluding artifact regions (아티팩트 제외 순수 객체 정확도)
  - **Higher is better** - pure object detection accuracy

- `artifact_contamination`: Ratio of artifact pixels in predictions (예측 중 아티팩트 비율)
  - **Lower is better** - less artifact contamination

- `object_purity`: Ratio of object pixels in meaningful activations (의미있는 활성화 중 객체 비율)
  - **Higher is better** - cleaner object focus

- `distraction_score`: Artifact IoU / Object IoU (객체 대비 아티팩트 현혹도)
  - **Lower is better** - less distracted by artifacts

- `dependency_ratio`: Artifact IoU / (Object IoU + Artifact IoU) (아티팩트 의존도)
  - Shows model's reliance on artifacts for decision-making
  - Interpretation: Higher = model uses artifacts more

**Use Cases**:
- Identify if model relies on spurious correlations (artifacts)
- Compare robustness across different CAM methods
- Validate that model focuses on true object features

### Advanced Metrics Evaluation (when enabled)
- `detailed_results.csv`: Per-image comprehensive metrics for all methods
- `summary_report.csv`: Mean, std, median statistics for all metrics
- All metrics from Object Localization + Distribution analysis

### Computation Time Analysis
- `computation_times.csv`: Total computation time for each method
- `timing_breakdown.json`: Detailed breakdown of CasCAM timing
  - (1) Training time per iteration
  - (2) Thresholding time
  - (3) Preprocessing time (dataloader + image processing)
  - (4) CAM generation time per method

For detailed information about evaluation metrics, see the method documentation in `evaluator.py`.

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