# CasCAM: Cascaded Class Activation Mapping

A counterfactual reasoning-based explainable AI method for comprehensive feature discovery through iterative CAM-based image retraining.

## Installation

Requires Python >= 3.8 and CUDA GPU (recommended).

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/guebin/CasCAM.git
cd CasCAM
uv sync
source .venv/bin/activate
```

## Data

Download datasets:
- [oxford-pets-cascam](https://github.com/guebin/oxford-pets-cascam)
- [coco-catdog-cascam](https://github.com/guebin/coco-catdog-cascam)

Place in `data/` directory:
```
data/
└── oxford-pets-cascam/
    └── with_artifact/
        ├── Abyssinian_1.jpg    # Uppercase = cat
        ├── beagle_1.jpg        # Lowercase = dog
        └── ...
```

## Usage

```bash
# Basic run
python run.py

# With parameters
python run.py --num_iter 5 --theta 0.3 --lambda_vals 0.1 0.2

# Quick test
python run.py --num_iter 1 --max_comparison_images 2 --max_epochs 1
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num_iter` | Number of iterations | 3 |
| `--theta` | Suppression strength | 0.3 |
| `--lambda_vals` | Decay rate (multiple allowed) | 0.1 |
| `--data_path` | Data path | `./data/oxford-pets-cascam/with_artifact/` |
| `--max_comparison_images` | Max images to process | all |
| `--max_epochs` | Max training epochs | 10 |
| `--patience` | Early stopping patience | 5 |
| `--random_seed` | Random seed | 43052 |

## Output

```
results/run_YYYYMMDD_HHMMSS/
├── training/
│   ├── checkpoints/iter_K/model.pth    # Model weights per iteration
│   └── removedK/                       # Suppressed images
├── cams/lambda_0.1/
│   ├── CasCAM_Abyssinian_1.npy         # 224x224 float32, values 0-1
│   ├── GradCAM_Abyssinian_1.npy
│   └── ...
├── comparison_figures/lambda_0.1/
│   └── Abyssinian_1_comparison.pdf     # 11-panel CAM comparison
└── timing/
    └── computation_times.csv
```

**Compared methods:** CAM, GradCAM, GradCAM++, XGradCAM, HiResCAM, ScoreCAM, AblationCAM, EigenGradCAM, LayerCAM, FullGrad

## Citation

```bibtex
@article{cascam2025,
  title={CasCAM: Cascading Class Activation Mapping for Comprehensive Feature Discovery},
  author={Choi, Seoyeon and Kim, Hayoung and Choi, Guebin},
  year={2025}
}
```

## Authors

- **Seoyeon Choi** - Jeonbuk National University
- **Hayoung Kim** - KT Corporation
- **Guebin Choi** (Corresponding) - Jeonbuk National University

Results: [guebin.github.io/cascam-results](https://guebin.github.io/cascam-results/)
