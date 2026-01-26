"""
Configuration module for CasCAM
"""

import os
import yaml
import numpy as np
from datetime import datetime
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, 
    AblationCAM, XGradCAM, FullGrad, EigenGradCAM, LayerCAM
)


class CasCAMConfig:
    """Configuration class for CasCAM analysis parameters"""
    
    def __init__(self,
                 num_iter=3,
                 theta=0.3,
                 lambda_vals=[0.1],
                 data_path="./data/oxford-pets-cascam/with_artifact/",
                 random_seed=43052,
                 max_comparison_images=None,
                 threshold_method=None,
                 threshold_params=None,
                 max_epochs=10,
                 patience=5):
        self.theta = theta
        self.lambda_vals = lambda_vals if isinstance(lambda_vals, list) else [lambda_vals]
        self.num_iter = num_iter
        self.data_path = data_path
        self.random_seed = random_seed
        self.max_comparison_images = max_comparison_images
        self.threshold_method = threshold_method
        self.threshold_params = threshold_params or {}
        self.max_epochs = max_epochs
        self.patience = patience

        # Derived parameters
        self.dataset_name = self.data_path.strip("./").strip("/").split("/")[-1]
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"./results/run_{self.run_id}"

        # CAM methods for comparison
        self.methods = (
            GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
            AblationCAM, XGradCAM, FullGrad, EigenGradCAM, LayerCAM
        )
    
    def calculate_cascam_weights(self, lambda_val):
        """Calculate weights using exp(-lambda*k) formula where k is 0-indexed"""
        weights = [np.exp(-lambda_val * k) for k in range(self.num_iter)]
        total = sum(weights)
        return [w / total for w in weights]
    
    def get_data_path(self, iteration):
        """Get data path for specific iteration"""
        if iteration == 0:
            return self.data_path
        else:
            return f'{self.experiment_dir}/training/removed{iteration}'

    def get_save_dir(self, iteration):
        """Get save directory for processed images"""
        return f"{self.experiment_dir}/training/removed{iteration+1}"
    
    def get_fig_dir(self, lambda_val):
        """Get figure save directory for specific lambda value"""
        return f"{self.experiment_dir}/comparison_figures/lambda_{lambda_val}"

    def get_cams_dir(self, lambda_val):
        """Get CAMs directory for specific lambda value"""
        return f"{self.experiment_dir}/cams/lambda_{lambda_val}"

    def get_timing_dir(self):
        """Get timing directory"""
        return f"{self.experiment_dir}/timing"

    def get_training_dir(self):
        """Get training metrics directory"""
        return f"{self.experiment_dir}/training"

    def get_checkpoint_dir(self, iteration):
        """Get checkpoint directory for specific iteration"""
        return f"{self.experiment_dir}/training/checkpoints/iter_{iteration+1}"

    def save_config(self):
        """Save configuration as YAML file"""
        config_dict = {
            'data_path': self.data_path,
            'dataset_name': self.dataset_name,
            'lambda_vals': self.lambda_vals,
            'max_comparison_images': self.max_comparison_images,
            'max_epochs': self.max_epochs,
            'num_iter': self.num_iter,
            'patience': self.patience,
            'random_seed': self.random_seed,
            'run_id': self.run_id,
            'theta': self.theta,
            'threshold_method': self.threshold_method,
            'threshold_params': self.threshold_params,
        }

        os.makedirs(self.experiment_dir, exist_ok=True)
        config_path = f"{self.experiment_dir}/config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

        return config_path