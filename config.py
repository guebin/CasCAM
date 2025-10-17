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
                 annotation_dir="./data/oxford-pets-cascam/annotations"):
        self.theta = theta
        self.lambda_vals = lambda_vals if isinstance(lambda_vals, list) else [lambda_vals]
        self.num_iter = num_iter
        self.data_path = data_path
        self.random_seed = random_seed
        self.max_comparison_images = max_comparison_images
        self.threshold_method = threshold_method
        self.threshold_params = threshold_params or {}
        self.annotation_dir = annotation_dir

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
            return f'{self.experiment_dir}/artifacts/removed{iteration}'
    
    def get_save_dir(self, iteration):
        """Get save directory for processed images"""
        return f"{self.experiment_dir}/artifacts/removed{iteration+1}"
    
    def get_fig_dir(self, lambda_val):
        """Get figure save directory for specific lambda value"""
        return f"{self.experiment_dir}/artifacts/comparisons/lambda_{lambda_val}"
    
    def save_config(self):
        """Save configuration as YAML file"""
        config_dict = {
            'theta': self.theta,
            'lambda_vals': self.lambda_vals,
            'num_iter': self.num_iter,
            'original_data_path': self.data_path,
            'random_seed': self.random_seed,
            'max_comparison_images': self.max_comparison_images,
            'dataset_name': self.dataset_name,
            'run_id': self.run_id,
            'threshold_method': self.threshold_method,
            'threshold_params': self.threshold_params,
            'annotation_dir': self.annotation_dir
        }

        os.makedirs(self.experiment_dir, exist_ok=True)
        config_path = f"{self.experiment_dir}/config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

        return config_path