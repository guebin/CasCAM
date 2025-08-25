"""
CasCAM Analysis Source Module

This module contains reusable functions and classes for CasCAM (Cascaded Class Activation Mapping) analysis.
Provides core functionality for CAM generation, weight calculation, and visualization.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import json
import warnings
from datetime import datetime
from pytorch_grad_cam import (
    GradCAM, 
    HiResCAM, 
    ScoreCAM, 
    GradCAMPlusPlus, 
    AblationCAM, 
    XGradCAM, 
    FullGrad, 
    EigenGradCAM, 
    LayerCAM
)
from fastai.vision.all import *

# Suppress warnings and progress bars
warnings.filterwarnings('ignore')
os.environ['TQDM_DISABLE'] = '1'

# Suppress fastai verbose output


class TrainingLogger:
    """Class to capture and log training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.history = {
            'iterations': {}  # Will store per-iteration metrics
        }
    
    def log_iteration(self, iteration, epochs_data):
        """Log metrics for an entire iteration (all epochs)"""
        self.history['iterations'][f'iteration_{iteration+1}'] = epochs_data
    
    def log_epoch(self, epoch, train_loss, valid_loss, error_rate):
        """Log metrics for one epoch (backward compatibility)"""
        if 'epoch' not in self.history:
            self.history.update({
                'epoch': [],
                'train_loss': [],
                'valid_loss': [],
                'error_rate': [],
                'accuracy': []
            })
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['valid_loss'].append(float(valid_loss))
        self.history['error_rate'].append(float(error_rate))
        self.history['accuracy'].append(float(1.0 - error_rate))
    
    def save_to_file(self, filepath):
        """Save training history to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class CasCAMConfig:
    """Configuration class for CasCAM analysis parameters"""
    
    def __init__(self, 
                 num_iter=3,
                 theta=0.1,
                 lambda_vals=[0.369],
                 data_path="./data/pet/",
                 random_seed=43052,
                 max_comparison_images=None):
        self.theta = theta
        self.lambda_vals = lambda_vals if isinstance(lambda_vals, list) else [lambda_vals]
        self.num_iter = num_iter
        self.data_path = data_path
        self.random_seed = random_seed
        self.max_comparison_images = max_comparison_images
        
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
            'run_id': self.run_id
        }
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        config_path = f"{self.experiment_dir}/config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return config_path


class CAMGenerator:
    """Class for generating various types of Class Activation Maps"""
    
    
    @staticmethod
    def original_cam(model, input_tensor, label):
        """Generate original CAM using model weights"""
        cam = torch.einsum('ocij,kc -> okij', model[0](input_tensor), model[1][2].weight).data.cpu()
        cam = cam[0,0,:,:] if label == 0 else cam[0,1,:,:]
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0), 
            size=(512, 512), 
            mode='bilinear', 
            align_corners=False
        )
        return cam.squeeze(0)
    
    @staticmethod
    def get_img_and_originalcam(dls, idx, model, dataset='combined'):
        """Get image and original CAM for given index"""
        if dataset == 'combined':
            # Get from combined dataset (train + valid)
            all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
            img_path = all_items[idx]
            # Load image and get label from filename
            img = PILImage.create(img_path)
            label = 0 if str(img_path).split('/')[-1][0].isupper() else 1  # Cat=0, Dog=1
        else:
            img, label = dls.train_ds[idx]
        
        img_norm, = next(iter(dls.test_dl([img])))
        cam = CAMGenerator.original_cam(model=model, input_tensor=img_norm, label=label)
        return img, cam
    
    @staticmethod
    def get_img_and_allcams(dls, idx, model, methods, dataset='combined'):
        """Get image and all CAM variants (original + comparison methods)"""
        if dataset == 'combined':
            # Get from combined dataset (train + valid)
            all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
            img_path = all_items[idx]
            # Load image and get label from filename
            img = PILImage.create(img_path)
            label = 0 if str(img_path).split('/')[-1][0].isupper() else 1  # Cat=0, Dog=1
        else:
            img, label = dls.train_ds[idx]
            
        img_norm, = next(iter(dls.test_dl([img])))
        cam = CAMGenerator.original_cam(model=model, input_tensor=img_norm, label=label)
        cascam = cam
        allcams = [cam, cascam]
        
        for method in methods:
            allcams.append(torch.tensor(method(model=model, target_layers=model[0][-1])(input_tensor=img_norm,targets=None)))
        return img, allcams


class ImageProcessor:
    """Class for processing images using CAM-based weighting"""
    
    
    @staticmethod
    def calculate_adaptive_theta(cam, base_theta):
        """Calculate adaptive theta using CAM standard deviation"""
        adaptive_theta = base_theta / cam.std()**2 
        return adaptive_theta
    
    @staticmethod
    def apply_cam_weighting(img, cam, theta, use_adaptive=True):
        """Apply CAM-based weighting to image with optional adaptive theta"""
        if use_adaptive:
            adaptive_theta = ImageProcessor.calculate_adaptive_theta(cam, theta)
        else:
            adaptive_theta = theta
            
        img_tensor = torchvision.transforms.ToTensor()(img)
        weight = np.exp(-adaptive_theta * cam)
        res_img_tensor = img_tensor * weight / (img_tensor * weight).max()
        res_img = torchvision.transforms.ToPILImage()(res_img_tensor)
        return res_img
    
    @staticmethod
    def save_processed_image(img, save_path):
        """Save processed image to specified path"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)


class CasCAMVisualizer:
    """Class for creating CasCAM visualizations"""
    
    @staticmethod
    def make_figure(img, allcams, methods):
        """Create comparison figure with all CAM methods"""
        fig, axs = plt.subplots(3, 4)
        for ax in axs.flatten():
            img.show(ax=ax)
        
        axs[0][0].set_title("Original Image")
        axs[0][1].set_title("CasCAM (proposed)")
        axs[0][2].set_title("CAM")
        
        for ax, method in zip(axs.flatten()[3:], methods):
            ax.set_title(f"{method.__name__}")
        
        for ax, cam in zip(axs.flatten()[1:], allcams):
            ax.imshow(cam.squeeze(), alpha=0.7, cmap="magma")
        
        fig.set_figwidth(10)            
        fig.set_figheight(7.5)
        fig.tight_layout() 
        return fig
    
    @staticmethod
    def save_figure(fig, save_path):
        """Save figure to specified path"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)


def _get_label_from_filename(filepath):
    """Helper function to get label from filename - must be at module level for pickling"""
    filename = str(filepath).split('/')[-1]
    return "cat" if filename[0].isupper() else "dog"


class ModelTrainer:
    """Class for training models with consistent configuration"""
    
    @staticmethod
    def create_dataloader(path, random_seed=43052):
        """Create FastAI DataLoader for given path"""
        torch.manual_seed(random_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dls = ImageDataLoaders.from_name_func(
                path=path,
                fnames=get_image_files(path),
                label_func=_get_label_from_filename,
                item_tfms=Resize(512),
                batch_tfms=ToTensor(),
                num_workers=0
            )
        return dls
    
    @staticmethod
    def create_learner(dls, reset_weights=False):
        """Create and configure FastAI learner"""
        if reset_weights:
            # Create fresh model without pretrained weights
            lrnr = vision_learner(dls, resnet34, metrics=error_rate, pretrained=False)
        else:
            lrnr = vision_learner(dls, resnet34, metrics=error_rate)
            lrnr.model[1] = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(output_size=1), 
                torch.nn.Flatten(),
                torch.nn.Linear(512, out_features=2, bias=False)
            )
        return lrnr
    
    @staticmethod
    def train_with_early_stopping(lrnr, max_epochs=10, patience=1):
        """Train model with early stopping and return training logger"""
        logger = TrainingLogger()
        best_valid_loss = float('inf')
        patience_counter = 0
        
        # For first epoch, do initial training
        lrnr.fine_tune(1)
        
        if lrnr.recorder.values is None or len(lrnr.recorder.values) == 0:
            logger.log_epoch(1, float('inf'), float('inf'), 1.0)
            return logger
            
        # Get metrics from first training
        train_loss = float(lrnr.recorder.values[-1][0])
        valid_loss = float(lrnr.recorder.values[-1][1]) 
        error_rate = float(lrnr.recorder.values[-1][2])
        logger.log_epoch(1, train_loss, valid_loss, error_rate)
        best_valid_loss = valid_loss
        
        # Continue with additional epochs if needed
        for epoch in range(1, max_epochs):
            # Train one more epoch
            lrnr.fine_tune(1)
            
            if lrnr.recorder.values is None or len(lrnr.recorder.values) == 0:
                break
                
            # Get current metrics
            train_loss = float(lrnr.recorder.values[-1][0])
            valid_loss = float(lrnr.recorder.values[-1][1]) 
            error_rate = float(lrnr.recorder.values[-1][2])
            
            # Log the epoch
            logger.log_epoch(epoch + 1, train_loss, valid_loss, error_rate)
            
            # Early stopping logic
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return logger


class CasCAMAnalyzer:
    """Main analyzer class that orchestrates the entire CasCAM pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.dls_list = []
        self.lrnr_list = []
    
    def train_models(self):
        """Train models for all iterations"""
        # Initialize consolidated training logger
        consolidated_logger = TrainingLogger()
        
        for k in range(self.config.num_iter):
            path = self.config.get_data_path(k)
            dls = ModelTrainer.create_dataloader(path, self.config.random_seed)
            self.dls_list.append(dls)
            
            # Use pretrained weights for all iterations
            lrnr = ModelTrainer.create_learner(dls, reset_weights=False)
            self.lrnr_list.append(lrnr)
            
            # Train with early stopping (temporarily set to 1 epoch)
            training_logger = ModelTrainer.train_with_early_stopping(lrnr, max_epochs=1, patience=1)
            
            # Add this iteration's data to consolidated logger
            consolidated_logger.log_iteration(k, training_logger.history)
            
            # Process images for next iteration
            if k < self.config.num_iter - 1:  # Don't process for last iteration
                self._process_images_for_next_iteration(dls, lrnr, k)
        
        # Save consolidated training metrics
        metrics_path = f"{self.config.experiment_dir}/training_metrics.json"
        consolidated_logger.save_to_file(metrics_path)
    
    @staticmethod
    def _process_single_image(args):
        """Process single image for parallel execution"""
        idx, dls, model, theta, save_dir = args
        
        img, cam = CAMGenerator.get_img_and_originalcam(dls, idx, model, dataset='combined')
        res_img = ImageProcessor.apply_cam_weighting(img, cam, theta, use_adaptive=True)
        
        # Get filename from combined dataset
        all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
        fname = str(all_items[idx]).split("/")[-1]
        save_path = f"{save_dir}/{fname}"
        ImageProcessor.save_processed_image(res_img, save_path)
        
        return fname
    
    def _process_images_for_next_iteration(self, dls, lrnr, iteration):
        """Process images using CAM weighting for next iteration sequentially"""
        save_dir = self.config.get_save_dir(iteration)
        # Use combined dataset (train + valid)
        total_images = len(dls.train_ds) + len(dls.valid_ds)
        
        print(f"Processing {total_images} images for iteration {iteration+2} sequentially...")
        
        # Process images sequentially
        results = []
        for idx in range(total_images):
            print(f"  Processing iteration {iteration+2}: {idx + 1}/{total_images} images", end='\r')
            args = (idx, dls, lrnr.model, self.config.theta, save_dir)
            result = self._process_single_image(args)
            results.append(result)
        print()  # New line after progress
        
        print(f"  Completed processing {len(results)} images")
    
    @staticmethod
    def _generate_cam_data(args):
        """Generate CAM data for single image (parallel execution)"""
        idx, path, dls, lrnr_list, methods = args
        
        lrnr = lrnr_list[0]
        img, allcams = CAMGenerator.get_img_and_allcams(dls, idx, lrnr.model, methods, dataset='combined')
        
        # Collect CAMs from all iterations
        cascams = [allcams[0]]
        for lrnr in lrnr_list[1:]:
            _, cam = CAMGenerator.get_img_and_originalcam(dls, idx, lrnr.model, dataset='combined')
            cascams.append(cam)
        
        fname = str(path).split("/")[-1].split(".")[0]
        return (fname, img, allcams, cascams)
    
    @staticmethod
    def _save_lambda_result(args):
        """Save PDF for single image with specific lambda (parallel execution)"""
        fname, img, allcams, cascams, lambda_val, cascam_weights, fig_dir, methods = args
        
        # Apply weighted combination for this lambda
        combined_cam = sum(w * cam for w, cam in zip(cascam_weights, cascams))
        allcams_copy = allcams.copy()
        allcams_copy[0] = combined_cam
        
        # Create and save figure
        fig = CasCAMVisualizer.make_figure(img, allcams_copy, methods)
        fig_path = f"{fig_dir}/{fname}.pdf"
        CasCAMVisualizer.save_figure(fig, fig_path)
        
        return [fname, allcams_copy, cascams, lambda_val, cascam_weights]

    def generate_cascam_analysis(self, max_images=None):
        """Generate final CasCAM analysis with combined weights for all lambda values"""
        import random
        
        # Use config value if max_images not specified
        if max_images is None:
            max_images = self.config.max_comparison_images
        
        dls = self.dls_list[0]
        all_lambda_results = {}
        # Use combined dataset (train + valid)
        all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
        
        # Randomly sample max_images from all available images
        if max_images is not None and len(all_items) > max_images:
            random.seed(self.config.random_seed)  # Use same seed for reproducibility
            selected_indices = random.sample(range(len(all_items)), max_images)
            selected_items = [(i, all_items[i]) for i in selected_indices]
            print(f"Processing {len(selected_items)} randomly selected images (max {max_images}) sequentially...")
        else:
            selected_items = [(i, item) for i, item in enumerate(all_items)]
            print(f"Processing all {len(selected_items)} images sequentially...")
        
        total_images = len(selected_items)
        
        # Collect base CAMs from all iterations sequentially
        base_cams_data = []
        for proc_idx, (original_idx, path) in enumerate(selected_items):
            print(f"  Processing CAM data: {proc_idx + 1}/{total_images} images", end='\r')
            args = (original_idx, path, dls, self.lrnr_list, self.config.methods)
            result = self._generate_cam_data(args)
            base_cams_data.append(result)
        print()  # New line after progress
        
        print(f"  CAM generation completed for {len(base_cams_data)} images")
        
        # Generate comparison for each lambda value
        for lambda_val in self.config.lambda_vals:
            print(f"Saving lambda={lambda_val} results sequentially...")
            cascam_weights = self.config.calculate_cascam_weights(lambda_val)
            fig_dir = self.config.get_fig_dir(lambda_val)
            
            # Generate PDFs sequentially
            lambda_results = []
            for idx, (fname, img, allcams, cascams) in enumerate(base_cams_data):
                print(f"  Generating PDF λ={lambda_val}: {idx + 1}/{len(base_cams_data)} images", end='\r')
                args = (fname, img, allcams, cascams, lambda_val, cascam_weights, fig_dir, self.config.methods)
                result = self._save_lambda_result(args)
                lambda_results.append(result)
            print()  # New line after progress
            
            print(f"  Completed {len(lambda_results)} PDFs for lambda={lambda_val}")
            all_lambda_results[lambda_val] = lambda_results
        
        return all_lambda_results
    
    def run_full_analysis(self):
        """Run complete CasCAM analysis pipeline"""        
        # Save configuration
        config_path = self.config.save_config()
        
        # Train models once (removed images generation)
        self.train_models()
        
        # Generate comparisons for all lambda values
        all_results = self.generate_cascam_analysis()
        
        return all_results