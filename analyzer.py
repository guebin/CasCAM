"""
Main analyzer module for CasCAM
"""

import torch
from config import CasCAMConfig
from logger import TrainingLogger
from trainer import ModelTrainer
from cam_generator import CAMGenerator, CasCAM, OtherCAMGenerator
from image_processor import ImageProcessor
from visualizer import CasCAMVisualizer


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
            
            # Train with early stopping
            training_logger = ModelTrainer.train_with_early_stopping(lrnr, max_epochs=10, patience=1)
            
            # Add this iteration's data to consolidated logger
            consolidated_logger.log_iteration(k, training_logger.history)
            
            # Process images for next iteration
            if k < self.config.num_iter - 1:  # Don't process for last iteration
                self._process_images_for_next_iteration(dls, lrnr, k)
        
        # Save consolidated training metrics
        metrics_path = f"{self.config.experiment_dir}/training_metrics.json"
        consolidated_logger.save_to_file(metrics_path)
    
    def _process_single_image(self, idx, dls, model, theta, save_dir):
        """Process single image"""
        # Step 1: Generate original CAM
        img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, model, dataset='combined')
        
        # Step 2: Create CasCAM with thresholding
        cascam_obj = CasCAM(original_cam, self.config.threshold_method, self.config.threshold_params)
        cascam = cascam_obj.processed_cam if cascam_obj.processed_cam is not None else cascam_obj.source_cam
        
        # Step 3: Apply weighting using original ImageProcessor method
        res_img = ImageProcessor.apply_cam_weighting(img, cascam, theta)
        
        # Get filename from combined dataset
        all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
        fname = str(all_items[idx]).split("/")[-1]
        save_path = f"{save_dir}/{fname}"
        ImageProcessor.save_processed_image(res_img, save_path)
        
        return fname
    
    def _process_images_for_next_iteration(self, dls, lrnr, iteration):
        """Process images using CAM weighting for next iteration"""
        save_dir = self.config.get_save_dir(iteration)
        # Use combined dataset (train + valid)
        total_images = len(dls.train_ds) + len(dls.valid_ds)
        
        print(f"Processing {total_images} images for iteration {iteration+2}...")
        
        # Process images sequentially
        results = []
        for idx in range(total_images):
            print(f"  Processing iteration {iteration+2}: {idx + 1}/{total_images} images", end='\r')
            result = self._process_single_image(idx, dls, lrnr.model, self.config.theta, save_dir)
            results.append(result)
        print()  # New line after progress
        
        print(f"  Completed processing {len(results)} images")
    
    def generate_comparison_figures(self, lambda_val):
        """Generate comparison figures for specific lambda value"""
        if not self.lrnr_list:
            return
        
        # Use the first dataloader for generating comparisons
        dls = self.dls_list[0]
        
        # Calculate CasCAM weights
        weights = self.config.calculate_cascam_weights(lambda_val)
        
        # Generate figures for images (all images if max_comparison_images is None)
        total_dataset_size = len(dls.train_ds) + len(dls.valid_ds)
        if self.config.max_comparison_images is None:
            total_images = total_dataset_size
            print(f"Processing all {total_images} images for comparison")
        else:
            total_images = min(self.config.max_comparison_images, total_dataset_size)
            print(f"Processing {total_images} out of {total_dataset_size} images for comparison")
        
        for idx in range(total_images):
            # Get all CAMs for this image
            all_cams = []
            
            # Generate CasCAM (weighted combination)
            cascam_total = None
            for i, (lrnr, weight) in enumerate(zip(self.lrnr_list, weights)):
                # Step 1: Generate original CAM
                img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, lrnr.model, dataset='combined')
                # Step 2: Create CasCAM with thresholding  
                cascam_obj = CasCAM(original_cam, self.config.threshold_method, self.config.threshold_params)
                cascam = cascam_obj.processed_cam if cascam_obj.processed_cam is not None else cascam_obj.source_cam
                
                if cascam_total is None:
                    cascam_total = weight * cascam
                else:
                    cascam_total += weight * cascam
            
            all_cams.append(cascam_total)
            
            # Generate original CAM (without thresholding for comparison) using first model
            first_lrnr = self.lrnr_list[0]
            img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, first_lrnr.model, dataset='combined')
            all_cams.append(original_cam)

            # Generate other CAM methods using OtherCAMGenerator with first model
            img, other_cams = OtherCAMGenerator.get_img_and_allcams(dls, idx, first_lrnr.model, self.config.methods[:9], dataset='combined')
            # Add other CAM methods
            all_cams.extend(other_cams)
            
            # Create visualization
            method_names = ['CasCAM (proposed)', 'CAM']
            # Add method names for the remaining CAMs
            remaining_methods = self.config.methods[:len(all_cams)-2] if len(all_cams) > 2 else []
            for m in remaining_methods:
                if hasattr(m, '__name__'):
                    method_names.append(m.__name__)
                else:
                    method_names.append(str(m))
            fig = CasCAMVisualizer.make_figure(img, all_cams, method_names)
            
            # Save figure with actual image filename
            save_dir = self.config.get_fig_dir(lambda_val)
            # Get actual image filename
            all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
            img_name = str(all_items[idx]).split("/")[-1].split(".")[0]  # Remove extension
            save_path = f"{save_dir}/{img_name}_comparison.pdf"
            CasCAMVisualizer.save_figure(fig, save_path)
        
        print(f"Generated {total_images} comparison figures for λ={lambda_val}")
    
    def run_full_analysis(self):
        """Run complete CasCAM analysis"""
        print(f"Starting CasCAM analysis: {self.config.dataset_name} | θ={self.config.theta} | λ={self.config.lambda_vals} | {self.config.num_iter} iter")
        
        # Train models for all iterations
        self.train_models()
        
        # Generate comparisons for all lambda values
        all_results = {}
        for lambda_val in self.config.lambda_vals:
            print(f"\\nGenerating comparisons for λ={lambda_val}")
            self.generate_comparison_figures(lambda_val)
            
            weights = self.config.calculate_cascam_weights(lambda_val)
            all_results[lambda_val] = {
                'weights': weights,
                'num_iterations': len(self.lrnr_list)
            }
        
        # Save configuration
        self.config.save_config()
        
        return all_results