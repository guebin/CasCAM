"""
Main analyzer module for CasCAM
"""

import torch
import numpy as np
import os
import time
from config import CasCAMConfig
from logger import TrainingLogger
from trainer import ModelTrainer
from cam_generator import CAMGenerator, CasCAM, OtherCAMGenerator
from image_processor import ImageProcessor
from visualizer import CasCAMVisualizer
from iou_evaluator import IoUEvaluator, create_comparison_table


class CasCAMAnalyzer:
    """Main analyzer class that orchestrates the entire CasCAM pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.dls_list = []
        self.lrnr_list = []
        # Storage for CAMs and image names for IoU evaluation
        self.saved_cams = {}
        self.image_names = []
        # Storage for computation times
        self.computation_times = {}
        # Detailed timing breakdown for CasCAM
        self.cascam_timing_breakdown = {
            'training_times': [],           # (1) Training time per iteration
            'threshold_times': [],          # (2) Thresholding time per iteration
            'preprocessing_times': [],      # (3) Preprocessing + dataloader creation time
            'cam_generation_times': {}      # (4) CAM generation time per method
        }
    
    def train_models(self):
        """Train models for all iterations"""
        # Initialize consolidated training logger
        consolidated_logger = TrainingLogger()

        for k in range(self.config.num_iter):
            # (3) Preprocessing time: dataloader creation (only for k>0)
            if k > 0:
                preprocessing_start = time.time()

            path = self.config.get_data_path(k)
            dls = ModelTrainer.create_dataloader(path, self.config.random_seed)
            self.dls_list.append(dls)

            if k > 0:
                preprocessing_time = time.time() - preprocessing_start
                self.cascam_timing_breakdown['preprocessing_times'].append(preprocessing_time)
                print(f"  Iteration {k+1} preprocessing time: {preprocessing_time:.4f}s")

            # Use pretrained weights for all iterations
            lrnr = ModelTrainer.create_learner(dls, reset_weights=False)
            self.lrnr_list.append(lrnr)

            # (1) Training time
            training_start = time.time()
            training_logger = ModelTrainer.train_with_early_stopping(lrnr, max_epochs=10, patience=1)
            training_time = time.time() - training_start
            self.cascam_timing_breakdown['training_times'].append(training_time)
            print(f"  Iteration {k+1} training time: {training_time:.4f}s")

            # Add this iteration's data to consolidated logger
            consolidated_logger.log_iteration(k, training_logger.history)

            # Process images for next iteration
            if k < self.config.num_iter - 1:  # Don't process for last iteration
                self._process_images_for_next_iteration(dls, lrnr, k)

        # Save consolidated training metrics
        metrics_path = f"{self.config.experiment_dir}/training_metrics.json"
        consolidated_logger.save_to_file(metrics_path)
    
    def _process_single_image(self, idx, dls, model, theta, save_dir, timing_dict=None):
        """Process single image"""
        # Step 1: Generate original CAM
        img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, model, dataset='combined')

        # Step 2: Create CasCAM with thresholding
        # (2) Threshold time
        threshold_start = time.time()
        cascam_obj = CasCAM(original_cam, self.config.threshold_method, self.config.threshold_params)
        cascam = cascam_obj.processed_cam if cascam_obj.processed_cam is not None else cascam_obj.source_cam
        threshold_time = time.time() - threshold_start

        if timing_dict is not None:
            timing_dict['threshold_times'].append(threshold_time)

        # Step 3: Apply weighting using original ImageProcessor method (excluding save time)
        preprocess_start = time.time()
        res_img = ImageProcessor.apply_cam_weighting(img, cascam, theta)
        preprocess_time = time.time() - preprocess_start

        if timing_dict is not None:
            if 'image_processing_times' not in timing_dict:
                timing_dict['image_processing_times'] = []
            timing_dict['image_processing_times'].append(preprocess_time)

        # Get filename from combined dataset
        all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
        fname = str(all_items[idx]).split("/")[-1]
        save_path = f"{save_dir}/{fname}"
        # Image saving (excluded from timing)
        ImageProcessor.save_processed_image(res_img, save_path)

        return fname
    
    def _process_images_for_next_iteration(self, dls, lrnr, iteration):
        """Process images using CAM weighting for next iteration"""
        save_dir = self.config.get_save_dir(iteration)
        # Use combined dataset (train + valid)
        total_images = len(dls.train_ds) + len(dls.valid_ds)

        print(f"Processing {total_images} images for iteration {iteration+2}...")

        # Process images sequentially with timing
        results = []
        for idx in range(total_images):
            print(f"  Processing iteration {iteration+2}: {idx + 1}/{total_images} images", end='\r')
            result = self._process_single_image(idx, dls, lrnr.model, self.config.theta, save_dir,
                                               timing_dict=self.cascam_timing_breakdown)
            results.append(result)
        print()  # New line after progress

        print(f"  Completed processing {len(results)} images")
    
    def generate_comparison_figures(self, lambda_val):
        """Generate comparison figures for specific lambda value and save CAMs for IoU evaluation"""
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

        # Get all image items and names
        all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)

        # Initialize CAM storage for this lambda value
        if lambda_val not in self.saved_cams:
            self.saved_cams[lambda_val] = {}

        # Initialize timing storage for this lambda value
        if lambda_val not in self.computation_times:
            self.computation_times[lambda_val] = {}

        for idx in range(total_images):
            # Get all CAMs for this image
            all_cams = []

            # Generate CasCAM (weighted combination) with timing
            # (4) CAM generation time for CasCAM
            start_time = time.time()
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

            cascam_time = time.time() - start_time

            # Store in detailed breakdown
            if 'CasCAM' not in self.cascam_timing_breakdown['cam_generation_times']:
                self.cascam_timing_breakdown['cam_generation_times']['CasCAM'] = []
            self.cascam_timing_breakdown['cam_generation_times']['CasCAM'].append(cascam_time)

            all_cams.append(cascam_total)

            # Generate original CAM (without thresholding for comparison) using first model
            # (4) CAM generation time for other methods
            first_lrnr = self.lrnr_list[0]
            start_time = time.time()
            img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, first_lrnr.model, dataset='combined')
            cam_time = time.time() - start_time

            if 'CAM' not in self.cascam_timing_breakdown['cam_generation_times']:
                self.cascam_timing_breakdown['cam_generation_times']['CAM'] = []
            self.cascam_timing_breakdown['cam_generation_times']['CAM'].append(cam_time)

            all_cams.append(original_cam)

            # Generate other CAM methods using OtherCAMGenerator with first model
            img, other_cams = OtherCAMGenerator.get_img_and_allcams(dls, idx, first_lrnr.model, self.config.methods[:9], dataset='combined', timing_dict=self.cascam_timing_breakdown['cam_generation_times'])
            # Add other CAM methods
            all_cams.extend(other_cams)

            # Get actual image filename
            img_name = str(all_items[idx]).split("/")[-1]

            # Save image name (only once, not per lambda)
            if idx >= len(self.image_names):
                self.image_names.append(img_name)

            # Store CAMs for IoU evaluation
            method_names = ['CasCAM', 'CAM']
            remaining_methods = self.config.methods[:len(all_cams)-2] if len(all_cams) > 2 else []
            for m in remaining_methods:
                if hasattr(m, '__name__'):
                    method_names.append(m.__name__)
                else:
                    method_names.append(str(m))

            for method_name, cam in zip(method_names, all_cams):
                if method_name not in self.saved_cams[lambda_val]:
                    self.saved_cams[lambda_val][method_name] = []
                self.saved_cams[lambda_val][method_name].append(cam)

            # Create visualization with display names
            display_names = ['CasCAM (proposed)', 'CAM'] + method_names[2:]
            fig = CasCAMVisualizer.make_figure(img, all_cams, display_names)

            # Save figure with actual image filename
            save_dir = self.config.get_fig_dir(lambda_val)
            img_name_no_ext = img_name.split(".")[0]
            save_path = f"{save_dir}/{img_name_no_ext}_comparison.pdf"
            CasCAMVisualizer.save_figure(fig, save_path)

        print(f"Generated {total_images} comparison figures for λ={lambda_val}")
        print(f"Saved CAMs for {len(method_names)} methods")
    
    def evaluate_iou(self, annotation_dir):
        """
        Evaluate IoU for all methods and create comparison tables

        Args:
            annotation_dir: Path to annotation directory containing trimaps

        Returns:
            Dictionary of IoU results for each lambda value
        """
        print("\n" + "="*60)
        print("IoU Evaluation")
        print("="*60)

        # Initialize evaluator
        evaluator = IoUEvaluator(annotation_dir)

        # Store all results
        iou_results = {}

        for lambda_val in self.config.lambda_vals:
            print(f"\nEvaluating λ={lambda_val}")

            # Get CAMs for this lambda value
            cams_dict = self.saved_cams.get(lambda_val, {})

            if not cams_dict:
                print(f"  Warning: No CAMs found for λ={lambda_val}")
                continue

            # Evaluate IoU
            results_df = evaluator.evaluate_multiple_cams(
                cams_dict,
                self.image_names,
                threshold=0.5
            )

            # Create summary table
            summary_df = evaluator.create_summary_table(results_df)

            # Create comparison table
            comparison_df = create_comparison_table(results_df, baseline_method='CAM')

            # Save results
            save_dir = self.config.get_fig_dir(lambda_val)
            os.makedirs(save_dir, exist_ok=True)

            results_df.to_csv(f"{save_dir}/iou_detailed.csv", index=False)
            summary_df.to_csv(f"{save_dir}/iou_summary.csv", index=False)
            comparison_df.to_csv(f"{save_dir}/iou_comparison.csv", index=False)

            # Print summary
            print(f"\n  Summary Statistics (λ={lambda_val}):")
            print(summary_df.to_string(index=False))

            print(f"\n  Comparison vs CAM (λ={lambda_val}):")
            print(comparison_df.to_string(index=False))

            # Store results
            iou_results[lambda_val] = {
                'detailed': results_df,
                'summary': summary_df,
                'comparison': comparison_df
            }

        return iou_results

    def save_computation_times(self):
        """Save computation time statistics to CSV files"""
        import pandas as pd

        print("\n" + "="*60)
        print("Computation Time Summary")
        print("="*60)

        # Calculate CasCAM total time breakdown
        breakdown = self.cascam_timing_breakdown

        # (1) Total training time (all iterations)
        total_training_time = sum(breakdown['training_times'])

        # (2) Total threshold time (num_iter - 1 iterations)
        total_threshold_time = sum(breakdown['threshold_times']) if breakdown['threshold_times'] else 0

        # (3) Total preprocessing time (dataloader creation + image processing)
        total_preprocessing_time = sum(breakdown['preprocessing_times']) if breakdown['preprocessing_times'] else 0
        total_image_processing_time = sum(breakdown.get('image_processing_times', []))

        # (4) CAM generation time (per method, average per image)
        cam_times = breakdown['cam_generation_times']

        print("\n" + "="*60)
        print("CasCAM Timing Breakdown")
        print("="*60)
        print(f"\n(1) Training time ({self.config.num_iter} iterations):")
        for i, t in enumerate(breakdown['training_times']):
            print(f"    Iteration {i+1}: {t:.4f}s")
        print(f"    Total: {total_training_time:.4f}s")

        print(f"\n(2) Thresholding time ({len(breakdown['threshold_times'])} images × {self.config.num_iter-1} iterations):")
        print(f"    Total: {total_threshold_time:.4f}s")
        if breakdown['threshold_times']:
            print(f"    Mean per image: {np.mean(breakdown['threshold_times']):.6f}s")

        print(f"\n(3) Preprocessing time:")
        print(f"    Dataloader creation: {total_preprocessing_time:.4f}s")
        print(f"    Image processing: {total_image_processing_time:.4f}s")
        print(f"    Total: {total_preprocessing_time + total_image_processing_time:.4f}s")

        print(f"\n(4) CAM generation time (per image):")

        # Build comparison table
        comparison_data = []

        for method_name, times in cam_times.items():
            if len(times) > 0:
                mean_cam_time = np.mean(times)

                if method_name == 'CasCAM':
                    # CasCAM total time = (1) + (2) + (3) + (4)
                    total_time = total_training_time + total_threshold_time + total_preprocessing_time + total_image_processing_time + np.sum(times)
                    print(f"    {method_name}: {mean_cam_time:.6f}s (mean per image)")
                else:
                    # Other methods total time = (1 iteration training) + (4)
                    first_iter_training_time = breakdown['training_times'][0] if breakdown['training_times'] else 0
                    total_time = first_iter_training_time + np.sum(times)
                    print(f"    {method_name}: {mean_cam_time:.6f}s (mean per image)")

                comparison_data.append({
                    'Method': method_name,
                    'Mean CAM Time (s)': mean_cam_time,
                    'Total CAM Time (s)': np.sum(times),
                    'Total Time (s)': total_time,
                    'Count': len(times)
                })

        # CasCAM total
        cascam_total = total_training_time + total_threshold_time + total_preprocessing_time + total_image_processing_time
        if 'CasCAM' in cam_times:
            cascam_total += sum(cam_times['CasCAM'])

        print(f"\n" + "="*60)
        print(f"CasCAM Total Time: {cascam_total:.4f}s")
        print(f"  = Training({total_training_time:.4f}s)")
        print(f"  + Thresholding({total_threshold_time:.4f}s)")
        print(f"  + Preprocessing({total_preprocessing_time + total_image_processing_time:.4f}s)")
        print(f"  + CAM Generation({sum(cam_times.get('CasCAM', [])):.4f}s)")
        print("="*60)

        # Save detailed breakdown to CSV
        for lambda_val in self.config.lambda_vals:
            save_dir = self.config.get_fig_dir(lambda_val)
            os.makedirs(save_dir, exist_ok=True)

            # Save comparison table
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('Total Time (s)')
                save_path = f"{save_dir}/computation_times.csv"
                comparison_df.to_csv(save_path, index=False)

                print(f"\nTotal Computation Times:")
                print(comparison_df.to_string(index=False))
                print(f"\nSaved to: {save_path}")

            # Save detailed breakdown
            breakdown_data = {
                'training_times': breakdown['training_times'],
                'total_training_time': total_training_time,
                'threshold_times_sample': breakdown['threshold_times'][:10] if breakdown['threshold_times'] else [],
                'total_threshold_time': total_threshold_time,
                'preprocessing_times': breakdown['preprocessing_times'],
                'total_preprocessing_time': total_preprocessing_time + total_image_processing_time,
                'cascam_total_time': cascam_total
            }

            breakdown_path = f"{save_dir}/timing_breakdown.json"
            import json
            with open(breakdown_path, 'w') as f:
                json.dump(breakdown_data, f, indent=2)
            print(f"Detailed breakdown saved to: {breakdown_path}")

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