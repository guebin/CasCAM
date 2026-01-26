"""
Main analyzer module for CasCAM
"""

import torch
import numpy as np
import pandas as pd
import os
import time
from config import CasCAMConfig
from logger import TrainingLogger
from trainer import ModelTrainer
from cam_generator import CAMGenerator, CasCAM, OtherCAMGenerator
from image_processor import ImageProcessor
from visualizer import CasCAMVisualizer
from evaluator import IoUEvaluator, AdvancedCAMEvaluator, create_comparison_table, compare_with_baseline


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
            'training_times': [],           # (1) Pure training time per iteration
            'threshold_times': [],          # (2) Threshold processing time between iterations
            'image_save_times': [],         # (3) Time to save thresholded images
            'dataloader_construction_times': [],  # (4) Dataloader construction time
            'preprocessing_times': [],      # Legacy: kept for compatibility
            'cam_generation_times': {},     # (5) CAM generation time per method
            'per_image_cam_times': []       # Per-image timing data
        }
    
    def train_models(self):
        """Train models for all iterations"""
        # Initialize consolidated training logger
        consolidated_logger = TrainingLogger()

        for k in range(self.config.num_iter):
            # (4) Dataloader construction time (only for k>0)
            if k > 0:
                dataloader_start = time.time()

            path = self.config.get_data_path(k)
            dls = ModelTrainer.create_dataloader(path, self.config.random_seed)
            self.dls_list.append(dls)

            if k > 0:
                dataloader_time = time.time() - dataloader_start
                self.cascam_timing_breakdown['dataloader_construction_times'].append(dataloader_time)
                print(f"  [4] Dataloader construction time (iter {k} → {k+1}): {dataloader_time:.4f}s")

            # Use pretrained weights for all iterations
            lrnr = ModelTrainer.create_learner(dls, reset_weights=False)
            self.lrnr_list.append(lrnr)

            # (1) Pure training time
            training_start = time.time()
            training_logger = ModelTrainer.train_with_early_stopping(
                lrnr, max_epochs=self.config.max_epochs, patience=self.config.patience
            )
            training_time = time.time() - training_start
            self.cascam_timing_breakdown['training_times'].append(training_time)
            print(f"  [1] Pure training time (iter {k+1}): {training_time:.4f}s")

            # Add this iteration's data to consolidated logger
            consolidated_logger.log_iteration(k, training_logger.history)

            # Save model and dataloader for this iteration
            self._save_iteration_checkpoint(k, lrnr, dls)

            # Process images for next iteration
            if k < self.config.num_iter - 1:  # Don't process for last iteration
                self._process_images_for_next_iteration(dls, lrnr, k)

        # Save consolidated training metrics
        training_dir = self.config.get_training_dir()
        os.makedirs(training_dir, exist_ok=True)
        metrics_path = f"{training_dir}/metrics.json"
        consolidated_logger.save_to_file(metrics_path)
    
    def _save_iteration_checkpoint(self, iteration, lrnr, dls):
        """Save trained model and dataloader for a specific iteration"""
        checkpoint_dir = self.config.get_checkpoint_dir(iteration)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model using torch.save (simpler and more reliable than lrnr.export)
        import torch
        model_path = f"{checkpoint_dir}/model.pth"
        torch.save(lrnr.model.state_dict(), model_path)

        # Save dataloader configuration
        dls_config = {
            'data_path': self.config.get_data_path(iteration),
            'batch_size': dls.bs,
            'num_train': len(dls.train_ds),
            'num_valid': len(dls.valid_ds),
            'train_items': [str(item) for item in dls.train_ds.items],
            'valid_items': [str(item) for item in dls.valid_ds.items]
        }

        import json
        config_path = f"{checkpoint_dir}/dls_config.json"
        with open(config_path, 'w') as f:
            json.dump(dls_config, f, indent=2)

        print(f"  Saved checkpoint for iteration {iteration+1}:")
        print(f"    Model: {model_path}")
        print(f"    DataLoader config: {config_path}")

    def load_iteration_checkpoint(self, iteration):
        """Load trained model and recreate dataloader for a specific iteration"""
        checkpoint_dir = self.config.get_checkpoint_dir(iteration)

        # Load FastAI learner
        model_path = f"{checkpoint_dir}/model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        from fastai.learner import load_learner
        lrnr = load_learner(model_path)

        # Load dataloader configuration
        config_path = f"{checkpoint_dir}/dls_config.json"
        import json
        with open(config_path, 'r') as f:
            dls_config = json.load(f)

        # Recreate dataloader
        dls = ModelTrainer.create_dataloader(dls_config['data_path'], self.config.random_seed)

        print(f"Loaded checkpoint for iteration {iteration+1}:")
        print(f"  Model: {model_path}")
        print(f"  DataLoader: {dls_config['num_train']} train + {dls_config['num_valid']} valid images")

        return lrnr, dls

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

        # (2) Threshold processing time
        threshold_start = time.time()
        # (3) Image save time
        save_start_total = 0

        # Process images sequentially with detailed timing
        results = []
        for idx in range(total_images):
            print(f"  Processing iteration {iteration+2}: {idx + 1}/{total_images} images", end='\r')

            # Get image and CAM
            img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, lrnr.model, dataset='combined')

            # Threshold CAM (part of threshold time)
            cascam_obj = CasCAM(original_cam, self.config.threshold_method, self.config.threshold_params)
            cascam = cascam_obj.processed_cam if cascam_obj.processed_cam is not None else cascam_obj.source_cam

            # Apply weighting
            res_img = ImageProcessor.apply_cam_weighting(img, cascam, self.config.theta)

            # (3) Measure image save time separately
            save_start = time.time()
            all_items = list(dls.train_ds.items) + list(dls.valid_ds.items)
            fname = str(all_items[idx]).split("/")[-1]
            save_path = f"{save_dir}/{fname}"
            ImageProcessor.save_processed_image(res_img, save_path)
            save_start_total += (time.time() - save_start)

            results.append(fname)

        threshold_time = time.time() - threshold_start - save_start_total

        # Record timing
        self.cascam_timing_breakdown['threshold_times'].append(threshold_time)
        self.cascam_timing_breakdown['image_save_times'].append(save_start_total)

        print()  # New line after progress
        print(f"  Completed processing {len(results)} images")
        print(f"  [2] Threshold processing time (iter {iteration+1} → {iteration+2}): {threshold_time:.4f}s")
        print(f"  [3] Image save time (iter {iteration+1} → {iteration+2}): {save_start_total:.4f}s")
    
    def generate_comparison_figures(self, lambda_val):
        """Generate comparison figures for specific lambda value and save CAMs for IoU evaluation"""
        if not self.lrnr_list:
            return

        # Use the first dataloader for generating comparisons
        dls = self.dls_list[0]

        # Calculate CasCAM weights
        weights = self.config.calculate_cascam_weights(lambda_val)

        # Process validation images
        total_dataset_size = len(dls.valid_ds)  # Validation only

        # Limit images to process if max_comparison_images is set
        if self.config.max_comparison_images is None:
            total_images = total_dataset_size
            print(f"Processing all {total_images} validation images")
        else:
            total_images = min(self.config.max_comparison_images, total_dataset_size)
            print(f"Processing {total_images}/{total_dataset_size} validation images")

        max_figures = total_images

        # Get all validation image items and names
        all_items = list(dls.valid_ds.items)  # Validation only

        # Initialize CAM storage for this lambda value
        if lambda_val not in self.saved_cams:
            self.saved_cams[lambda_val] = {}

        # Initialize timing storage for this lambda value
        if lambda_val not in self.computation_times:
            self.computation_times[lambda_val] = {}

        for idx in range(total_images):
            # Per-image timing dictionary (5)
            image_timing = {}

            # Get all CAMs for this image
            all_cams = []

            # Generate CasCAM (weighted combination) with timing
            start_time = time.time()
            cascam_total = None
            for i, (lrnr, weight) in enumerate(zip(self.lrnr_list, weights)):
                # Step 1: Generate original CAM
                img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, lrnr.model, dataset='valid')
                # Step 2: Create CasCAM with thresholding
                cascam_obj = CasCAM(original_cam, self.config.threshold_method, self.config.threshold_params)
                cascam = cascam_obj.processed_cam if cascam_obj.processed_cam is not None else cascam_obj.source_cam

                if cascam_total is None:
                    cascam_total = weight * cascam
                else:
                    cascam_total += weight * cascam

            cascam_time = time.time() - start_time
            image_timing['CasCAM'] = cascam_time

            # Store in detailed breakdown
            if 'CasCAM' not in self.cascam_timing_breakdown['cam_generation_times']:
                self.cascam_timing_breakdown['cam_generation_times']['CasCAM'] = []
            self.cascam_timing_breakdown['cam_generation_times']['CasCAM'].append(cascam_time)

            all_cams.append(cascam_total)

            # Generate original CAM (without thresholding for comparison) using first model
            first_lrnr = self.lrnr_list[0]
            start_time = time.time()
            img, original_cam = CAMGenerator.get_img_and_originalcam(dls, idx, first_lrnr.model, dataset='valid')
            cam_time = time.time() - start_time
            image_timing['CAM'] = cam_time

            if 'CAM' not in self.cascam_timing_breakdown['cam_generation_times']:
                self.cascam_timing_breakdown['cam_generation_times']['CAM'] = []
            self.cascam_timing_breakdown['cam_generation_times']['CAM'].append(cam_time)

            all_cams.append(original_cam)

            # Generate other CAM methods using OtherCAMGenerator with first model
            # Pass image_timing dict to collect per-method times
            img, other_cams = OtherCAMGenerator.get_img_and_allcams(dls, idx, first_lrnr.model, self.config.methods[:9], dataset='valid', timing_dict=self.cascam_timing_breakdown['cam_generation_times'], per_image_timing=image_timing)
            # Add other CAM methods
            all_cams.extend(other_cams)

            # Get actual image filename
            img_name = str(all_items[idx]).split("/")[-1]

            # Save per-image timing with image name
            image_timing['image'] = img_name
            self.cascam_timing_breakdown['per_image_cam_times'].append(image_timing)

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

            # Save figure only if within max_figures limit
            if idx < max_figures:
                # Create visualization with display names
                display_names = ['CasCAM (proposed)', 'CAM'] + method_names[2:]
                fig = CasCAMVisualizer.make_figure(img, all_cams, display_names)

                # Save figure with actual image filename
                save_dir = self.config.get_fig_dir(lambda_val)
                img_name_no_ext = img_name.split(".")[0]
                save_path = f"{save_dir}/{img_name_no_ext}_comparison.pdf"
                CasCAMVisualizer.save_figure(fig, save_path)

        print(f"Processed all {total_images} images for λ={lambda_val}")
        print(f"Saved {max_figures} comparison figures")
        print(f"Saved CAMs for {len(method_names)} methods")
    
    def evaluate_iou(self, annotation_dir, artifact_masks_dir=None):
        """
        Evaluate IoU for all methods and create comparison tables

        Args:
            annotation_dir: Path to annotation directory containing trimaps
            artifact_masks_dir: Path to artifact masks directory (optional)

        Returns:
            Dictionary of IoU results for each lambda value
        """
        print("\n" + "="*60)
        print("IoU Evaluation")
        if self.config.eval_use_topk:
            print(f"Evaluation Method: Top-{int(self.config.eval_k_percent*100)}%")
        else:
            print(f"Evaluation Method: Fixed Threshold (0.5)")
        print("="*60)

        # Initialize evaluator
        evaluator = IoUEvaluator(annotation_dir, artifact_masks_dir)

        # Store all results
        iou_results = {}

        for lambda_val in self.config.lambda_vals:
            print(f"\nEvaluating λ={lambda_val}")

            # Get CAMs for this lambda value
            cams_dict = self.saved_cams.get(lambda_val, {})

            # Debug: print CAM storage status
            print(f"  DEBUG: saved_cams keys: {list(self.saved_cams.keys())}")
            print(f"  DEBUG: cams_dict keys for λ={lambda_val}: {list(cams_dict.keys())}")
            if cams_dict:
                for method_name, cams_list in cams_dict.items():
                    print(f"  DEBUG: Method '{method_name}' has {len(cams_list)} CAMs")

            if not cams_dict:
                print(f"  Warning: No CAMs found for λ={lambda_val}")
                continue

            # === EVALUATION 1: Object Localization (vs GT annotation) ===
            print(f"\n  [1/2] Object Localization Evaluation (vs GT annotation)")
            object_results_df = evaluator.evaluate_object_localization(
                cams_dict,
                self.image_names,
                threshold=0.5,
                use_topk=self.config.eval_use_topk,
                k_percent=self.config.eval_k_percent
            )

            # Create summary table
            object_summary_df = evaluator.create_summary_table(object_results_df)

            # Create comparison table
            object_comparison_df = create_comparison_table(object_results_df, baseline_method='CAM')

            # Save results
            save_dir = self.config.get_evaluation_dir(lambda_val)
            os.makedirs(save_dir, exist_ok=True)

            object_results_df.to_csv(f"{save_dir}/object_localization_detailed.csv", index=False)
            object_summary_df.to_csv(f"{save_dir}/object_localization_summary.csv", index=False)
            object_comparison_df.to_csv(f"{save_dir}/object_localization_comparison.csv", index=False)

            # Print summary
            print(f"\n  Object Localization Summary (λ={lambda_val}):")
            print(object_summary_df.to_string(index=False))

            print(f"\n  Object Localization Comparison vs CAM (λ={lambda_val}):")
            print(object_comparison_df.to_string(index=False))

            # === EVALUATION 2: Artifact Detection (vs artifact boxes) ===
            print(f"\n  [2/2] Artifact Detection Evaluation (vs artifact boxes)")
            artifact_results_df = evaluator.evaluate_artifact_detection(
                cams_dict,
                self.image_names,
                threshold=0.5,
                use_topk=self.config.eval_use_topk,
                k_percent=self.config.eval_k_percent
            )

            if not artifact_results_df.empty:
                # Create summary table
                artifact_summary_df = evaluator.create_summary_table(artifact_results_df)

                # Create comparison table
                artifact_comparison_df = create_comparison_table(artifact_results_df, baseline_method='CAM')

                # Save results
                artifact_results_df.to_csv(f"{save_dir}/artifact_detection_detailed.csv", index=False)
                artifact_summary_df.to_csv(f"{save_dir}/artifact_detection_summary.csv", index=False)
                artifact_comparison_df.to_csv(f"{save_dir}/artifact_detection_comparison.csv", index=False)

                # Print summary
                print(f"\n  Artifact Detection Summary (λ={lambda_val}):")
                print(artifact_summary_df.to_string(index=False))

                print(f"\n  Artifact Detection Comparison vs CAM (λ={lambda_val}):")
                print(artifact_comparison_df.to_string(index=False))
            else:
                print(f"\n  No artifact masks found - skipping artifact detection evaluation")

            # === EVALUATION 3: Cross Analysis (Object vs Artifact) ===
            print(f"\n  [3/3] Cross-Analysis Evaluation (Object vs Artifact relationship)")
            cross_results_df = evaluator.evaluate_cross_analysis(
                cams_dict,
                self.image_names,
                threshold=0.5,
                use_topk=self.config.eval_use_topk,
                k_percent=self.config.eval_k_percent
            )

            if not cross_results_df.empty:
                # Create summary table
                cross_summary_df = evaluator.create_summary_table(cross_results_df)

                # Create comparison table
                cross_comparison_df = create_comparison_table(cross_results_df, baseline_method='CAM')

                # Save results in cross_analysis subdirectory
                cross_save_dir = f"{save_dir}/cross_analysis"
                os.makedirs(cross_save_dir, exist_ok=True)

                cross_results_df.to_csv(f"{cross_save_dir}/per_image.csv", index=False)
                cross_summary_df.to_csv(f"{cross_save_dir}/summary.csv", index=False)
                cross_comparison_df.to_csv(f"{cross_save_dir}/vs_CAM.csv", index=False)

                # Print summary (key metrics only)
                print(f"\n  Cross-Analysis Summary (λ={lambda_val}):")
                key_cols = ['Method', 'clean_object_precision_mean', 'artifact_contamination_mean',
                           'distraction_score_mean', 'dependency_ratio_mean']
                display_cols = [col for col in key_cols if col in cross_summary_df.columns]
                if display_cols:
                    print(cross_summary_df[display_cols].to_string(index=False))
                else:
                    print(cross_summary_df.to_string(index=False))

                print(f"\n  Cross-Analysis Comparison vs CAM (λ={lambda_val}):")
                print(cross_comparison_df.to_string(index=False))
            else:
                print(f"\n  No artifact masks found - skipping cross-analysis evaluation")
                cross_results_df = pd.DataFrame()
                cross_summary_df = None
                cross_comparison_df = None

            # Store results
            iou_results[lambda_val] = {
                'object_localization': {
                    'detailed': object_results_df,
                    'summary': object_summary_df,
                    'comparison': object_comparison_df
                },
                'artifact_detection': {
                    'detailed': artifact_results_df,
                    'summary': artifact_summary_df if not artifact_results_df.empty else None,
                    'comparison': artifact_comparison_df if not artifact_results_df.empty else None
                },
                'cross_analysis': {
                    'detailed': cross_results_df,
                    'summary': cross_summary_df,
                    'comparison': cross_comparison_df
                }
            }

        return iou_results

    def evaluate_advanced_metrics(self, annotation_dir, artifact_masks_dir=None):
        """
        Evaluate comprehensive advanced metrics for all methods

        Args:
            annotation_dir: Path to annotation directory containing trimaps
            artifact_masks_dir: Optional path to artifact masks directory

        Returns:
            Dictionary of advanced evaluation results for each lambda value
        """
        print("\n" + "="*60)
        print("Advanced Metrics Evaluation")
        if self.config.eval_use_topk:
            print(f"Evaluation Method: Top-{int(self.config.eval_k_percent*100)}%")
        else:
            print(f"Evaluation Method: Fixed Threshold (0.5)")
        print("="*60)

        # Initialize advanced evaluator
        evaluator = AdvancedCAMEvaluator(
            annotation_dir=annotation_dir,
            artifact_masks_dir=artifact_masks_dir
        )

        # Store all results
        advanced_results = {}

        for lambda_val in self.config.lambda_vals:
            print(f"\nEvaluating λ={lambda_val} with comprehensive metrics")

            # Get CAMs for this lambda value
            cams_dict = self.saved_cams.get(lambda_val, {})

            if not cams_dict:
                print(f"  Warning: No CAMs found for λ={lambda_val}")
                continue

            # Run comprehensive evaluation
            results_df = evaluator.evaluate_multiple_methods(
                cams_dict,
                self.image_names,
                threshold=0.5,
                use_topk=self.config.eval_use_topk,
                k_percent=self.config.eval_k_percent,
                include_curves=True,
                include_sweep=False  # Set to True for threshold sweep analysis
            )

            # Create summary report
            summary_df = evaluator.create_summary_report(
                results_df,
                save_dir=self.config.get_evaluation_dir(lambda_val)
            )

            # Create baseline comparison
            comparison_df = compare_with_baseline(results_df, baseline_method='CAM')

            # Save results
            save_dir = self.config.get_evaluation_dir(lambda_val)
            os.makedirs(save_dir, exist_ok=True)

            # Print key results
            print(f"\n  Summary Statistics (λ={lambda_val}):")
            # Show only key metrics
            key_metrics = [col for col in summary_df.columns
                          if any(metric in col for metric in ['Method', 'iou_mean', 'dice_mean', 'ap_mean',
                                                              'top15_precision_mean', 'pointing_game_hit_mean',
                                                              'centroid_distance_mean', 'boundary_f1_mean'])]
            if key_metrics:
                print(summary_df[key_metrics].to_string(index=False))
            else:
                print(summary_df.to_string(index=False))

            if not comparison_df.empty:
                print(f"\n  Improvements vs CAM (λ={lambda_val}):")
                print(comparison_df.to_string(index=False))

            # Store results
            advanced_results[lambda_val] = {
                'detailed': results_df,
                'summary': summary_df,
                'comparison': comparison_df
            }

        return advanced_results

    def save_cams(self):
        """Save CAM numpy arrays to files for each lambda value and method"""
        print("\n" + "="*60)
        print("Saving CAM files")
        print("="*60)

        for lambda_val, methods_dict in self.saved_cams.items():
            cams_dir = self.config.get_cams_dir(lambda_val)
            os.makedirs(cams_dir, exist_ok=True)

            saved_count = 0
            for method_name, cams_list in methods_dict.items():
                for idx, cam in enumerate(cams_list):
                    if idx < len(self.image_names):
                        img_name = self.image_names[idx]
                        img_name_no_ext = img_name.rsplit('.', 1)[0]
                        # Convert to numpy if tensor
                        if hasattr(cam, 'cpu'):
                            cam_np = cam.cpu().numpy()
                        else:
                            cam_np = np.array(cam)
                        save_path = f"{cams_dir}/{method_name}_{img_name_no_ext}.npy"
                        np.save(save_path, cam_np)
                        saved_count += 1

            print(f"  lambda={lambda_val}: Saved {saved_count} CAM files to {cams_dir}")

    def save_computation_times(self):
        """Save computation time statistics to CSV files with detailed breakdown"""
        import pandas as pd

        print("\n" + "="*60)
        print("Computation Time Summary")
        print("="*60)

        breakdown = self.cascam_timing_breakdown

        # Print summary
        print("\n[1] Pure training time per iteration:")
        for i, t in enumerate(breakdown['training_times']):
            print(f"    iter={i+1}: {t:.4f}s")

        print("\n[2] Threshold processing time between iterations:")
        for i, t in enumerate(breakdown['threshold_times']):
            print(f"    iter {i+1} → {i+2}: {t:.4f}s")

        print("\n[3] Image save time between iterations:")
        for i, t in enumerate(breakdown['image_save_times']):
            print(f"    iter {i+1} → {i+2}: {t:.4f}s")

        print("\n[4] Dataloader construction time:")
        for i, t in enumerate(breakdown['dataloader_construction_times']):
            print(f"    iter {i+1} → {i+2}: {t:.4f}s")

        print("\n[5] Per-image CAM generation time (mean):")
        cam_times = breakdown['cam_generation_times']
        for method_name, times in cam_times.items():
            if len(times) > 0:
                print(f"    {method_name}: {np.mean(times):.6f}s")

        # Save timing files to timing directory
        timing_dir = self.config.get_timing_dir()
        os.makedirs(timing_dir, exist_ok=True)

        # (1-4) Iteration-level timing CSV
        iteration_data = []
        for i in range(self.config.num_iter):
            row = {
                'iteration': i + 1,
                'training_time': breakdown['training_times'][i] if i < len(breakdown['training_times']) else 0
            }
            # These only exist for iterations 1 to num_iter-1
            if i < self.config.num_iter - 1:
                row['threshold_time'] = breakdown['threshold_times'][i] if i < len(breakdown['threshold_times']) else 0
                row['image_save_time'] = breakdown['image_save_times'][i] if i < len(breakdown['image_save_times']) else 0
                row['dataloader_construction_time'] = breakdown['dataloader_construction_times'][i] if i < len(breakdown['dataloader_construction_times']) else 0
            else:
                row['threshold_time'] = None
                row['image_save_time'] = None
                row['dataloader_construction_time'] = None
            iteration_data.append(row)

        iteration_df = pd.DataFrame(iteration_data)
        iteration_csv = f"{timing_dir}/timing_per_iteration.csv"
        iteration_df.to_csv(iteration_csv, index=False)
        print(f"\nSaved iteration timing to: {iteration_csv}")

        # (5) Per-image CAM timing CSV (table format)
        if breakdown['per_image_cam_times']:
            per_image_df = pd.DataFrame(breakdown['per_image_cam_times'])
            # Reorder columns to have 'image' first
            cols = ['image'] + [col for col in per_image_df.columns if col != 'image']
            per_image_df = per_image_df[cols]
            per_image_csv = f"{timing_dir}/timing_per_image_cam.csv"
            per_image_df.to_csv(per_image_csv, index=False)
            print(f"Saved per-image CAM timing to: {per_image_csv}")

            # Print first few rows
            print("\nPer-image CAM timing (first 5 rows):")
            print(per_image_df.head().to_string(index=False))

        # Also save legacy computation_times.csv for compatibility
        comparison_data = []
        for method_name, times in cam_times.items():
            if len(times) > 0:
                comparison_data.append({
                    'Method': method_name,
                    'Mean CAM Time (s)': np.mean(times),
                    'Total CAM Time (s)': np.sum(times),
                    'Count': len(times)
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Mean CAM Time (s)')
            save_path = f"{timing_dir}/computation_times.csv"
            comparison_df.to_csv(save_path, index=False)
            print(f"Saved legacy computation times to: {save_path}")

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

        # Save CAM files
        self.save_cams()

        # Save configuration
        self.config.save_config()

        return all_results