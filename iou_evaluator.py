"""
IoU (Intersection over Union) Evaluator for CAM methods

This module calculates IoU between CAM heatmaps and ground truth annotations (trimaps).
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import pandas as pd


class IoUEvaluator:
    """Calculate IoU metrics for CAM evaluation"""

    def __init__(self, annotation_dir):
        """
        Initialize IoU Evaluator

        Args:
            annotation_dir: Path to trimap annotations directory
        """
        self.annotation_dir = Path(annotation_dir)

    def load_trimap(self, image_name):
        """
        Load trimap annotation

        Args:
            image_name: Image filename (e.g., 'Abyssinian_1.jpg')

        Returns:
            Binary mask (foreground=1, background=0)
        """
        # Convert image name to trimap name
        trimap_name = image_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        trimap_path = self.annotation_dir / 'trimaps' / trimap_name

        if not trimap_path.exists():
            raise FileNotFoundError(f"Trimap not found: {trimap_path}")

        # Load trimap (values: 1=foreground, 2=boundary, 3=background)
        trimap = np.array(Image.open(trimap_path))

        # Convert to binary: 1=foreground, 0=background/boundary
        # Only pixels labeled as 1 (foreground) are considered positive
        binary_mask = (trimap == 1).astype(np.uint8)

        return binary_mask

    def cam_to_binary_mask(self, cam, threshold=0.5):
        """
        Convert CAM to binary mask using threshold

        Args:
            cam: CAM tensor or numpy array (H, W)
            threshold: Threshold value (0-1)

        Returns:
            Binary mask
        """
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()

        # Normalize CAM to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = cam

        # Apply threshold
        binary_mask = (cam_norm >= threshold).astype(np.uint8)

        return binary_mask

    def calculate_iou(self, pred_mask, gt_mask):
        """
        Calculate IoU between prediction and ground truth masks

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask

        Returns:
            IoU score (0-1)
        """
        # Ensure same shape
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")

        # Calculate intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        # Avoid division by zero
        if union == 0:
            return 0.0

        iou = intersection / union
        return float(iou)

    def evaluate_cam(self, cam, image_name, threshold=0.5):
        """
        Evaluate single CAM against ground truth

        Args:
            cam: CAM tensor or numpy array
            image_name: Image filename
            threshold: Threshold for binarization

        Returns:
            IoU score
        """
        # Load ground truth
        gt_mask = self.load_trimap(image_name)

        # Resize CAM to match ground truth size if needed
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()

        if cam.shape != gt_mask.shape:
            from scipy.ndimage import zoom
            scale_factors = (gt_mask.shape[0] / cam.shape[0],
                           gt_mask.shape[1] / cam.shape[1])
            cam = zoom(cam, scale_factors, order=1)

        # Convert to binary mask
        pred_mask = self.cam_to_binary_mask(cam, threshold)

        # Calculate IoU
        iou = self.calculate_iou(pred_mask, gt_mask)

        return iou

    def evaluate_multiple_cams(self, cams_dict, image_names, threshold=0.5):
        """
        Evaluate multiple CAM methods

        Args:
            cams_dict: Dictionary of {method_name: list_of_cams}
            image_names: List of image filenames
            threshold: Threshold for binarization

        Returns:
            DataFrame with IoU results
        """
        results = []

        for idx, image_name in enumerate(image_names):
            row = {'image': image_name}

            for method_name, cams in cams_dict.items():
                try:
                    iou = self.evaluate_cam(cams[idx], image_name, threshold)
                    row[method_name] = iou
                except Exception as e:
                    print(f"Warning: Failed to evaluate {method_name} for {image_name}: {e}")
                    row[method_name] = np.nan

            results.append(row)

        df = pd.DataFrame(results)
        return df

    def create_summary_table(self, results_df, save_path=None):
        """
        Create summary statistics table

        Args:
            results_df: DataFrame from evaluate_multiple_cams
            save_path: Optional path to save table

        Returns:
            Summary DataFrame
        """
        # Calculate statistics for each method
        methods = [col for col in results_df.columns if col != 'image']

        summary_data = []
        for method in methods:
            values = results_df[method].dropna()
            if len(values) > 0:
                summary_data.append({
                    'Method': method,
                    'Mean IoU': values.mean(),
                    'Std IoU': values.std(),
                    'Median IoU': values.median(),
                    'Min IoU': values.min(),
                    'Max IoU': values.max(),
                    'Count': len(values)
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean IoU', ascending=False)

        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"Summary table saved to: {save_path}")

        return summary_df


def create_comparison_table(results_df, baseline_method='CAM'):
    """
    Create comparison table with improvements over baseline

    Args:
        results_df: DataFrame from evaluate_multiple_cams
        baseline_method: Name of baseline method to compare against

    Returns:
        DataFrame with comparison statistics
    """
    methods = [col for col in results_df.columns if col != 'image']

    if baseline_method not in methods:
        print(f"Warning: Baseline method '{baseline_method}' not found in results")
        baseline_method = methods[0]

    baseline_scores = results_df[baseline_method].values

    comparison_data = []
    for method in methods:
        scores = results_df[method].values

        # Calculate improvement
        valid_idx = ~(np.isnan(scores) | np.isnan(baseline_scores))
        if valid_idx.sum() > 0:
            improvements = scores[valid_idx] - baseline_scores[valid_idx]

            comparison_data.append({
                'Method': method,
                'Mean IoU': np.nanmean(scores),
                'vs ' + baseline_method: np.mean(improvements),
                'Improved (%)': (improvements > 0).mean() * 100,
                'Same (%)': (improvements == 0).mean() * 100,
                'Worse (%)': (improvements < 0).mean() * 100
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Mean IoU', ascending=False)

    return comparison_df
