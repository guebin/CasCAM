"""
CAM Evaluation Metrics

Comprehensive evaluation suite for CAM methods including:
- Basic metrics: IoU, Dice/F1, Precision, Recall
- Curve-based metrics: PR Curve, ROC Curve, AP, AUC
- Localization metrics: Top-k Precision, Pointing Game, Centroid Distance
- Boundary metrics: Boundary F1, Chamfer Distance, Hausdorff Distance
- Artifact detection: FPR, Outlier Score
- Analysis statistics: Threshold sweep, distribution analysis

This module provides two evaluators:
- CAMEvaluator: Comprehensive evaluation with all metrics
- IoUEvaluator: Simple IoU-only evaluation (lightweight, faster)
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import auc, roc_curve, precision_recall_curve


class AdvancedCAMEvaluator:
    """Comprehensive CAM evaluation with multiple metrics"""

    def __init__(self, annotation_dir, artifact_masks_dir=None):
        """
        Initialize Advanced CAM Evaluator

        Args:
            annotation_dir: Path to trimap annotations directory
            artifact_masks_dir: Optional path to artifact masks directory
        """
        self.annotation_dir = Path(annotation_dir)
        self.artifact_masks_dir = Path(artifact_masks_dir) if artifact_masks_dir else None

    def load_trimap(self, image_name):
        """Load trimap annotation and convert to binary mask"""
        trimap_name = image_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        trimap_path = self.annotation_dir / 'trimaps' / trimap_name

        if not trimap_path.exists():
            raise FileNotFoundError(f"Trimap not found: {trimap_path}")

        trimap = np.array(Image.open(trimap_path))
        binary_mask = (trimap == 1).astype(np.uint8)

        return binary_mask

    def load_artifact_mask(self, image_name):
        """Load artifact mask if available"""
        if self.artifact_masks_dir is None:
            return None

        # Convert image name to artifact mask name (add _artifact suffix)
        mask_name = image_name.replace('.jpg', '_artifact.png').replace('.jpeg', '_artifact.png')
        mask_path = self.artifact_masks_dir / mask_name

        if not mask_path.exists():
            return None

        artifact_mask = np.array(Image.open(mask_path))
        return (artifact_mask > 0).astype(np.uint8)

    def normalize_cam(self, cam):
        """Normalize CAM to [0, 1] range"""
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()

        # Ensure CAM is 2D (squeeze out any extra dimensions)
        while cam.ndim > 2:
            cam = cam.squeeze()

        # If still not 2D after squeeze, take first/last channel
        if cam.ndim > 2:
            if cam.shape[0] == 1:
                cam = cam[0]
            elif cam.shape[-1] == 1:
                cam = cam[..., 0]
            else:
                # Take first channel as fallback
                cam = cam[0] if cam.shape[0] < cam.shape[-1] else cam[..., 0]

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam)

        return cam_norm

    def resize_cam(self, cam, target_shape):
        """Resize CAM to match target shape"""
        if cam.shape == target_shape:
            return cam

        from scipy.ndimage import zoom
        scale_factors = (target_shape[0] / cam.shape[0],
                        target_shape[1] / cam.shape[1])
        cam_resized = zoom(cam, scale_factors, order=1)

        return cam_resized

    # ========== Basic Segmentation Metrics ==========

    def calculate_iou(self, pred_mask, gt_mask):
        """Calculate Intersection over Union"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)

    def calculate_dice(self, pred_mask, gt_mask):
        """Calculate Dice coefficient (F1 score)"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()

        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()

        if pred_sum + gt_sum == 0:
            return 0.0

        dice = 2.0 * intersection / (pred_sum + gt_sum)
        return float(dice)

    def calculate_precision_recall(self, pred_mask, gt_mask):
        """Calculate Precision and Recall"""
        tp = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, ~gt_mask.astype(bool)).sum()
        fn = np.logical_and(~pred_mask.astype(bool), gt_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return float(precision), float(recall)

    def cam_to_binary_mask(self, cam, gt_mask, threshold=None, use_topk=False, k_percent=0.10):
        """
        Convert CAM to binary mask using either threshold or top-k method

        Args:
            cam: CAM array (any shape)
            gt_mask: Ground truth mask for shape reference
            threshold: Fixed threshold value (0-1). If None and use_topk=False, uses 0.5
            use_topk: If True, use top-k% method instead of fixed threshold
            k_percent: Percentage of top activations to keep (0-1, default 0.10 = 10%)

        Returns:
            pred_mask: Binary mask (numpy array)
            cam_resized: Normalized and resized CAM
        """
        cam_norm = self.normalize_cam(cam)
        cam_resized = self.resize_cam(cam_norm, gt_mask.shape)

        if use_topk:
            # Top-k method: keep only top k% of pixels
            flat_cam = cam_resized.flatten()
            threshold_val = np.percentile(flat_cam, (1 - k_percent) * 100)
            pred_mask = (cam_resized >= threshold_val).astype(np.uint8)
        else:
            # Fixed threshold method
            if threshold is None:
                threshold = 0.5
            pred_mask = (cam_resized >= threshold).astype(np.uint8)

        return pred_mask, cam_resized

    def calculate_basic_metrics(self, cam, gt_mask, threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Calculate all basic segmentation metrics

        Args:
            cam: CAM array
            gt_mask: Ground truth mask
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            Dictionary with IoU, Dice, Precision, Recall, F1
        """
        pred_mask, _ = self.cam_to_binary_mask(cam, gt_mask, threshold, use_topk, k_percent)

        iou = self.calculate_iou(pred_mask, gt_mask)
        dice = self.calculate_dice(pred_mask, gt_mask)
        precision, recall = self.calculate_precision_recall(pred_mask, gt_mask)

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # ========== Threshold Sweep and Curve-based Metrics ==========

    def threshold_sweep(self, cam, gt_mask, thresholds=None):
        """
        Evaluate metrics across multiple thresholds

        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        cam_norm = self.normalize_cam(cam)
        cam_resized = self.resize_cam(cam_norm, gt_mask.shape)

        results = []
        for threshold in thresholds:
            pred_mask = (cam_resized >= threshold).astype(np.uint8)

            iou = self.calculate_iou(pred_mask, gt_mask)
            dice = self.calculate_dice(pred_mask, gt_mask)
            precision, recall = self.calculate_precision_recall(pred_mask, gt_mask)

            results.append({
                'threshold': threshold,
                'iou': iou,
                'dice': dice,
                'precision': precision,
                'recall': recall
            })

        return pd.DataFrame(results)

    def calculate_pr_curve(self, cam, gt_mask):
        """Calculate Precision-Recall curve and Average Precision"""
        cam_norm = self.normalize_cam(cam)
        cam_resized = self.resize_cam(cam_norm, gt_mask.shape)

        # Flatten arrays
        y_true = gt_mask.flatten()
        y_scores = cam_resized.flatten()

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Calculate AP using AUC of PR curve
        ap = auc(recall, precision)

        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'ap': ap
        }

    def calculate_roc_curve(self, cam, gt_mask):
        """Calculate ROC curve and AUC"""
        cam_norm = self.normalize_cam(cam)
        cam_resized = self.resize_cam(cam_norm, gt_mask.shape)

        # Flatten arrays
        y_true = gt_mask.flatten()
        y_scores = cam_resized.flatten()

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # Calculate AUC
        auc_score = auc(fpr, tpr)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc_score
        }

    # ========== Localization-specific Metrics ==========

    def calculate_topk_precision(self, cam, gt_mask, k_percent=0.15):
        """
        Energy Inside GT: Calculate precision using top k% of CAM activations

        Args:
            k_percent: Percentage of top activations to consider (0-1)
        """
        cam_norm = self.normalize_cam(cam)
        cam_resized = self.resize_cam(cam_norm, gt_mask.shape)

        # Get threshold for top k%
        flat_cam = cam_resized.flatten()
        threshold = np.percentile(flat_cam, (1 - k_percent) * 100)

        # Create mask for top k% activations
        topk_mask = (cam_resized >= threshold).astype(np.uint8)

        # Calculate precision: how much of top k% is inside GT
        if topk_mask.sum() == 0:
            return 0.0

        precision = np.logical_and(topk_mask, gt_mask).sum() / topk_mask.sum()

        # Also calculate energy inside GT
        total_energy = cam_resized.sum()
        energy_in_gt = (cam_resized * gt_mask).sum()
        energy_ratio = energy_in_gt / total_energy if total_energy > 0 else 0.0

        return {
            f'top{int(k_percent*100)}_precision': float(precision),
            f'top{int(k_percent*100)}_energy_ratio': float(energy_ratio)
        }

    def calculate_pointing_game(self, cam, gt_mask):
        """
        Pointing Game: Check if maximum activation point is inside GT

        Returns hit rate (0 or 1)
        """
        cam_norm = self.normalize_cam(cam)
        cam_resized = self.resize_cam(cam_norm, gt_mask.shape)

        # Find location of maximum activation
        max_idx = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)

        # Check if inside GT
        hit = int(gt_mask[max_idx] == 1)

        return {
            'pointing_game_hit': hit,
            'max_location': max_idx
        }

    def calculate_centroid_distance(self, cam, gt_mask):
        """
        Calculate distance between CAM centroid and GT centroid

        Returns normalized distance (0-1, where 0 is perfect)
        """
        cam_norm = self.normalize_cam(cam)
        cam_resized = self.resize_cam(cam_norm, gt_mask.shape)

        # Calculate CAM centroid (weighted by activation)
        y_indices, x_indices = np.mgrid[0:cam_resized.shape[0], 0:cam_resized.shape[1]]
        total_activation = cam_resized.sum()

        if total_activation == 0:
            cam_centroid = (cam_resized.shape[0] / 2, cam_resized.shape[1] / 2)
        else:
            cam_centroid_y = (y_indices * cam_resized).sum() / total_activation
            cam_centroid_x = (x_indices * cam_resized).sum() / total_activation
            cam_centroid = (cam_centroid_y, cam_centroid_x)

        # Calculate GT centroid
        if gt_mask.sum() == 0:
            return {'centroid_distance': 1.0}

        gt_centroid_y = (y_indices * gt_mask).sum() / gt_mask.sum()
        gt_centroid_x = (x_indices * gt_mask).sum() / gt_mask.sum()
        gt_centroid = (gt_centroid_y, gt_centroid_x)

        # Calculate Euclidean distance
        distance = np.sqrt((cam_centroid[0] - gt_centroid[0])**2 +
                          (cam_centroid[1] - gt_centroid[1])**2)

        # Normalize by image diagonal
        diagonal = np.sqrt(cam_resized.shape[0]**2 + cam_resized.shape[1]**2)
        normalized_distance = distance / diagonal

        return {
            'centroid_distance': float(normalized_distance),
            'cam_centroid': cam_centroid,
            'gt_centroid': gt_centroid
        }

    # ========== Boundary Metrics ==========

    def get_boundary_pixels(self, mask, thickness=2):
        """Extract boundary pixels from binary mask"""
        eroded = ndimage.binary_erosion(mask, iterations=thickness)
        boundary = mask.astype(bool) & ~eroded
        return boundary.astype(np.uint8)

    def calculate_boundary_f1(self, cam, gt_mask, threshold=0.5, boundary_thickness=2, use_topk=False, k_percent=0.10):
        """
        Calculate F1 score for boundary pixels only

        Args:
            boundary_thickness: Thickness of boundary region in pixels
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)
        """
        pred_mask, _ = self.cam_to_binary_mask(cam, gt_mask, threshold, use_topk, k_percent)

        # Extract boundaries
        pred_boundary = self.get_boundary_pixels(pred_mask, boundary_thickness)
        gt_boundary = self.get_boundary_pixels(gt_mask, boundary_thickness)

        # Calculate boundary F1
        tp = np.logical_and(pred_boundary, gt_boundary).sum()
        fp = np.logical_and(pred_boundary, ~gt_boundary.astype(bool)).sum()
        fn = np.logical_and(~pred_boundary.astype(bool), gt_boundary).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'boundary_f1': float(f1),
            'boundary_precision': float(precision),
            'boundary_recall': float(recall)
        }

    def calculate_chamfer_distance(self, cam, gt_mask, threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Calculate Chamfer Distance between predicted and GT boundaries

        Args:
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            Dictionary with normalized average distance
        """
        pred_mask, cam_resized = self.cam_to_binary_mask(cam, gt_mask, threshold, use_topk, k_percent)

        # Extract boundary coordinates
        pred_boundary = self.get_boundary_pixels(pred_mask)
        gt_boundary = self.get_boundary_pixels(gt_mask)

        pred_coords = np.argwhere(pred_boundary)
        gt_coords = np.argwhere(gt_boundary)

        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return {'chamfer_distance': 1.0}

        # Calculate one-way distances
        from scipy.spatial.distance import cdist
        dist_pred_to_gt = cdist(pred_coords, gt_coords).min(axis=1).mean()
        dist_gt_to_pred = cdist(gt_coords, pred_coords).min(axis=1).mean()

        # Chamfer distance is average of both directions
        chamfer = (dist_pred_to_gt + dist_gt_to_pred) / 2

        # Normalize by image diagonal
        diagonal = np.sqrt(cam_resized.shape[0]**2 + cam_resized.shape[1]**2)
        chamfer_normalized = chamfer / diagonal

        return {
            'chamfer_distance': float(chamfer_normalized),
            'chamfer_distance_raw': float(chamfer)
        }

    def calculate_hausdorff_distance(self, cam, gt_mask, threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Calculate Hausdorff Distance between boundaries

        Args:
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)
        """
        pred_mask, cam_resized = self.cam_to_binary_mask(cam, gt_mask, threshold, use_topk, k_percent)

        # Extract boundary coordinates
        pred_boundary = self.get_boundary_pixels(pred_mask)
        gt_boundary = self.get_boundary_pixels(gt_mask)

        pred_coords = np.argwhere(pred_boundary)
        gt_coords = np.argwhere(gt_boundary)

        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return {'hausdorff_distance': 1.0}

        # Calculate Hausdorff distance
        hausdorff_dist = max(
            directed_hausdorff(pred_coords, gt_coords)[0],
            directed_hausdorff(gt_coords, pred_coords)[0]
        )

        # Normalize by image diagonal
        diagonal = np.sqrt(cam_resized.shape[0]**2 + cam_resized.shape[1]**2)
        hausdorff_normalized = hausdorff_dist / diagonal

        return {
            'hausdorff_distance': float(hausdorff_normalized),
            'hausdorff_distance_raw': float(hausdorff_dist)
        }

    # ========== Artifact Detection Metrics ==========

    def calculate_artifact_fpr(self, cam, gt_mask, artifact_mask, threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Calculate False Positive Rate in artifact regions

        Args:
            artifact_mask: Binary mask where 1 = artifact region
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)
        """
        if artifact_mask is None:
            return {}

        pred_mask, cam_resized = self.cam_to_binary_mask(cam, gt_mask, threshold, use_topk, k_percent)

        # Resize artifact mask to match GT shape
        artifact_mask_resized = self.resize_cam(artifact_mask.astype(np.float32), gt_mask.shape)
        artifact_mask_resized = (artifact_mask_resized > 0.5).astype(np.uint8)

        # Calculate FP in artifact regions
        artifact_pixels = artifact_mask_resized.sum()
        if artifact_pixels == 0:
            return {'artifact_fpr': 0.0}

        false_positives_in_artifacts = np.logical_and(pred_mask, artifact_mask_resized).sum()
        artifact_fpr = false_positives_in_artifacts / artifact_pixels

        # Also calculate artifact coverage by high activations
        topk_threshold = np.percentile(cam_resized, 85)  # Top 15%
        topk_in_artifacts = ((cam_resized >= topk_threshold) & artifact_mask_resized).sum()
        topk_total = (cam_resized >= topk_threshold).sum()
        artifact_coverage = topk_in_artifacts / topk_total if topk_total > 0 else 0.0

        return {
            'artifact_fpr': float(artifact_fpr),
            'artifact_coverage_top15': float(artifact_coverage)
        }

    def calculate_cam_outlier_score(self, cam):
        """
        Calculate outlier/artifact detection score based on CAM distribution

        Measures sparsity and multi-modality of CAM
        """
        cam_norm = self.normalize_cam(cam)
        flat_cam = cam_norm.flatten()

        # Gini coefficient (sparsity measure)
        sorted_cam = np.sort(flat_cam)
        n = len(sorted_cam)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_cam)) / (n * np.sum(sorted_cam)) - (n + 1) / n

        # Entropy
        hist, _ = np.histogram(flat_cam, bins=50, range=(0, 1))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-10))

        # Number of clusters (modes) using threshold-based counting
        binary_high = (cam_norm > 0.7).astype(np.uint8)
        labeled, num_clusters = ndimage.label(binary_high)

        return {
            'gini_coefficient': float(gini),
            'entropy': float(entropy),
            'num_high_clusters': int(num_clusters),
            'max_activation': float(cam_norm.max()),
            'mean_activation': float(cam_norm.mean())
        }

    def calculate_cross_analysis_metrics(self, cam, gt_mask, artifact_mask,
                                         threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Calculate cross-analysis metrics between object and artifact

        Args:
            cam: CAM heatmap
            gt_mask: Ground truth object mask
            artifact_mask: Artifact region mask
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            dict with cross-analysis metrics:
            - artifact_fpr: 객체 찾으려다 artifact 오탐 비율
            - clean_object_precision: Artifact 제외 순수 객체 정확도
            - artifact_contamination: 예측 중 artifact 비율
            - object_purity: 의미있는 활성화 중 객체 비율
            - distraction_score: 객체 대비 artifact 현혹도
            - dependency_ratio: Artifact 의존도
            - object_iou: Object IoU (참고용)
            - artifact_iou: Artifact IoU (참고용)
        """
        if artifact_mask is None:
            return {}

        # CAM to binary mask
        pred_mask, cam_resized = self.cam_to_binary_mask(cam, gt_mask, threshold, use_topk, k_percent)

        # Resize masks to match prediction shape
        gt_resized = self.resize_cam(gt_mask.astype(np.float32), pred_mask.shape)
        artifact_resized = self.resize_cam(artifact_mask.astype(np.float32), pred_mask.shape)

        gt_binary = (gt_resized > 0.5).astype(np.uint8)
        artifact_binary = (artifact_resized > 0.5).astype(np.uint8)

        # Basic intersections
        pred_and_object = np.logical_and(pred_mask, gt_binary)
        pred_and_artifact = np.logical_and(pred_mask, artifact_binary)
        pred_and_object_clean = np.logical_and(pred_and_object, ~artifact_binary)

        # Calculate metrics
        total_pred = pred_mask.sum()
        total_artifact = artifact_binary.sum()

        # 1. Artifact FPR (객체 찾으려다 artifact 오탐)
        artifact_fpr = pred_and_artifact.sum() / total_artifact if total_artifact > 0 else 0.0

        # 2. Clean Object Precision (아티팩트 제외 순수 객체 정확도)
        clean_precision = pred_and_object_clean.sum() / total_pred if total_pred > 0 else 0.0

        # 3. Artifact Contamination (예측 중 artifact 비율)
        contamination = pred_and_artifact.sum() / total_pred if total_pred > 0 else 0.0

        # 4. Object Purity (의미있는 활성화 중 객체 비율)
        meaningful = np.logical_or(gt_binary, artifact_binary)
        pred_and_meaningful = np.logical_and(pred_mask, meaningful)
        purity = pred_and_object.sum() / pred_and_meaningful.sum() if pred_and_meaningful.sum() > 0 else 0.0

        # 5. Calculate IoUs for distraction and dependency
        object_iou = self.calculate_iou(pred_mask, gt_binary)
        artifact_iou = self.calculate_iou(pred_mask, artifact_binary)

        # 6. Distraction Score (객체 대비 artifact 현혹도)
        distraction = artifact_iou / object_iou if object_iou > 0 else 0.0

        # 7. Dependency Ratio (전체 설명력 중 artifact 비율)
        total_explanation = object_iou + artifact_iou
        dependency = artifact_iou / total_explanation if total_explanation > 0 else 0.0

        return {
            'artifact_fpr': float(artifact_fpr),
            'clean_object_precision': float(clean_precision),
            'artifact_contamination': float(contamination),
            'object_purity': float(purity),
            'distraction_score': float(distraction),
            'dependency_ratio': float(dependency),
            'object_iou': float(object_iou),
            'artifact_iou': float(artifact_iou)
        }

    # ========== Comprehensive Evaluation ==========

    def evaluate_single_cam(self, cam, image_name,
                           threshold=0.5,
                           use_topk=False,
                           k_percent=0.10,
                           include_curves=True,
                           include_sweep=True):
        """
        Comprehensive evaluation of a single CAM

        Args:
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold for all metrics
            k_percent: Percentage for top-k method (default 0.10 = 10%)
            include_curves: Include PR/ROC curves (threshold-free metrics)
            include_sweep: Include threshold sweep analysis

        Returns:
            Dictionary with all metrics
        """
        # Load ground truth
        gt_mask = self.load_trimap(image_name)
        artifact_mask = self.load_artifact_mask(image_name)

        results = {'image': image_name}

        # Basic metrics
        basic = self.calculate_basic_metrics(cam, gt_mask, threshold, use_topk, k_percent)
        results.update(basic)

        # Localization metrics
        topk = self.calculate_topk_precision(cam, gt_mask, k_percent=0.15)
        results.update(topk)

        pointing = self.calculate_pointing_game(cam, gt_mask)
        results.update(pointing)

        centroid = self.calculate_centroid_distance(cam, gt_mask)
        results.update(centroid)

        # Boundary metrics
        boundary_f1 = self.calculate_boundary_f1(cam, gt_mask, threshold, use_topk=use_topk, k_percent=k_percent)
        results.update(boundary_f1)

        chamfer = self.calculate_chamfer_distance(cam, gt_mask, threshold, use_topk=use_topk, k_percent=k_percent)
        results.update(chamfer)

        hausdorff = self.calculate_hausdorff_distance(cam, gt_mask, threshold, use_topk=use_topk, k_percent=k_percent)
        results.update(hausdorff)

        # Artifact metrics
        if artifact_mask is not None:
            artifact = self.calculate_artifact_fpr(cam, gt_mask, artifact_mask, threshold, use_topk=use_topk, k_percent=k_percent)
            results.update(artifact)

        outlier = self.calculate_cam_outlier_score(cam)
        results.update(outlier)

        # Curve-based metrics
        if include_curves:
            pr_curve = self.calculate_pr_curve(cam, gt_mask)
            results['ap'] = pr_curve['ap']
            results['pr_curve'] = pr_curve

            roc_curve_data = self.calculate_roc_curve(cam, gt_mask)
            results['auc'] = roc_curve_data['auc']
            results['roc_curve'] = roc_curve_data

        # Threshold sweep
        if include_sweep:
            sweep_df = self.threshold_sweep(cam, gt_mask)
            results['threshold_sweep'] = sweep_df
            results['max_dice'] = sweep_df['dice'].max()
            results['max_iou'] = sweep_df['iou'].max()

        return results

    def evaluate_multiple_methods(self, cams_dict, image_names,
                                  threshold=0.5,
                                  use_topk=False,
                                  k_percent=0.10,
                                  include_curves=True,
                                  include_sweep=False):
        """
        Evaluate multiple CAM methods across multiple images

        Args:
            cams_dict: {method_name: [cam1, cam2, ...]}
            image_names: List of image filenames
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)
            include_curves: Include PR/ROC curves
            include_sweep: Include threshold sweep

        Returns:
            DataFrame with results for all methods and images
        """
        all_results = []

        for idx, image_name in enumerate(image_names):
            print(f"Evaluating {idx+1}/{len(image_names)}: {image_name}")

            for method_name, cams in cams_dict.items():
                try:
                    result = self.evaluate_single_cam(
                        cams[idx],
                        image_name,
                        threshold=threshold,
                        use_topk=use_topk,
                        k_percent=k_percent,
                        include_curves=include_curves,
                        include_sweep=include_sweep
                    )
                    result['method'] = method_name

                    # Store complex data separately
                    if 'pr_curve' in result:
                        result.pop('pr_curve')
                    if 'roc_curve' in result:
                        result.pop('roc_curve')
                    if 'threshold_sweep' in result:
                        result.pop('threshold_sweep')

                    # Remove non-scalar values from main results
                    result = {k: v for k, v in result.items()
                             if not isinstance(v, (tuple, dict, pd.DataFrame))}

                    all_results.append(result)

                except Exception as e:
                    print(f"  Warning: Failed to evaluate {method_name}: {e}")
                    all_results.append({
                        'image': image_name,
                        'method': method_name,
                        'error': str(e)
                    })

        df = pd.DataFrame(all_results)
        return df

    def evaluate_object_localization(self, cams_dict, image_names,
                                     threshold=0.5,
                                     use_topk=False,
                                     k_percent=0.10):
        """
        Evaluate CAM methods for OBJECT LOCALIZATION (vs GT annotation masks)

        Measures how well CAMs localize the target object.

        Args:
            cams_dict: {method_name: [cam1, cam2, ...]}
            image_names: List of image filenames
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            DataFrame with object localization metrics for all methods and images
        """
        all_results = []

        for idx, image_name in enumerate(image_names):
            gt_mask = self.load_trimap(image_name)

            for method_name, cams in cams_dict.items():
                try:
                    cam = cams[idx]

                    # Convert CAM to binary mask
                    pred_mask, cam_resized = self.cam_to_binary_mask(
                        cam, gt_mask, threshold, use_topk, k_percent
                    )

                    # Calculate all object localization metrics
                    result = {'image': image_name, 'method': method_name}

                    # 1. Basic overlap metrics (IoU, Dice, F1, Precision, Recall)
                    basic = self.calculate_basic_metrics(cam, gt_mask, threshold, use_topk, k_percent)
                    result.update(basic)

                    # 2. Boundary metrics
                    boundary_f1 = self.calculate_boundary_f1(cam, gt_mask, threshold, use_topk, k_percent)
                    result.update(boundary_f1)

                    chamfer = self.calculate_chamfer_distance(cam, gt_mask, threshold, use_topk, k_percent)
                    result.update(chamfer)

                    hausdorff = self.calculate_hausdorff_distance(cam, gt_mask, threshold, use_topk, k_percent)
                    result.update(hausdorff)

                    # 3. Localization-specific metrics
                    pointing = self.calculate_pointing_game(cam, gt_mask)
                    result.update(pointing)

                    centroid = self.calculate_centroid_distance(cam, gt_mask)
                    result.update(centroid)

                    topk_prec = self.calculate_topk_precision(cam, gt_mask, k_percent=0.15)
                    result.update(topk_prec)

                    # 4. Threshold-free metrics
                    pr_curve = self.calculate_pr_curve(cam, gt_mask)
                    result['ap'] = pr_curve['ap']

                    roc_curve_data = self.calculate_roc_curve(cam, gt_mask)
                    result['auc'] = roc_curve_data['auc']

                    all_results.append(result)

                except Exception as e:
                    print(f"  Warning: Failed to evaluate {method_name} on {image_name}: {e}")
                    all_results.append({
                        'image': image_name,
                        'method': method_name,
                        'error': str(e)
                    })

        df = pd.DataFrame(all_results)
        return df

    def evaluate_artifact_detection(self, cams_dict, image_names,
                                    threshold=0.5,
                                    use_topk=False,
                                    k_percent=0.10):
        """
        Evaluate CAM methods for ARTIFACT DETECTION (vs artifact box masks)

        Measures how well CAMs avoid/detect artifacts (lower activation on artifacts is better).

        Args:
            cams_dict: {method_name: [cam1, cam2, ...]}
            image_names: List of image filenames
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            DataFrame with artifact detection metrics for all methods and images
        """
        all_results = []

        for idx, image_name in enumerate(image_names):
            artifact_mask = self.load_artifact_mask(image_name)

            if artifact_mask is None:
                # No artifact mask available for this image
                continue

            gt_mask = self.load_trimap(image_name)  # For shape reference

            for method_name, cams in cams_dict.items():
                try:
                    cam = cams[idx]

                    # Convert CAM to binary mask
                    pred_mask, cam_resized = self.cam_to_binary_mask(
                        cam, gt_mask, threshold, use_topk, k_percent
                    )

                    # Resize artifact mask to match CAM
                    if artifact_mask.shape != cam_resized.shape:
                        from scipy.ndimage import zoom
                        scale_factors = (cam_resized.shape[0] / artifact_mask.shape[0],
                                       cam_resized.shape[1] / artifact_mask.shape[1])
                        artifact_mask_resized = zoom(artifact_mask.astype(np.float32), scale_factors, order=0).astype(np.uint8)
                    else:
                        artifact_mask_resized = artifact_mask

                    # Calculate artifact-specific metrics
                    result = {'image': image_name, 'method': method_name}

                    # 1. Overlap metrics WITHIN artifact regions
                    # Here we treat artifact_mask as "ground truth" to measure how much CAM activates on artifacts
                    basic = self.calculate_basic_metrics(cam, artifact_mask, threshold, use_topk, k_percent)
                    # Rename metrics to indicate they're artifact-based
                    result['artifact_iou'] = basic['iou']
                    result['artifact_dice'] = basic['dice']
                    result['artifact_precision'] = basic['precision']  # What % of predicted pixels are in artifact
                    result['artifact_recall'] = basic['recall']  # What % of artifact is covered by prediction

                    # 2. Artifact FPR (false positive rate in artifact regions)
                    artifact_fpr_result = self.calculate_artifact_fpr(cam, gt_mask, artifact_mask, threshold, use_topk, k_percent)
                    result.update(artifact_fpr_result)

                    # 3. Boundary metrics within artifact regions
                    boundary_f1 = self.calculate_boundary_f1(cam, artifact_mask, threshold, use_topk, k_percent)
                    result['artifact_boundary_f1'] = boundary_f1['boundary_f1']

                    chamfer = self.calculate_chamfer_distance(cam, artifact_mask, threshold, use_topk, k_percent)
                    result['artifact_chamfer'] = chamfer['chamfer_distance']

                    hausdorff = self.calculate_hausdorff_distance(cam, artifact_mask, threshold, use_topk, k_percent)
                    result['artifact_hausdorff'] = hausdorff['hausdorff_distance']

                    all_results.append(result)

                except Exception as e:
                    print(f"  Warning: Failed to evaluate {method_name} on {image_name}: {e}")
                    all_results.append({
                        'image': image_name,
                        'method': method_name,
                        'error': str(e)
                    })

        df = pd.DataFrame(all_results)
        return df

    def evaluate_cross_analysis(self, cams_dict, image_names,
                                threshold=0.5,
                                use_topk=False,
                                k_percent=0.10):
        """
        Evaluate cross-analysis metrics between object and artifact

        Args:
            cams_dict: {method_name: [cam1, cam2, ...]}
            image_names: List of image filenames
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            DataFrame with cross-analysis metrics for all methods and images
        """
        all_results = []

        for idx, image_name in enumerate(image_names):
            gt_mask = self.load_trimap(image_name)
            artifact_mask = self.load_artifact_mask(image_name)

            if artifact_mask is None:
                # No artifact mask available for this image
                continue

            for method_name, cams in cams_dict.items():
                try:
                    cam = cams[idx]

                    # Calculate cross-analysis metrics
                    result = {'image': image_name, 'method': method_name}

                    cross_metrics = self.calculate_cross_analysis_metrics(
                        cam, gt_mask, artifact_mask, threshold, use_topk, k_percent
                    )
                    result.update(cross_metrics)

                    all_results.append(result)

                except Exception as e:
                    print(f"  Warning: Failed to evaluate {method_name} on {image_name}: {e}")
                    all_results.append({
                        'image': image_name,
                        'method': method_name,
                        'error': str(e)
                    })

        df = pd.DataFrame(all_results)
        return df

    def create_summary_report(self, results_df, save_dir=None):
        """
        Create comprehensive summary report with statistics

        Args:
            results_df: DataFrame from evaluate_multiple_methods
            save_dir: Optional directory to save reports
        """
        methods = results_df['method'].unique()

        # Summary statistics
        summary_data = []
        for method in methods:
            method_df = results_df[results_df['method'] == method]

            summary = {'Method': method}

            # Calculate statistics for each metric
            metric_cols = [col for col in method_df.columns
                          if col not in ['image', 'method', 'error']]

            for col in metric_cols:
                values = pd.to_numeric(method_df[col], errors='coerce').dropna()
                if len(values) > 0:
                    summary[f'{col}_mean'] = values.mean()
                    summary[f'{col}_std'] = values.std()
                    summary[f'{col}_median'] = values.median()

            summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)

        # Sort by mean IoU
        if 'iou_mean' in summary_df.columns:
            summary_df = summary_df.sort_values('iou_mean', ascending=False)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save full results
            results_df.to_csv(save_dir / 'detailed_results.csv', index=False)

            # Save summary
            summary_df.to_csv(save_dir / 'summary_report.csv', index=False)

            print(f"Reports saved to: {save_dir}")

        return summary_df


def compare_with_baseline(results_df, baseline_method):
    """
    Compare all methods against a baseline

    Returns DataFrame with improvement statistics
    """
    methods = results_df['method'].unique()
    baseline_df = results_df[results_df['method'] == baseline_method]

    comparison_data = []

    for method in methods:
        if method == baseline_method:
            continue

        method_df = results_df[results_df['method'] == method]

        # Check which columns are available
        available_metrics = []
        for metric in ['iou', 'dice', 'f1']:
            if metric in method_df.columns:
                available_metrics.append(metric)

        if not available_metrics:
            continue

        # Merge on image name with only available columns
        merge_cols = ['image'] + available_metrics
        merged = pd.merge(
            method_df[merge_cols],
            baseline_df[merge_cols],
            on='image',
            suffixes=('', '_baseline')
        )

        comparison = {'Method': method}

        for metric in available_metrics:
            if metric in merged.columns and f'{metric}_baseline' in merged.columns:
                improvements = merged[metric] - merged[f'{metric}_baseline']
                comparison[f'{metric}_mean'] = merged[metric].mean()
                comparison[f'{metric}_improvement'] = improvements.mean()
                comparison[f'{metric}_win_rate'] = (improvements > 0).mean() * 100

        comparison_data.append(comparison)

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


# ========== Simple IoU-only Evaluator ==========

class IoUEvaluator:
    """Lightweight IoU-only evaluator for fast evaluation"""

    def __init__(self, annotation_dir, artifact_masks_dir=None):
        """
        Initialize IoU Evaluator

        Args:
            annotation_dir: Path to trimap annotations directory
            artifact_masks_dir: Path to artifact masks directory (optional)
        """
        self.annotation_dir = Path(annotation_dir)
        self.artifact_masks_dir = Path(artifact_masks_dir) if artifact_masks_dir else None

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
        binary_mask = (trimap == 1).astype(np.uint8)

        return binary_mask

    def cam_to_binary_mask(self, cam, threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Convert CAM to binary mask using threshold or top-k method

        Args:
            cam: CAM tensor or numpy array (H, W)
            threshold: Threshold value (0-1), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

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

        # Apply threshold or top-k
        if use_topk:
            flat_cam = cam_norm.flatten()
            threshold_val = np.percentile(flat_cam, (1 - k_percent) * 100)
            binary_mask = (cam_norm >= threshold_val).astype(np.uint8)
        else:
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

    def evaluate_cam(self, cam, image_name, threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Evaluate single CAM against ground truth

        Args:
            cam: CAM tensor or numpy array
            image_name: Image filename
            threshold: Threshold for binarization (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            IoU score
        """
        # Load ground truth
        gt_mask = self.load_trimap(image_name)

        # Resize CAM to match ground truth size if needed
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()

        # Ensure CAM is 2D (squeeze out any extra dimensions)
        while cam.ndim > 2:
            cam = cam.squeeze()

        # If still not 2D after squeeze, take first/last channel
        if cam.ndim > 2:
            if cam.shape[0] == 1:
                cam = cam[0]
            elif cam.shape[-1] == 1:
                cam = cam[..., 0]
            else:
                # Take first channel as fallback
                cam = cam[0] if cam.shape[0] < cam.shape[-1] else cam[..., 0]

        if cam.shape != gt_mask.shape:
            from scipy.ndimage import zoom
            scale_factors = (gt_mask.shape[0] / cam.shape[0],
                           gt_mask.shape[1] / cam.shape[1])
            cam = zoom(cam, scale_factors, order=1)

        # Convert to binary mask
        pred_mask = self.cam_to_binary_mask(cam, threshold, use_topk, k_percent)

        # Calculate IoU
        iou = self.calculate_iou(pred_mask, gt_mask)

        return iou

    def evaluate_multiple_cams(self, cams_dict, image_names, threshold=0.5, use_topk=False, k_percent=0.10):
        """
        Evaluate multiple CAM methods

        Args:
            cams_dict: Dictionary of {method_name: list_of_cams}
            image_names: List of image filenames
            threshold: Threshold for binarization (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            DataFrame with IoU results
        """
        results = []

        for idx, image_name in enumerate(image_names):
            row = {'image': image_name}

            for method_name, cams in cams_dict.items():
                try:
                    iou = self.evaluate_cam(cams[idx], image_name, threshold, use_topk, k_percent)
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

        # Only sort if dataframe is not empty and has the column
        if not summary_df.empty and 'Mean IoU' in summary_df.columns:
            summary_df = summary_df.sort_values('Mean IoU', ascending=False)

        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"Summary table saved to: {save_path}")

        return summary_df

    def load_artifact_mask(self, image_name):
        """
        Load artifact mask if available

        Args:
            image_name: Image filename

        Returns:
            Binary artifact mask or None if not found
        """
        # Try to load from artifact_masks_dir if it exists
        if not hasattr(self, 'artifact_masks_dir') or self.artifact_masks_dir is None:
            return None

        # Convert image name to artifact mask name (add _artifact suffix)
        mask_name = image_name.replace('.jpg', '_artifact.png').replace('.jpeg', '_artifact.png')
        artifact_path = Path(self.artifact_masks_dir) / mask_name

        if not artifact_path.exists():
            return None

        artifact_mask = np.array(Image.open(artifact_path).convert('L'))
        artifact_mask = (artifact_mask > 0).astype(np.uint8)

        return artifact_mask

    def evaluate_object_localization(self, cams_dict, image_names,
                                     threshold=0.5,
                                     use_topk=False,
                                     k_percent=0.10):
        """
        Evaluate CAM methods for OBJECT LOCALIZATION (vs GT annotation masks)

        Simplified version for IoUEvaluator that only computes IoU.

        Args:
            cams_dict: {method_name: [cam1, cam2, ...]}
            image_names: List of image filenames
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            DataFrame with IoU results for object localization
        """
        # This is essentially the same as evaluate_multiple_cams
        return self.evaluate_multiple_cams(cams_dict, image_names, threshold, use_topk, k_percent)

    def evaluate_artifact_detection(self, cams_dict, image_names,
                                    threshold=0.5,
                                    use_topk=False,
                                    k_percent=0.10):
        """
        Evaluate CAM methods for ARTIFACT DETECTION (vs artifact box masks)

        Measures IoU with artifact regions.

        Args:
            cams_dict: {method_name: [cam1, cam2, ...]}
            image_names: List of image filenames
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            DataFrame with IoU results for artifact detection
        """
        results = []

        for idx, image_name in enumerate(image_names):
            artifact_mask = self.load_artifact_mask(image_name)

            if artifact_mask is None:
                # No artifact mask for this image
                continue

            row = {'image': image_name}

            for method_name, cams in cams_dict.items():
                try:
                    cam = cams[idx]

                    # Load GT mask for shape reference
                    gt_mask = self.load_trimap(image_name)

                    # Process CAM
                    if isinstance(cam, torch.Tensor):
                        cam = cam.cpu().numpy()

                    # Ensure CAM is 2D
                    while cam.ndim > 2:
                        cam = cam.squeeze()

                    if cam.ndim > 2:
                        if cam.shape[0] == 1:
                            cam = cam[0]
                        elif cam.shape[-1] == 1:
                            cam = cam[..., 0]
                        else:
                            cam = cam[0] if cam.shape[0] < cam.shape[-1] else cam[..., 0]

                    # Resize CAM to match GT mask size
                    if cam.shape != gt_mask.shape:
                        from scipy.ndimage import zoom
                        scale_factors = (gt_mask.shape[0] / cam.shape[0],
                                       gt_mask.shape[1] / cam.shape[1])
                        cam = zoom(cam, scale_factors, order=1)

                    # Resize artifact mask to match
                    if artifact_mask.shape != gt_mask.shape:
                        from scipy.ndimage import zoom
                        scale_factors = (gt_mask.shape[0] / artifact_mask.shape[0],
                                       gt_mask.shape[1] / artifact_mask.shape[1])
                        artifact_mask = zoom(artifact_mask.astype(np.float32), scale_factors, order=0).astype(np.uint8)

                    # Convert CAM to binary mask
                    pred_mask = self.cam_to_binary_mask(cam, threshold, use_topk, k_percent)

                    # Calculate IoU with artifact mask
                    artifact_iou = self.calculate_iou(pred_mask, artifact_mask)
                    row[method_name] = artifact_iou

                except Exception as e:
                    print(f"Warning: Failed to evaluate {method_name} for {image_name}: {e}")
                    row[method_name] = np.nan

            results.append(row)

        df = pd.DataFrame(results)
        return df

    def evaluate_cross_analysis(self, cams_dict, image_names,
                                threshold=0.5,
                                use_topk=False,
                                k_percent=0.10):
        """
        Evaluate cross-analysis metrics between object and artifact
        (Simplified version for IoUEvaluator - returns empty DataFrame)

        Args:
            cams_dict: {method_name: [cam1, cam2, ...]}
            image_names: List of image filenames
            threshold: Fixed threshold (default 0.5), ignored if use_topk=True
            use_topk: If True, use top-k% instead of threshold
            k_percent: Percentage for top-k method (default 0.10 = 10%)

        Returns:
            Empty DataFrame (IoUEvaluator doesn't support cross-analysis)
        """
        # IoUEvaluator is lightweight and doesn't include cross-analysis
        # Return empty DataFrame for compatibility
        print("Note: IoUEvaluator does not support cross-analysis. Use AdvancedCAMEvaluator for full metrics.")
        return pd.DataFrame()


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

    # Only sort if dataframe is not empty and has the column
    if not comparison_df.empty and 'Mean IoU' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('Mean IoU', ascending=False)

    return comparison_df


# Alias for backward compatibility
CAMEvaluator = AdvancedCAMEvaluator