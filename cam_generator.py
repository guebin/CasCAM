"""
CAM generation module for CasCAM
"""

import torch
import numpy as np
from fastai.vision.all import *
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline


class CAMGenerator:
    """Class for generating original CAM using model weights"""
    
    @staticmethod
    def original_cam(model, input_tensor, label):
        """Generate original CAM using model weights (without thresholding)"""
        cam = torch.einsum('ocij,kc -> okij', model[0](input_tensor), model[1][2].weight).data.cpu()
        cam = cam[0,0,:,:] if label == 0 else cam[0,1,:,:]
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


class CasCAM:
    """CasCAM class that maintains state and provides processing methods"""
    
    def __init__(self, source_cam, threshold_method=None, threshold_params=None):
        """Initialize CasCAM with source CAM"""
        self.source_cam = source_cam
        self.processed_cam = None
        self.threshold_method = threshold_method
        self.threshold_params = threshold_params or {}
        
        # Apply initial processing if method specified
        if threshold_method is not None:
            self.apply_threshold(threshold_method, threshold_params)
    
    def apply_threshold(self, method=None, params=None):
        """Apply specified thresholding method to source CAM"""
        self.threshold_method = method
        self.threshold_params = params or {}

        if method is None:
            self.processed_cam = self.source_cam
        elif method == 'top_k':
            k = self.threshold_params.get('k', 20)
            self.processed_cam = self._top_k_threshold(self.source_cam, k)
        elif method == 'ebayesthresh':
            eb_method = self.threshold_params.get('method', 'sure')
            prior = self.threshold_params.get('prior', 'laplace')
            a = self.threshold_params.get('a', 0.5)
            self.processed_cam = self._ebayesthresh_soft_threshold(self.source_cam, eb_method, prior, a)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        return self.processed_cam
    
    def _ebayesthresh_soft_threshold(self, cam, method='sure', prior='laplace', a=0.5):
        """EBayesThresh-style soft thresholding"""
        cam_flat = cam.view(-1).detach().cpu().numpy()
        sigma = np.median(np.abs(cam_flat - np.median(cam_flat))) / 0.6745
        
        # Simplified threshold calculation
        n = len(cam_flat)
        if method == 'sure':
            # Simplified SURE: use percentile-based approximation
            sorted_vals = np.sort(np.abs(cam_flat))
            threshold = sorted_vals[int(0.9 * len(sorted_vals))]  # Top 10%
        elif method == 'bayes' and prior == 'laplace':
            threshold = sigma**2 / a
        else:  # bayes+cauchy, bic, riskshrink
            threshold = sigma * np.sqrt(2 * np.log(n))
        
        # Apply soft thresholding
        cam_np = cam.detach().cpu().numpy()
        thresholded = np.sign(cam_np) * np.maximum(np.abs(cam_np) - threshold, 0)
        
        return torch.tensor(thresholded, dtype=cam.dtype, device=cam.device)
    
    def _top_k_threshold(self, cam, k=20):
        """Keep only top-k% values, set rest to 0"""
        cam_flat = cam.view(-1)
        k_val = max(1, int(len(cam_flat) * k / 100))  # Convert percentage to actual count
        
        # Get top-k values
        top_k_vals, top_k_indices = torch.topk(cam_flat, k_val)
        min_top_k = top_k_vals[-1]  # Minimum value among top-k
        
        # Create thresholded CAM: keep values >= min_top_k, set others to 0
        thresholded_cam = torch.where(cam >= min_top_k, cam, torch.zeros_like(cam))
        
        return thresholded_cam
    
    def get_statistics(self):
        """Calculate various statistics for the processed CAM"""
        if self.processed_cam is None:
            cam = self.source_cam
        else:
            cam = self.processed_cam
        
        cam_np = cam.detach().cpu().numpy()
        
        # Basic statistics
        stats = {
            'min': float(np.min(cam_np)),
            'max': float(np.max(cam_np)),
            'mean': float(np.mean(cam_np)),
            'std': float(np.std(cam_np)),
            'median': float(np.median(cam_np)),
        }
        
        # Sharpness (gradient magnitude)
        if len(cam_np.shape) == 2:
            grad_x = np.gradient(cam_np, axis=1)
            grad_y = np.gradient(cam_np, axis=0)
            sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            stats['sharpness'] = float(sharpness)
        
        # Sparsity (percentage of near-zero values)
        threshold = 0.01 * np.max(np.abs(cam_np))
        sparsity = np.mean(np.abs(cam_np) < threshold)
        stats['sparsity'] = float(sparsity)
        
        return stats
    
    def get_weighted_image(self, img, method='exp', alpha=1.0):
        """Apply CAM weighting to image using different methods"""
        if self.processed_cam is None:
            cam = self.source_cam
        else:
            cam = self.processed_cam
        
        # Convert image to tensor if needed
        if hasattr(img, 'convert'):  # PIL Image
            img_tensor = torch.tensor(np.array(img.convert('RGB')), dtype=torch.float32) / 255.0
        else:
            img_tensor = img
        
        # Ensure CAM is same size as image
        if cam.shape != img_tensor.shape[:2]:
            cam_resized = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=img_tensor.shape[:2],
                mode='bilinear',
                align_corners=False
            ).squeeze()
        else:
            cam_resized = cam
        
        # Normalize CAM to [0, 1]
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        
        if method == 'exp':
            # Exponential weighting: img * exp(alpha * cam)
            weight = torch.exp(alpha * cam_norm)
            if len(img_tensor.shape) == 3:
                weight = weight.unsqueeze(-1).expand_as(img_tensor)
            weighted_img = img_tensor * weight
        elif method == 'linear':
            # Linear weighting: img * (1 + alpha * cam)
            weight = 1 + alpha * cam_norm
            if len(img_tensor.shape) == 3:
                weight = weight.unsqueeze(-1).expand_as(img_tensor)
            weighted_img = img_tensor * weight
        elif method == 'mask':
            # Binary mask: img * (cam > threshold)
            threshold = 0.5  # Could be parameterized
            mask = (cam_norm > threshold).float()
            if len(img_tensor.shape) == 3:
                mask = mask.unsqueeze(-1).expand_as(img_tensor)
            weighted_img = img_tensor * mask
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Clamp to valid range
        weighted_img = torch.clamp(weighted_img, 0, 1)
        
        return weighted_img
    
    @classmethod
    def from_original_cam(cls, original_cam, threshold_method=None, threshold_params=None):
        """Create CasCAM from original CAM (for backward compatibility)"""
        return cls(original_cam, threshold_method, threshold_params)


class OtherCAMGenerator:
    """Class for generating other CAM methods (GradCAM, HiResCAM, etc.)"""
    
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
        
        # Generate original CAM
        original_cam = CAMGenerator.original_cam(model=model, input_tensor=img_norm, label=label)

        # Start with empty list for other CAM methods only
        allcams = []

        # Generate other CAM methods
        for method in methods:
            try:
                other_cam = method(model=model, target_layers=[model[0][-1]])(input_tensor=img_norm, targets=None)
                
                # Debug: Check CAM values
                cam_tensor = torch.tensor(other_cam)
                print(f"\n=== {method.__name__ if hasattr(method, '__name__') else str(method)} ===")
                print(f"Shape: {cam_tensor.shape}")
                print(f"Min: {cam_tensor.min().item():.6f}")
                print(f"Max: {cam_tensor.max().item():.6f}")
                print(f"Mean: {cam_tensor.mean().item():.6f}")
                print(f"Std: {cam_tensor.std().item():.6f}")
                print(f"Non-zero count: {(cam_tensor != 0).sum().item()}/{cam_tensor.numel()}")
                
                # Check for problematic values
                if cam_tensor.max().item() - cam_tensor.min().item() < 1e-8:
                    print(f"WARNING: {method.__name__ if hasattr(method, '__name__') else str(method)} has extremely low dynamic range!")
                if torch.isnan(cam_tensor).any():
                    print(f"WARNING: {method.__name__ if hasattr(method, '__name__') else str(method)} contains NaN values!")
                if (cam_tensor == 0).all():
                    print(f"WARNING: {method.__name__ if hasattr(method, '__name__') else str(method)} is all zeros!")
                
                allcams.append(cam_tensor)
            except Exception as e:
                print(f"Warning: Failed to generate {method}: {e}")
                continue
                
        return img, allcams