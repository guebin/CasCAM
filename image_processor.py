"""
Image processing module for CasCAM
"""

import os
import numpy as np
import torch
import torchvision


class ImageProcessor:
    """Class for processing images using CAM-based weighting"""
    
    @staticmethod
    def apply_cam_weighting(img, cam, theta):
        """Apply CAM-based weighting to image"""
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # Ensure cam is a tensor
        if isinstance(cam, np.ndarray):
            cam = torch.tensor(cam, dtype=torch.float32)
        
        # Now both img and cam should be 224x224, so no resize needed
        # But keep resize logic as fallback for safety
        if cam.shape != img_tensor.shape[1:]:  # Compare with H, W dimensions
            # Ensure cam is 4D for interpolation: (N, C, H, W)
            if len(cam.shape) == 2:  # (H, W)
                cam_4d = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif len(cam.shape) == 3:  # (1, H, W) or (C, H, W)
                cam_4d = cam.unsqueeze(0) if cam.shape[0] != 1 else cam.unsqueeze(1)
            else:
                cam_4d = cam
                
            cam_resized = torch.nn.functional.interpolate(
                cam_4d,
                size=img_tensor.shape[1:],  # (H, W)
                mode='bilinear',
                align_corners=False
            )
            
            # Remove extra dimensions to get back to (H, W)
            cam_resized = cam_resized.squeeze(0).squeeze(0)
        else:
            cam_resized = cam
        
        weight = torch.exp(-theta * cam_resized)
        res_img_tensor = img_tensor * weight

        # Apply min-max normalization to 0-1 range
        min_val = res_img_tensor.min()
        max_val = res_img_tensor.max()
        if max_val > min_val:
            res_img_tensor = (res_img_tensor - min_val) / (max_val - min_val)

        res_img = torchvision.transforms.ToPILImage()(res_img_tensor)
        return res_img
    
    @staticmethod
    def save_processed_image(img, save_path):
        """Save processed image to specified path"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)