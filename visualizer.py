"""
Visualization module for CasCAM
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class CasCAMVisualizer:
    """Class for creating CasCAM visualizations"""
    
    @staticmethod
    def smooth_cam(cam, sigma=7.0):
        """Apply Gaussian smoothing to CAM for better visualization"""
        if hasattr(cam, 'detach'):  # PyTorch tensor
            cam_np = cam.detach().cpu().numpy()
        else:
            cam_np = np.array(cam)
        
        # Apply Gaussian filter for smoothing
        smoothed = ndimage.gaussian_filter(cam_np, sigma=sigma)
        
        # Convert back to original type
        if hasattr(cam, 'detach'):  # Return as tensor
            import torch
            return torch.tensor(smoothed, dtype=cam.dtype, device=cam.device)
        else:
            return smoothed
    
    @staticmethod
    def make_figure(img, allcams, methods):
        """Create comparison figure with all CAM methods"""
        fig, axs = plt.subplots(3, 4)
        
        # Set titles first
        axs[0][0].set_title("Original Image")
        axs[0][1].set_title("CasCAM (proposed)")
        axs[0][2].set_title("CAM")
        
        for ax, method_name in zip(axs.flatten()[3:], methods[2:] if len(methods) > 2 else []):
            ax.set_title(f"{method_name}")
        
        # Show original image on first subplot only
        img.show(ax=axs[0][0])
        
        # Show image + CAM overlays for all other subplots
        for i, (ax, cam) in enumerate(zip(axs.flatten()[1:], allcams)):
            # Show original image as background
            img.show(ax=ax)

            # Apply smoothing only to CasCAM (first CAM)
            if i == 0:  # CasCAM (proposed)
                smoothed_cam = CasCAMVisualizer.smooth_cam(cam, sigma=7.0)
                display_cam = smoothed_cam.squeeze()
            else:  # Other CAMs - no smoothing
                display_cam = cam.squeeze()

            # Overlay the CAM heatmap
            if hasattr(display_cam, 'detach'):  # PyTorch tensor
                display_cam = display_cam.detach().cpu().numpy()

            # Get the extent from the background image if it exists, otherwise use default
            try:
                extent = ax.images[0].get_extent() if ax.images else None
            except (IndexError, AttributeError):
                extent = None

            # Overlay CAM with proper alpha blending using magma colormap
            ax.imshow(display_cam, alpha=0.8, cmap='magma', extent=extent)
            ax.axis('off')

        fig.set_figwidth(10)
        fig.set_figheight(7.5)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig, save_path):
        """Save figure to specified path"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)


def _get_label_from_filename(filepath):
    """Helper function to get label from filename - must be at module level for pickling"""
    filename = str(filepath).split('/')[-1]
    return "cat" if filename[0].isupper() else "dog"