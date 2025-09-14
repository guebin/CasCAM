"""
Model training module for CasCAM
"""

import torch
import warnings
from fastai.vision.all import *
from logger import TrainingLogger
from visualizer import _get_label_from_filename


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
    def train_with_early_stopping(lrnr, max_epochs=10, patience=3):
        """Train model with early stopping using fine_tune and return training logger"""
        logger = TrainingLogger()
        best_valid_loss = float('inf')
        patience_counter = 0
        
        # Initial fine_tune
        lrnr.fine_tune(1)
        
        if lrnr.recorder.values is None or len(lrnr.recorder.values) == 0:
            logger.log_epoch(1, float('inf'), float('inf'), 1.0)
            return logger
            
        # Get metrics from initial training
        train_loss = float(lrnr.recorder.values[-1][0])
        valid_loss = float(lrnr.recorder.values[-1][1]) 
        error_rate = float(lrnr.recorder.values[-1][2])
        logger.log_epoch(1, train_loss, valid_loss, error_rate)
        best_valid_loss = valid_loss
        
        # Continue training with early stopping
        for epoch in range(1, max_epochs):
            # Continue fine-tuning
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
                print(f"Early stopping counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return logger