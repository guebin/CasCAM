"""
Training logger module for CasCAM
"""

import os
import json


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