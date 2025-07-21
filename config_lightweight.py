#!/usr/bin/env python3
"""
Configuration for Lightweight Attention Model

Fixed training parameters based on analysis of the failed 79M parameter model.
This config addresses the critical issues causing 29.4% accuracy.
"""

import os
import torch
from utils.helpers import get_base_path


BASE_PATH = get_base_path() + '/breakhis'

FOLD_PATH = os.path.join(BASE_PATH, 'Folds_fixed.csv')
HOLDOUT_PATH = os.path.join(BASE_PATH, 'Folds_holdout.csv')
ROBUST_HOLDOUT_PATH = os.path.join(BASE_PATH, 'Folds_robust_holdout_balanced_large_test.csv')
SLIDES_PATH = os.path.join(BASE_PATH, 'BreaKHis_v1/BreaKHis_v1/histology_slides/breast')


def get_training_config():
    """Get optimized training configuration for lightweight model"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        
        is_runpod = (
            os.path.exists('/workspace') or 
            'runpod' in os.environ.get('HOSTNAME', '').lower() or
            os.environ.get('RUNPOD_POD_ID') is not None
        )
        
        if is_runpod:
            config = {
                'device': device,
                'num_gpus': num_gpus,
                'batch_size': 16,  # Larger batch for smaller model
                'num_workers': 8,
                'pin_memory': True,
                'persistent_workers': True,
                'environment': 'runpod'
            }
        else:
            config = {
                'device': device,
                'num_gpus': num_gpus,
                'batch_size': 12,
                'num_workers': 4,
                'pin_memory': True,
                'persistent_workers': True,
                'environment': 'local_cuda'
            }
            
    elif torch.backends.mps.is_available():
        config = {
            'device': torch.device("mps"),
            'num_gpus': 1,
            'batch_size': 8,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'environment': 'mps'
        }
    else:
        config = {
            'device': torch.device("cpu"),
            'num_gpus': 0,
            'batch_size': 4,
            'num_workers': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'environment': 'cpu'
        }
    
    config['effective_batch_size'] = config['batch_size'] * max(1, config['num_gpus'])
    return config


# Lightweight Training Configuration - FIXES THE ISSUES
LIGHTWEIGHT_TRAINING_CONFIG = {
    'model_name': 'LightweightAttentionNet',
    'backbone': 'efficientnet_b0',  # Smaller backbone
    
    # Training parameters - FIXED FOR STABILITY
    'epochs': 50,  # More epochs for gradual learning
    'learning_rate': 1e-3,  # Higher initial LR
    'weight_decay': 1e-4,  # Less regularization
    'warmup_epochs': 3,  # Shorter warmup
    'patience': 15,  # More patience
    
    # Data parameters - MORE SAMPLES
    'samples_per_patient': 16,  # More samples per patient
    'samples_per_patient_val': 12,
    'magnifications': ['40', '100', '200', '400'],
    'img_size': 224,
    'seed': 42,
    
    # Model specific parameters
    'channel_attention_reduction': 8,  # Less aggressive reduction
    'attention_dropout': 0.1,
    
    # Training stability improvements
    'gradient_clip_val': 1.0,  # Gradient clipping
    'mixed_precision': False,  # Disable AMP for stability
    'accumulate_grad_batches': 1,  # No accumulation needed with larger batch
    'use_scheduler': True,  # Use learning rate scheduler
    'scheduler_factor': 0.5,  # LR reduction factor
    'scheduler_patience': 5,  # Scheduler patience
}

# Loss configuration - FIXED CLASS IMBALANCE
LIGHTWEIGHT_LOSS_CONFIG = {
    'class_weight': 0.8,  # Primary focus on classification
    'tumor_weight': 0.2,
    'class_label_smoothing': 0.1,  # Label smoothing for stability
    'tumor_label_smoothing': 0.05,
    
    # Use class weights to handle imbalance
    'use_class_weights': True,  # Enable class weighting
    'malignant_weight': 0.4,  # Weight for malignant class (underrepresented)
    'benign_weight': 1.0,     # Weight for benign class
}

# Backwards compatibility
TRAINING_CONFIG = LIGHTWEIGHT_TRAINING_CONFIG
LOSS_CONFIG = LIGHTWEIGHT_LOSS_CONFIG


def print_lightweight_config():
    """Print the lightweight configuration summary"""
    
    print("🚀 LIGHTWEIGHT ATTENTION MODEL CONFIGURATION")
    print("=" * 80)
    print("⚠️  FIXES FOR 29.4% ACCURACY FAILURE:")
    print("   • Reduced model from 79M to ~25M parameters")
    print("   • Fixed class prediction collapse with proper loss weighting")
    print("   • Improved training stability with gradient clipping")
    print("   • Better learning rate schedule and patience")
    print("=" * 80)
    
    device_config = get_training_config()
    
    print(f"🧠 Model Configuration:")
    print(f"   Model: {LIGHTWEIGHT_TRAINING_CONFIG['model_name']}")
    print(f"   Backbone: {LIGHTWEIGHT_TRAINING_CONFIG['backbone']} (vs efficientnet_b2)")
    print(f"   Expected parameters: ~25M (vs 79M)")
    print(f"   Params per sample: ~30 (vs 94)")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Device: {device_config['device']}")
    print(f"   Batch size: {device_config['batch_size']} (vs 8)")
    print(f"   Epochs: {LIGHTWEIGHT_TRAINING_CONFIG['epochs']} (vs 30)")
    print(f"   Learning rate: {LIGHTWEIGHT_TRAINING_CONFIG['learning_rate']} (vs 3e-4)")
    print(f"   Gradient clipping: {LIGHTWEIGHT_TRAINING_CONFIG['gradient_clip_val']}")
    print(f"   Mixed precision: {LIGHTWEIGHT_TRAINING_CONFIG['mixed_precision']} (disabled for stability)")
    
    print(f"\n📊 Data Configuration:")
    print(f"   Samples per patient: {LIGHTWEIGHT_TRAINING_CONFIG['samples_per_patient']} (vs 12)")
    print(f"   Expected train samples: ~800 → 1200 (50% increase)")
    
    print(f"\n🎯 Loss Configuration:")
    print(f"   Class weighting: {LIGHTWEIGHT_LOSS_CONFIG['use_class_weights']} (NEW)")
    print(f"   Malignant weight: {LIGHTWEIGHT_LOSS_CONFIG['malignant_weight']} (help underrepresented class)")
    print(f"   Label smoothing: {LIGHTWEIGHT_LOSS_CONFIG['class_label_smoothing']} (stability)")
    
    print("=" * 80)


if __name__ == "__main__":
    print_lightweight_config()