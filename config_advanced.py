#!/usr/bin/env python3
"""
Configuration for Advanced Attention-Guided Multi-Magnification Training

This config uses the AdvancedMultiMagAttentionNet with state-of-the-art
attention mechanisms for journal publication quality results.
"""

import os
import torch
from utils.helpers import get_base_path


BASE_PATH = get_base_path() + '/breakhis'

FOLD_PATH = os.path.join(BASE_PATH, 'Folds_fixed.csv')  # Fixed cross-validation
HOLDOUT_PATH = os.path.join(BASE_PATH, 'Folds_holdout.csv')  # Simple holdout
ROBUST_HOLDOUT_PATH = os.path.join(BASE_PATH, 'Folds_robust_holdout_balanced_large_test.csv')  # Advanced holdout
SLIDES_PATH = os.path.join(BASE_PATH, 'BreaKHis_v1/BreaKHis_v1/histology_slides/breast')


def get_training_config():
    """Get optimized training configuration for advanced model"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        
        # Check environment
        is_runpod = (
            os.path.exists('/workspace') or 
            'runpod' in os.environ.get('HOSTNAME', '').lower() or
            os.environ.get('RUNPOD_POD_ID') is not None
        )
        
        if is_runpod:
            config = {
                'device': device,
                'num_gpus': num_gpus,
                'batch_size': 8,  # Smaller batch for advanced model (more parameters)
                'num_workers': 8,
                'pin_memory': True,
                'persistent_workers': True,
                'environment': 'runpod'
            }
        else:
            config = {
                'device': device,
                'num_gpus': num_gpus,
                'batch_size': 6,  # Even smaller for local systems
                'num_workers': 4,
                'pin_memory': True,
                'persistent_workers': True,
                'environment': 'local_cuda'
            }
            
    elif torch.backends.mps.is_available():
        config = {
            'device': torch.device("mps"),
            'num_gpus': 1,
            'batch_size': 4,  # MPS with attention modules needs smaller batch
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'environment': 'mps'
        }
    else:
        config = {
            'device': torch.device("cpu"),
            'num_gpus': 0,
            'batch_size': 2,  # CPU with attention is very slow
            'num_workers': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'environment': 'cpu'
        }
    
    config['effective_batch_size'] = config['batch_size'] * max(1, config['num_gpus'])
    return config


# Advanced Training Configuration
ADVANCED_TRAINING_CONFIG = {
    'model_name': 'AdvancedMultiMagAttentionNet',  # Use advanced model
    'backbone': 'efficientnet_b2',  # Upgraded backbone
    
    # Training parameters
    'epochs': 30,  # Slightly more epochs for complex model
    'learning_rate': 3e-4,  # Lower LR for stability
    'weight_decay': 1e-3,
    'warmup_epochs': 5,  # More warmup for attention modules
    'patience': 10,  # More patience for complex model
    
    # Data parameters
    'samples_per_patient': 12,  # Fewer samples due to memory constraints
    'samples_per_patient_val': 8,  # Reduced validation samples
    'magnifications': ['40', '100', '200', '400'],
    'img_size': 224,
    'seed': 42,
    
    # Advanced model specific parameters
    'spatial_attention_scales': [1, 2, 4],  # Multi-scale spatial attention
    'channel_attention_reduction': 16,  # Channel attention reduction ratio
    'hierarchical_attention_heads': 8,  # Number of attention heads
    'cross_mag_attention_heads': 8,  # Cross-magnification attention heads
    'attention_dropout': 0.1,  # Attention dropout rate
    
    # Memory optimization
    'gradient_checkpointing': True,  # Save memory during backprop
    'mixed_precision': True,  # Use AMP for faster training
    'accumulate_grad_batches': 2,  # Gradient accumulation for effective larger batch
}

# Loss configuration for advanced model
ADVANCED_LOSS_CONFIG = {
    'class_weight': 0.85,  # Slightly less weight on binary classification
    'tumor_weight': 0.15,  # More weight on tumor type (attention can help here)
    'class_label_smoothing': 0.05,  # Less smoothing (attention provides regularization)
    'tumor_label_smoothing': 0.02,
    
    # Attention-specific losses
    'attention_entropy_weight': 0.01,  # Encourage diverse attention patterns
    'magnification_balance_weight': 0.005,  # Encourage balanced magnification usage
}

# Backwards compatibility
TRAINING_CONFIG = ADVANCED_TRAINING_CONFIG
LOSS_CONFIG = ADVANCED_LOSS_CONFIG


def get_model_config():
    """Get model-specific configuration"""
    return {
        'model_type': 'advanced_attention',
        'architecture': {
            'backbone': ADVANCED_TRAINING_CONFIG['backbone'],
            'magnifications': ADVANCED_TRAINING_CONFIG['magnifications'],
            'spatial_attention_scales': ADVANCED_TRAINING_CONFIG['spatial_attention_scales'],
            'attention_heads': ADVANCED_TRAINING_CONFIG['hierarchical_attention_heads'],
            'attention_dropout': ADVANCED_TRAINING_CONFIG['attention_dropout'],
        },
        'training': {
            'epochs': ADVANCED_TRAINING_CONFIG['epochs'],
            'learning_rate': ADVANCED_TRAINING_CONFIG['learning_rate'],
            'weight_decay': ADVANCED_TRAINING_CONFIG['weight_decay'],
            'mixed_precision': ADVANCED_TRAINING_CONFIG['mixed_precision'],
            'gradient_checkpointing': ADVANCED_TRAINING_CONFIG['gradient_checkpointing'],
        }
    }


def print_advanced_config():
    """Print the advanced configuration summary"""
    
    print("🚀 ADVANCED ATTENTION MODEL CONFIGURATION")
    print("=" * 80)
    
    device_config = get_training_config()
    
    print(f"🧠 Model Configuration:")
    print(f"   Model: {ADVANCED_TRAINING_CONFIG['model_name']}")
    print(f"   Backbone: {ADVANCED_TRAINING_CONFIG['backbone']}")
    print(f"   Magnifications: {ADVANCED_TRAINING_CONFIG['magnifications']}")
    print(f"   Attention heads: {ADVANCED_TRAINING_CONFIG['hierarchical_attention_heads']}")
    print(f"   Spatial scales: {ADVANCED_TRAINING_CONFIG['spatial_attention_scales']}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Device: {device_config['device']}")
    print(f"   Batch size: {device_config['batch_size']}")
    print(f"   Epochs: {ADVANCED_TRAINING_CONFIG['epochs']}")
    print(f"   Learning rate: {ADVANCED_TRAINING_CONFIG['learning_rate']}")
    print(f"   Mixed precision: {ADVANCED_TRAINING_CONFIG['mixed_precision']}")
    print(f"   Gradient checkpointing: {ADVANCED_TRAINING_CONFIG['gradient_checkpointing']}")
    
    print(f"\n📊 Data Configuration:")
    print(f"   Samples per patient (train): {ADVANCED_TRAINING_CONFIG['samples_per_patient']}")
    print(f"   Samples per patient (val): {ADVANCED_TRAINING_CONFIG['samples_per_patient_val']}")
    print(f"   Image size: {ADVANCED_TRAINING_CONFIG['img_size']}x{ADVANCED_TRAINING_CONFIG['img_size']}")
    
    print(f"\n🎯 Loss Configuration:")
    print(f"   Class weight: {ADVANCED_LOSS_CONFIG['class_weight']}")
    print(f"   Tumor weight: {ADVANCED_LOSS_CONFIG['tumor_weight']}")
    print(f"   Attention entropy: {ADVANCED_LOSS_CONFIG['attention_entropy_weight']}")
    print(f"   Magnification balance: {ADVANCED_LOSS_CONFIG['magnification_balance_weight']}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_advanced_config()