import os
import torch
from utils.helpers import get_base_path


BASE_PATH = get_base_path() + '/breakhis'


FOLD_PATH = os.path.join(BASE_PATH, 'Folds_fixed.csv')  # Fixed cross-validation without data leakage
SLIDES_PATH = os.path.join(BASE_PATH, 'BreaKHis_v1/BreaKHis_v1/histology_slides/breast')


# Device-specific training configurations
def get_training_config():
    """Get optimized training configuration based on the device and environment"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        
        # Check if we're in RunPod environment (common indicators)
        is_runpod = (
            os.path.exists('/workspace') or 
            'runpod' in os.environ.get('HOSTNAME', '').lower() or
            os.environ.get('RUNPOD_POD_ID') is not None
        )
        
        if is_runpod:
            # RunPod configuration - optimized for cloud GPUs
            config = {
                'device': device,
                'num_gpus': num_gpus,
                'batch_size': 16,  # RTX 4090/A100 optimized
                'num_workers': 8,  # Parallel data loading
                'pin_memory': True,
                'persistent_workers': True,
                'environment': 'runpod'
            }
        else:
            # Local CUDA configuration - more conservative
            config = {
                'device': device,
                'num_gpus': num_gpus,
                'batch_size': 12,  # Slightly smaller for local GPUs
                'num_workers': 4,  # Conservative for local systems
                'pin_memory': True,
                'persistent_workers': True,
                'environment': 'local_cuda'
            }
            
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) configuration
        config = {
            'device': torch.device("mps"),
            'num_gpus': 1,
            'batch_size': 8,  # MPS has memory limitations
            'num_workers': 0,  # MPS doesn't work well with multiprocessing
            'pin_memory': False,  # Not needed for MPS
            'persistent_workers': False,
            'environment': 'mps'
        }
    else:
        # CPU fallback configuration
        config = {
            'device': torch.device("cpu"),
            'num_gpus': 0,
            'batch_size': 4,  # Small batch for CPU
            'num_workers': 2,  # Limited parallelism for CPU
            'pin_memory': False,
            'persistent_workers': False,
            'environment': 'cpu'
        }
    
    # Add derived configurations
    config['effective_batch_size'] = config['batch_size'] * max(1, config['num_gpus'])
    
    return config


# Training hyperparameters
TRAINING_CONFIG = {
    'num_epochs': 25,
    'learning_rate': 5e-4,
    'weight_decay': 1e-3,
    'warmup_epochs': 3,
    'patience': 8,
    'samples_per_patient_train': 16,
    'samples_per_patient_val': 8,  # Increased for more robust evaluation
    'magnifications': [40, 100, 200, 400],
    'img_size': 224,
    'seed': 42
}

# Loss function weights
LOSS_CONFIG = {
    'class_weight': 0.9,
    'tumor_weight': 0.1,
    'class_label_smoothing': 0.1,
    'tumor_label_smoothing': 0.05
}