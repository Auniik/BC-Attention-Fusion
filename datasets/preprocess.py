
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def create_multi_mag_dataset_info(df, fold=1, plot_distribution=False):
    """Create dataset info for multi-magnification training with summary stats"""
    fold_df = df[df['fold'] == fold].copy()

    # Group by patient
    patient_groups = fold_df.groupby('patient_id')

    multi_mag_patients = []
    single_mag_patients = []

    for patient_id, group in patient_groups:
        mags_available = sorted(group['magnification'].unique())

        patient_info = {
            'patient_id': patient_id,
            'tumor_class': group['tumor_class'].iloc[0],
            'tumor_type': group['tumor_type'].iloc[0],
            'magnifications': mags_available,
            'images': {}
        }

        for mag in mags_available:
            mag_images = group[group['magnification'] == mag]['filename'].tolist()
            patient_info['images'][mag] = mag_images

        if len(mags_available) == 4:
            multi_mag_patients.append(patient_info)
        else:
            single_mag_patients.append(patient_info)

    # Class distribution in multi-mag patients
    class_distribution = None
    if multi_mag_patients:
        class_distribution = pd.DataFrame(multi_mag_patients)['tumor_class'].value_counts()

    fold_statistics = {
        "fold_no": fold,
        "multi_mag_patients_count": len(multi_mag_patients),
        "single_mag_patients_count": len(single_mag_patients),
        "train_samples": len(fold_df[fold_df['grp'] == 'train']),
        "test_samples": len(fold_df[fold_df['grp'] == 'test']),
        "multi_mag_patients": multi_mag_patients,
        "class_distribution": class_distribution
    }

    return multi_mag_patients, single_mag_patients, fold_df, fold_statistics



import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def create_balanced_sampling_weights(fold_df, mode='train'):
    """Create sampling weights for balanced training"""
    
    train_df = fold_df[fold_df['grp'] == mode]
    
    # Calculate class weights (inverse frequency)
    class_counts = train_df['tumor_class'].value_counts()
    total_samples = len(train_df)
    class_weights = {
        cls: total_samples / (len(class_counts) * count) 
        for cls, count in class_counts.items()
    }
    
    # Calculate tumor type weights for fine-grained balancing
    tumor_counts = train_df['tumor_type'].value_counts()
    tumor_weights = {
        tumor: total_samples / (len(tumor_counts) * count)
        for tumor, count in tumor_counts.items()
    }
    
    # Patient-level weights to avoid patient bias
    patient_counts = train_df.groupby('patient_id').size()
    patient_weights = {
        pid: 1.0 / count for pid, count in patient_counts.items()
    }
    
    print(f"\n=== CALCULATED WEIGHTS ===")
    print(f"Class weights: {class_weights}")
    print(f"\nTumor type weights (top 5):")
    for tumor, weight in sorted(tumor_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {tumor}: {weight:.3f}")
    
    return class_weights, tumor_weights, patient_weights

def get_balanced_tumor_weights(fold_df, device='cuda'):
    """Get tensor weights for tumor type classification"""
    train_df = fold_df[fold_df['grp'] == 'train']
    
    # Get all unique tumor types in order
    all_tumor_types = sorted(fold_df['tumor_type'].unique())
    
    # Count samples per tumor type
    tumor_counts = train_df['tumor_type'].value_counts()
    
    # Calculate weights
    total_samples = len(train_df)
    weights = []
    for tumor in all_tumor_types:
        count = tumor_counts.get(tumor, 1)  # Avoid division by zero
        weight = total_samples / (len(all_tumor_types) * count)
        weights.append(weight)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Normalize weights to have mean=1
    weights_tensor = weights_tensor / weights_tensor.mean()
    
    print(f"\nTumor type weights tensor: {weights_tensor}")
    
    return weights_tensor
