#!/usr/bin/env python3
"""
Debug script to understand why dataset filtering is getting 0 samples
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datasets.preprocess import create_multi_mag_dataset_info, get_patients_for_mode
from datasets.examine import folds_df

def debug_filtering():
    """Debug why we're getting 0 samples"""
    
    print("Debugging dataset filtering...")
    
    # Test fold 1
    fold = 1
    multi_mag_patients, _, fold_df, _ = create_multi_mag_dataset_info(folds_df, fold=fold)
    
    print(f"Multi-mag patients: {len(multi_mag_patients)}")
    print(f"Fold DF shape: {fold_df.shape}")
    
    # Check train patients
    train_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='train')
    print(f"Train patients after filtering: {len(train_patients)}")
    
    # Check what's in the first patient
    if train_patients:
        first_patient = train_patients[0]
        print(f"First train patient: {first_patient['patient_id']}")
        print(f"Images available: {list(first_patient['images'].keys())}")
        print(f"Tumor class: {first_patient['tumor_class']}")
    
    # Test what happens when we create dataset
    from datasets.multi_mag import MultiMagnificationDataset
    
    if train_patients:
        print(f"\nTesting dataset creation...")
        
        # Test with require_all_mags=False first
        test_dataset = MultiMagnificationDataset(
            train_patients[:1], 
            fold_df,
            mode='train',
            mags=[40, 100, 200, 400],
            samples_per_patient=2,
            transform=None,
            balance_classes=False,
            require_all_mags=False
        )
        print(f"Dataset with require_all_mags=False: {len(test_dataset)} samples")
        
        # Test with require_all_mags=True
        test_dataset2 = MultiMagnificationDataset(
            train_patients[:1], 
            fold_df,
            mode='train',
            mags=[40, 100, 200, 400],
            samples_per_patient=2,
            transform=None,
            balance_classes=False,
            require_all_mags=True
        )
        print(f"Dataset with require_all_mags=True: {len(test_dataset2)} samples")
        
        if len(test_dataset2) > 0:
            sample = test_dataset2[0]
            print(f"Sample patient: {sample['patient_id']}")
            print(f"Available mags: {list(sample['images'].keys())}")

if __name__ == "__main__":
    debug_filtering()