#!/usr/bin/env python3
"""
Debug magnification type issues 
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datasets.preprocess import create_multi_mag_dataset_info, get_patients_for_mode
from datasets.examine import folds_df

def debug_types():
    """Debug magnification type issues"""
    
    print("Debugging magnification types...")
    
    # Test fold 1
    fold = 1
    multi_mag_patients, _, fold_df, _ = create_multi_mag_dataset_info(folds_df, fold=fold)
    train_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='train')
    
    if train_patients:
        first_patient = train_patients[0]
        print(f"Patient: {first_patient['patient_id']}")
        print(f"Images keys: {list(first_patient['images'].keys())}")
        print(f"Key types: {[type(k) for k in first_patient['images'].keys()]}")
        
        # Check what fold_df looks like for this patient
        patient_data = fold_df[fold_df['patient_id'] == first_patient['patient_id']]
        print(f"\nFold data for patient:")
        print(f"Magnifications: {patient_data['magnification'].unique()}")
        print(f"Magnification types: {[type(m) for m in patient_data['magnification'].unique()]}")
        print(f"Modes: {patient_data['grp'].unique()}")
        
        # Check what happens in filtering
        print(f"\nTesting filtering logic:")
        mags = [40, 100, 200, 400]
        mode = 'train'
        
        # Get fold data for this patient in train mode
        patient_fold_data = fold_df[
            (fold_df['patient_id'] == first_patient['patient_id']) & 
            (fold_df['grp'] == mode)
        ]
        print(f"Patient fold data shape: {patient_fold_data.shape}")
        
        mode_images = {}
        for mag in mags:
            # Convert to string for comparison (magnification column contains strings)
            mag_str = str(mag)
            mag_files = patient_fold_data[patient_fold_data['magnification'] == mag_str]['filename'].tolist()
            print(f"Mag {mag} (str: {mag_str}): {len(mag_files)} files")
            if mag_files:
                mode_images[mag] = mag_files
        
        print(f"Final mode_images keys: {list(mode_images.keys())}")
        print(f"Required mags: {mags}")
        print(f"Sets equal? {set(mode_images.keys()) == set(mags)}")

if __name__ == "__main__":
    debug_types()