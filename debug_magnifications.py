#!/usr/bin/env python3
"""
Debug script to understand magnification availability in train/test splits
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datasets.preprocess import create_multi_mag_dataset_info
from datasets.examine import extract_tumor_type_and_patient_id
from config import BASE_PATH

def debug_magnifications():
    """Debug magnification availability by mode"""
    
    print("Debugging magnification availability...")
    
    # Load fold data
    folds_path = os.path.join(BASE_PATH, 'Folds_fixed.csv')
    fold_df = pd.read_csv(folds_path)
    
    # Add patient_id, tumor_class, tumor_type to dataframe
    fold_df['tumor_class'], fold_df['tumor_type'], fold_df['patient_id'], fold_df['magnification'] = \
        zip(*fold_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    # Focus on fold 1
    fold_data = fold_df[fold_df['fold'] == 1]
    
    print(f"Fold 1 total samples: {len(fold_data)}")
    print(f"Unique patients in fold 1: {fold_data['patient_id'].nunique()}")
    
    # Check magnification availability by mode
    for mode in ['train', 'test']:
        mode_data = fold_data[fold_data['grp'] == mode]
        print(f"\n=== {mode.upper()} MODE ===")
        print(f"Total samples: {len(mode_data)}")
        print(f"Unique patients: {mode_data['patient_id'].nunique()}")
        
        # Check magnification availability per patient
        patient_mags = mode_data.groupby('patient_id')['magnification'].apply(set).reset_index()
        patient_mags['mag_count'] = patient_mags['magnification'].apply(len)
        
        print(f"Patients with all 4 mags: {(patient_mags['mag_count'] == 4).sum()}")
        print(f"Patients with 3 mags: {(patient_mags['mag_count'] == 3).sum()}")
        print(f"Patients with 2 mags: {(patient_mags['mag_count'] == 2).sum()}")
        print(f"Patients with 1 mag: {(patient_mags['mag_count'] == 1).sum()}")
        
        # Show examples of patients with fewer than 4 mags
        incomplete_patients = patient_mags[patient_mags['mag_count'] < 4]
        if len(incomplete_patients) > 0:
            print(f"\nExamples of patients with missing magnifications:")
            for _, row in incomplete_patients.head(5).iterrows():
                missing_mags = set([40, 100, 200, 400]) - row['magnification']
                print(f"  {row['patient_id']}: has {row['magnification']}, missing {missing_mags}")
        
        # Check if any patient has all 4 magnifications
        complete_patients = patient_mags[patient_mags['mag_count'] == 4]['patient_id'].tolist()
        if complete_patients:
            print(f"\nPatients with all 4 magnifications ({len(complete_patients)}):")
            print(f"  {complete_patients[:5]}...")  # Show first 5
        else:
            print(f"\n⚠️  NO patients have all 4 magnifications in {mode} mode!")
    
    # Check if the issue is in the split or in the original data
    print(f"\n=== ORIGINAL DATA CHECK ===")
    multi_mag_patients, _, _, _ = create_multi_mag_dataset_info(fold_df, fold=1)
    print(f"Multi-mag patients found: {len(multi_mag_patients)}")
    
    if multi_mag_patients:
        # Check first patient's image structure
        sample_patient = multi_mag_patients[0]
        print(f"\nSample patient: {sample_patient['patient_id']}")
        print(f"Available magnifications: {list(sample_patient['images'].keys())}")
        for mag, files in sample_patient['images'].items():
            print(f"  {mag}x: {len(files)} images")

if __name__ == "__main__":
    debug_magnifications()