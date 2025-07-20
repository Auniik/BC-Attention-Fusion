"""
Fix Cross-Validation Data Leakage

This script implements proper patient-level cross-validation to fix the data leakage issue
where all patients appear in all folds, making the ensemble results meaningless.

Works with both local and RunPod environments using config.py paths.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import os
import sys

# Add the parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_PATH
from utils.helpers import get_base_path

def create_proper_cross_validation_folds(folds_csv_path, output_path=None, n_folds=5, random_state=42):
    """
    Create proper patient-level cross-validation folds
    
    Args:
        folds_csv_path: Path to original Folds.csv
        output_path: Path to save new Folds_fixed.csv
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
    """
    
    # Load original data
    df = pd.read_csv(folds_csv_path)
    print(f"Original data shape: {df.shape}")
    
    # Extract patient information
    df['patient_id'] = df['filename'].str.split('/').str[6]
    df['tumor_class'] = df['filename'].str.split('/').str[3]  # benign/malignant
    df['tumor_type'] = df['filename'].str.split('/').str[5]   # specific tumor type
    
    # Get unique patients with their class information
    patients_df = df.groupby('patient_id').agg({
        'tumor_class': 'first',
        'tumor_type': 'first'
    }).reset_index()
    
    print(f"\nTotal unique patients: {len(patients_df)}")
    print(f"Class distribution:")
    print(patients_df['tumor_class'].value_counts())
    print(f"\nTumor type distribution:")
    print(patients_df['tumor_type'].value_counts())
    
    # Create stratified folds based on tumor class
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Assign fold numbers to patients
    patients_df['fold'] = 0
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patients_df, patients_df['tumor_class']), 1):
        patients_df.loc[test_idx, 'fold'] = fold_idx
    
    print(f"\nPatients per fold:")
    for fold in range(1, n_folds + 1):
        fold_patients = patients_df[patients_df['fold'] == fold]
        print(f"Fold {fold}: {len(fold_patients)} patients")
        print(f"  - Benign: {len(fold_patients[fold_patients['tumor_class'] == 'benign'])}")
        print(f"  - Malignant: {len(fold_patients[fold_patients['tumor_class'] == 'malignant'])}")
    
    # Create patient-to-fold mapping
    patient_fold_map = dict(zip(patients_df['patient_id'], patients_df['fold']))
    
    # Create new fold assignments for all samples
    df_fixed = df.copy()
    df_fixed['new_fold'] = df_fixed['patient_id'].map(patient_fold_map)
    
    # For each fold, the test patients for that fold become 'test', others become 'train'
    new_rows = []
    
    for fold in range(1, n_folds + 1):
        # Get test patients for this fold
        test_patients = patients_df[patients_df['fold'] == fold]['patient_id'].tolist()
        
        # Get all samples
        for _, row in df.iterrows():
            patient_id = row['patient_id']
            new_row = row.copy()
            new_row['fold'] = fold
            
            if patient_id in test_patients:
                new_row['grp'] = 'test'
            else:
                new_row['grp'] = 'train'
            
            new_rows.append(new_row)
    
    # Create the final DataFrame
    df_final = pd.DataFrame(new_rows)
    
    # Verify the fix
    print(f"\n=== VERIFICATION ===")
    for fold in range(1, n_folds + 1):
        fold_data = df_final[df_final['fold'] == fold]
        train_patients = set(fold_data[fold_data['grp'] == 'train']['patient_id'].unique())
        test_patients = set(fold_data[fold_data['grp'] == 'test']['patient_id'].unique())
        overlap = train_patients & test_patients
        
        train_samples = len(fold_data[fold_data['grp'] == 'train'])
        test_samples = len(fold_data[fold_data['grp'] == 'test'])
        
        print(f"Fold {fold}:")
        print(f"  Train: {len(train_patients)} patients, {train_samples} samples")
        print(f"  Test: {len(test_patients)} patients, {test_samples} samples") 
        print(f"  Patient overlap: {len(overlap)} (should be 0)")
        
        if len(overlap) > 0:
            print(f"  ERROR: Patient overlap detected: {list(overlap)[:5]}")
    
    # Save the fixed folds
    if output_path is None:
        output_path = folds_csv_path.replace('.csv', '_fixed.csv')
    
    # Select only the required columns
    df_output = df_final[['fold', 'mag', 'grp', 'filename']].copy()
    df_output.to_csv(output_path, index=False)
    
    print(f"\n✅ Fixed cross-validation folds saved to: {output_path}")
    print(f"Original shape: {df.shape}, Fixed shape: {df_output.shape}")
    
    return df_output, patient_fold_map

def get_environment_paths():
    """Get the correct paths based on current environment"""
    base_path = get_base_path()
    environment = "local"
    
    if os.path.exists('/workspace'):
        environment = "runpod"
    elif os.path.exists('/kaggle'):
        environment = "kaggle"
    
    folds_path = os.path.join(base_path, 'breakhis', 'Folds.csv')
    output_path = os.path.join(base_path, 'breakhis', 'Folds_fixed.csv')
    
    print(f"🔧 Environment detected: {environment}")
    print(f"📂 Base path: {base_path}")
    print(f"📄 Input: {folds_path}")
    print(f"📄 Output: {output_path}")
    
    return folds_path, output_path, environment

if __name__ == "__main__":
    # Get environment-specific paths
    folds_path, output_path, environment = get_environment_paths()
    
    # Check if input file exists
    if not os.path.exists(folds_path):
        print(f"❌ ERROR: Input file not found: {folds_path}")
        print(f"Please ensure the BreakHis dataset is properly placed in the {environment} environment.")
        sys.exit(1)
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"⚠️  Output file already exists: {output_path}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Fix the cross-validation
    df_fixed, patient_map = create_proper_cross_validation_folds(
        folds_path, 
        output_path, 
        n_folds=5, 
        random_state=42
    )
    
    print(f"\n🎯 Cross-validation data leakage has been FIXED!")
    print(f"📊 Environment: {environment}")
    print(f"📄 Fixed folds saved to: {output_path}")
    print(f"✅ Ready for training with proper cross-validation!")