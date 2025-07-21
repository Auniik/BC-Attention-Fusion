#!/usr/bin/env python3
"""
Analysis of patient distribution in BreakHis dataset for evaluating
feasibility of Option 1: True Holdout Patient Split (60 train + 10 validation + 12 test)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from config import FOLD_PATH

def extract_patient_info(filename):
    """Extract patient ID and class information from filename"""
    parts = filename.split('/')
    
    if len(parts) >= 8:
        tumor_class = parts[3]  # benign or malignant
        tumor_type = parts[5]   # adenosis, fibroadenoma, etc.
        patient_id = parts[6]   # SOB_B_A_14-22549AB, etc.
        magnification = parts[7].replace('X', '')  # 40, 100, 200, 400
        
        return tumor_class, tumor_type, patient_id, magnification
    return None, None, None, None

def analyze_patient_distribution():
    """Analyze patient distribution for feasibility of true holdout split"""
    
    print("=" * 80)
    print("BREAKHIS PATIENT DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Load the fixed folds CSV
    try:
        folds_df = pd.read_csv(FOLD_PATH)
        print(f"✅ Loaded Folds_fixed.csv: {folds_df.shape[0]} total samples")
    except FileNotFoundError:
        print(f"❌ Could not find {FOLD_PATH}")
        return
    
    # Extract patient information
    print("\n📊 Extracting patient information...")
    results = folds_df['filename'].apply(extract_patient_info)
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = zip(*results)
    
    # Remove any invalid entries
    valid_mask = folds_df['patient_id'].notna()
    folds_df = folds_df[valid_mask]
    print(f"✅ Valid samples after cleanup: {folds_df.shape[0]}")
    
    # 1. TOTAL UNIQUE PATIENTS
    unique_patients = folds_df['patient_id'].nunique()
    print(f"\n🏥 TOTAL UNIQUE PATIENTS: {unique_patients}")
    
    # 2. CLASS DISTRIBUTION BY PATIENT
    print(f"\n📈 CLASS DISTRIBUTION BY PATIENT:")
    patient_classes = folds_df.drop_duplicates('patient_id')[['patient_id', 'tumor_class']]
    class_counts = patient_classes['tumor_class'].value_counts()
    
    benign_patients = class_counts.get('benign', 0)
    malignant_patients = class_counts.get('malignant', 0)
    
    print(f"   Benign patients:    {benign_patients:2d} ({benign_patients/unique_patients*100:.1f}%)")
    print(f"   Malignant patients: {malignant_patients:2d} ({malignant_patients/unique_patients*100:.1f}%)")
    print(f"   Ratio (B:M):        {benign_patients}:{malignant_patients}")
    
    # 3. TUMOR TYPE DISTRIBUTION BY PATIENT
    print(f"\n🔬 TUMOR TYPE DISTRIBUTION BY PATIENT:")
    tumor_type_counts = folds_df.drop_duplicates('patient_id').groupby(['tumor_class', 'tumor_type']).size()
    
    for class_type in ['benign', 'malignant']:
        if class_type in tumor_type_counts.index:
            print(f"   {class_type.title()}:")
            for tumor_type, count in tumor_type_counts[class_type].items():
                print(f"     {tumor_type:<20} {count:2d} patients")
    
    # 4. MAGNIFICATION COMPLETENESS
    print(f"\n🔍 MAGNIFICATION COMPLETENESS BY PATIENT:")
    patient_mag_counts = folds_df.groupby('patient_id')['magnification'].nunique()
    mag_completeness = patient_mag_counts.value_counts().sort_index()
    
    for mag_count, patient_count in mag_completeness.items():
        print(f"   {mag_count} magnifications: {patient_count:2d} patients ({patient_count/unique_patients*100:.1f}%)")
    
    complete_patients = mag_completeness.get(4, 0)
    print(f"   ✅ Complete (4 mags): {complete_patients} patients ({complete_patients/unique_patients*100:.1f}%)")
    
    # 5. CURRENT FOLD STRUCTURE
    print(f"\n📁 CURRENT FOLD STRUCTURE:")
    print(f"   Total folds: {folds_df['fold'].nunique()}")
    
    # Check patient overlap between folds (should be 0 for patient-level splits)
    patient_fold_counts = folds_df.drop_duplicates(['patient_id', 'fold']).groupby('patient_id')['fold'].nunique()
    patients_in_multiple_folds = (patient_fold_counts > 1).sum()
    
    if patients_in_multiple_folds == 0:
        print(f"   ✅ Patient-level splits: No patient overlap between folds")
    else:
        print(f"   ❌ Patient overlap: {patients_in_multiple_folds} patients appear in multiple folds")
    
    # Show fold distribution
    fold_patient_counts = folds_df.drop_duplicates(['patient_id', 'fold']).groupby('fold')['patient_id'].nunique()
    print(f"   Patients per fold:")
    for fold, count in fold_patient_counts.items():
        print(f"     Fold {fold}: {count:2d} patients")
    
    # 6. FEASIBILITY ANALYSIS FOR 60/10/12 SPLIT
    print(f"\n🎯 FEASIBILITY ANALYSIS: 60 TRAIN + 10 VALIDATION + 12 TEST SPLIT")
    print(f"   Current total patients: {unique_patients}")
    print(f"   Proposed split total:   {60 + 10 + 12} = 82 patients")
    
    if unique_patients >= 82:
        print(f"   ✅ FEASIBLE: We have {unique_patients} patients (≥82 required)")
        
        # Check if we can maintain class balance
        proposed_benign_train = int(60 * benign_patients / unique_patients)
        proposed_benign_val = int(10 * benign_patients / unique_patients) 
        proposed_benign_test = benign_patients - proposed_benign_train - proposed_benign_val
        
        proposed_malignant_train = 60 - proposed_benign_train
        proposed_malignant_val = 10 - proposed_benign_val
        proposed_malignant_test = 12 - proposed_benign_test
        
        print(f"\n   📊 PROPOSED CLASS DISTRIBUTION:")
        print(f"                    Benign  Malignant  Total")
        print(f"   Train (60):      {proposed_benign_train:6d}  {proposed_malignant_train:9d}     60")
        print(f"   Validation (10): {proposed_benign_val:6d}  {proposed_malignant_val:9d}     10") 
        print(f"   Test (12):       {proposed_benign_test:6d}  {proposed_malignant_test:9d}     12")
        print(f"   Total:           {benign_patients:6d}  {malignant_patients:9d}     {unique_patients}")
        
        # Check if any proposed counts are negative or zero
        if min(proposed_benign_train, proposed_benign_val, proposed_benign_test,
               proposed_malignant_train, proposed_malignant_val, proposed_malignant_test) <= 0:
            print(f"   ⚠️  WARNING: Some splits would have ≤0 patients of a class")
        else:
            print(f"   ✅ All splits have >0 patients of each class")
            
    else:
        print(f"   ❌ NOT FEASIBLE: Only {unique_patients} patients (need ≥82)")
        max_split = unique_patients - 2  # Keep at least 1 for val and test
        print(f"   📉 Maximum possible split: {max_split} train + 1 validation + 1 test")
    
    # 7. ALTERNATIVE SPLIT RECOMMENDATIONS
    print(f"\n💡 ALTERNATIVE SPLIT RECOMMENDATIONS:")
    
    if unique_patients >= 60:
        # 70/15/15 split
        train_70 = int(unique_patients * 0.70)
        val_15 = int(unique_patients * 0.15)
        test_15 = unique_patients - train_70 - val_15
        print(f"   Option A (70/15/15%): {train_70} train + {val_15} val + {test_15} test")
        
        # 60/20/20 split  
        train_60 = int(unique_patients * 0.60)
        val_20 = int(unique_patients * 0.20)
        test_20 = unique_patients - train_60 - val_20
        print(f"   Option B (60/20/20%): {train_60} train + {val_20} val + {test_20} test")
        
    # Keep current 5-fold CV but add holdout
    if unique_patients >= 20:
        holdout_test = min(12, int(unique_patients * 0.15))
        remaining = unique_patients - holdout_test
        print(f"   Option C (Hybrid):    {remaining} for 5-fold CV + {holdout_test} holdout test")
    
    return {
        'total_patients': unique_patients,
        'benign_patients': benign_patients,
        'malignant_patients': malignant_patients,
        'complete_patients': complete_patients,
        'feasible_60_10_12': unique_patients >= 82
    }

if __name__ == "__main__":
    results = analyze_patient_distribution()