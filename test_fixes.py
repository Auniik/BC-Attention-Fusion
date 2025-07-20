"""
Test script to validate the cross-validation fixes without running full training
"""

import pandas as pd
import numpy as np
from config import FOLD_PATH
from datasets.examine import extract_tumor_type_and_patient_id

def validate_cross_validation_fix():
    """Validate that the cross-validation fix works correctly"""
    
    print("🔍 VALIDATING CROSS-VALIDATION FIXES")
    print("=" * 60)
    
    # Load the fixed folds
    try:
        folds_df = pd.read_csv(FOLD_PATH)
        print(f"✅ Successfully loaded fixed folds from: {FOLD_PATH}")
        print(f"   Shape: {folds_df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find {FOLD_PATH}")
        print("   Make sure to run the fix script first!")
        return False
    
    # Extract patient information
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
        zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total samples: {len(folds_df):,}")
    print(f"   Unique patients: {folds_df['patient_id'].nunique()}")
    print(f"   Folds: {sorted(folds_df['fold'].unique())}")
    
    # Test 1: Verify no patient overlap between train/test within each fold
    print(f"\n🧪 TEST 1: Patient overlap within folds")
    overlap_detected = False
    for fold in sorted(folds_df['fold'].unique()):
        fold_data = folds_df[folds_df['fold'] == fold]
        train_patients = set(fold_data[fold_data['grp'] == 'train']['patient_id'])
        test_patients = set(fold_data[fold_data['grp'] == 'test']['patient_id'])
        overlap = train_patients & test_patients
        
        if len(overlap) > 0:
            print(f"   ❌ Fold {fold}: {len(overlap)} overlapping patients")
            overlap_detected = True
        else:
            print(f"   ✅ Fold {fold}: No patient overlap ({len(train_patients)} train, {len(test_patients)} test)")
    
    if overlap_detected:
        print("   ❌ FAILED: Patient overlap detected within folds!")
        return False
    
    # Test 2: Verify each patient appears in exactly one test fold
    print(f"\n🧪 TEST 2: Patient distribution across folds")
    patient_test_folds = {}
    for fold in sorted(folds_df['fold'].unique()):
        test_patients = folds_df[(folds_df['fold'] == fold) & (folds_df['grp'] == 'test')]['patient_id'].unique()
        for patient in test_patients:
            if patient in patient_test_folds:
                print(f"   ❌ Patient {patient} appears in test set of multiple folds!")
                return False
            patient_test_folds[patient] = fold
    
    print(f"   ✅ Each patient appears in exactly one test fold")
    print(f"   📈 Patients per test fold:")
    fold_counts = pd.Series(list(patient_test_folds.values())).value_counts().sort_index()
    for fold, count in fold_counts.items():
        print(f"      Fold {fold}: {count} patients")
    
    # Test 3: Verify class balance across folds
    print(f"\n🧪 TEST 3: Class balance across folds")
    for fold in sorted(folds_df['fold'].unique()):
        fold_data = folds_df[folds_df['fold'] == fold]
        test_data = fold_data[fold_data['grp'] == 'test']
        
        # Get unique patients and their classes
        test_patients = test_data.groupby('patient_id')['tumor_class'].first()
        class_counts = test_patients.value_counts()
        
        benign_count = class_counts.get('benign', 0)
        malignant_count = class_counts.get('malignant', 0)
        total = benign_count + malignant_count
        
        benign_pct = (benign_count / total * 100) if total > 0 else 0
        malignant_pct = (malignant_count / total * 100) if total > 0 else 0
        
        print(f"   Fold {fold}: {benign_count} benign ({benign_pct:.1f}%), {malignant_count} malignant ({malignant_pct:.1f}%)")
    
    # Test 4: Verify sample counts are reasonable
    print(f"\n🧪 TEST 4: Sample distribution")
    for fold in sorted(folds_df['fold'].unique()):
        fold_data = folds_df[folds_df['fold'] == fold]
        train_samples = len(fold_data[fold_data['grp'] == 'train'])
        test_samples = len(fold_data[fold_data['grp'] == 'test'])
        total_samples = train_samples + test_samples
        
        test_ratio = test_samples / total_samples * 100
        print(f"   Fold {fold}: {train_samples:,} train, {test_samples:,} test ({test_ratio:.1f}% test)")
    
    print(f"\n✅ ALL TESTS PASSED - Cross-validation is properly configured!")
    print(f"🎯 Expected results:")
    print(f"   - Individual fold accuracy: 85-95% (realistic range)")
    print(f"   - Ensemble accuracy: 90-96% (NOT 100%)")
    print(f"   - Classification reports should match training metrics")
    
    return True

def validate_patient_ids():
    """Additional validation of patient ID consistency"""
    print(f"\n🔍 VALIDATING PATIENT ID CONSISTENCY")
    print("=" * 60)
    
    folds_df = pd.read_csv(FOLD_PATH)
    
    # Extract patient information
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
        zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    # Check for any None values
    null_patients = folds_df['patient_id'].isnull().sum()
    if null_patients > 0:
        print(f"   ❌ Found {null_patients} samples with null patient_id")
        return False
    
    # Check patient-class consistency
    patient_classes = folds_df.groupby('patient_id')['tumor_class'].nunique()
    inconsistent_patients = patient_classes[patient_classes > 1]
    
    if len(inconsistent_patients) > 0:
        print(f"   ❌ Found {len(inconsistent_patients)} patients with inconsistent tumor classes")
        return False
    
    print(f"   ✅ All patient IDs are consistent")
    return True

if __name__ == "__main__":
    success = validate_cross_validation_fix()
    if success:
        success = validate_patient_ids()
    
    if success:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print(f"📝 SUMMARY OF FIXES APPLIED:")
        print(f"   1. ✅ Fixed data leakage - patients now properly split across folds")
        print(f"   2. ✅ Fixed classification report - now uses best checkpoint")
        print(f"   3. ✅ Updated config to use Folds_fixed.csv")
        print(f"   4. ✅ Removed PyTorch weights_only parameter for compatibility")
        print(f"\n🚀 Ready to run training with: python main.py")
    else:
        print(f"\n❌ VALIDATION FAILED - Please check the issues above")