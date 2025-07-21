#!/usr/bin/env python3
"""
Simple test script for robust holdout configurations

This script tests one configuration to ensure everything works before
running the full robust evaluation.
"""

import os
import sys
import pandas as pd

# Test if the CSV files exist
def test_csv_files():
    print("🔍 Testing robust holdout CSV files...")
    
    configs = ['balanced_large_test', 'moderate_test', 'large_test']
    
    for config in configs:
        csv_path = f"data/breakhis/Folds_robust_holdout_{config}.csv"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ {config}: {len(df)} samples")
            
            # Check split distribution
            split_dist = df['grp'].value_counts()
            print(f"   Train: {split_dist.get('train', 0)}")
            print(f"   Val: {split_dist.get('val', 0)}")  
            print(f"   Test: {split_dist.get('test', 0)}")
        else:
            print(f"❌ Missing: {csv_path}")
    
    return True

def test_patient_counts():
    print("\n📊 Testing patient counts per configuration...")
    
    from datasets.examine import extract_tumor_type_and_patient_id
    
    configs = ['balanced_large_test', 'moderate_test', 'large_test']
    
    for config in configs:
        csv_path = f"data/breakhis/Folds_robust_holdout_{config}.csv"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Extract patient info
            df['tumor_class'], df['tumor_type'], df['patient_id'], df['magnification'] = \
                zip(*df['filename'].apply(extract_tumor_type_and_patient_id))
            
            # Count unique patients per split
            patient_counts = df.groupby('grp')['patient_id'].nunique()
            total_patients = df['patient_id'].nunique()
            
            print(f"\n{config}:")
            print(f"   Train: {patient_counts.get('train', 0)} patients")
            print(f"   Val: {patient_counts.get('val', 0)} patients")
            print(f"   Test: {patient_counts.get('test', 0)} patients") 
            print(f"   Total unique: {total_patients} patients")
            
            # Check for overlaps
            train_patients = set(df[df['grp'] == 'train']['patient_id'].unique())
            val_patients = set(df[df['grp'] == 'val']['patient_id'].unique())
            test_patients = set(df[df['grp'] == 'test']['patient_id'].unique())
            
            if len(train_patients & val_patients) == 0 and \
               len(train_patients & test_patients) == 0 and \
               len(val_patients & test_patients) == 0:
                print(f"   ✅ No patient overlaps")
            else:
                print(f"   ❌ Patient overlaps detected!")

def main():
    print("🧪 ROBUST HOLDOUT CONFIGURATION TEST")
    print("=" * 50)
    
    # Test 1: CSV files exist
    test_csv_files()
    
    # Test 2: Patient counts are correct
    test_patient_counts()
    
    print("\n✅ Test completed!")
    print("\n💡 Ready to run: python main_robust_holdout.py")
    print("   This will test all 3 configurations and compare results")

if __name__ == "__main__":
    main()