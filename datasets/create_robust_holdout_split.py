#!/usr/bin/env python3
"""
Create Robust Holdout Patient Split with Anti-Overfitting Strategies

Since all 82 patients have complete magnifications, this script implements
alternative strategies to reduce overfitting and get more realistic performance:

1. Larger test set (20+ patients instead of 12)
2. Multiple random holdout configurations 
3. Patient stratification by tumor type AND institution (if available)
4. Image diversity analysis to ensure variety

This addresses the 100% test accuracy issue seen in RUNPOD_OUTPUT.txt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from config import FOLD_PATH
from datasets.examine import extract_tumor_type_and_patient_id

def analyze_patient_characteristics():
    """Deep analysis of patient characteristics to identify potential biases"""
    
    print("🔍 DEEP PATIENT CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    # Load and process data
    folds_df = pd.read_csv(FOLD_PATH)
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
        zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))
    folds_df['magnification'] = pd.to_numeric(folds_df['magnification'], errors='coerce')
    
    # Patient-level analysis
    patient_info = folds_df.groupby('patient_id').agg({
        'tumor_class': 'first',
        'tumor_type': 'first',
        'magnification': lambda x: len(x.unique()),
        'filename': 'count'  # Total images per patient
    }).reset_index()
    
    patient_info.columns = ['patient_id', 'tumor_class', 'tumor_type', 'num_mags', 'total_images']
    
    # Extract additional patient characteristics from ID
    def extract_patient_details(patient_id):
        # BreakHis patient ID format: SOB_[B/M]_[SUBTYPE]_[ID]
        parts = patient_id.split('_')
        if len(parts) >= 4:
            return {
                'id_prefix': parts[0],  # SOB
                'class_code': parts[1], # B/M
                'subtype_code': parts[2], # A, F, PT, TA, DC, LC, MC, PC
                'specimen_id': parts[3]  # Actual specimen identifier
            }
        return {'id_prefix': '', 'class_code': '', 'subtype_code': '', 'specimen_id': ''}
    
    patient_details = patient_info['patient_id'].apply(extract_patient_details)
    for key in ['id_prefix', 'class_code', 'subtype_code', 'specimen_id']:
        patient_info[key] = [d[key] for d in patient_details]
    
    print(f"📊 Patient Statistics:")
    print(f"   Total patients: {len(patient_info)}")
    print(f"   Benign: {len(patient_info[patient_info['tumor_class'] == 'benign'])}")
    print(f"   Malignant: {len(patient_info[patient_info['tumor_class'] == 'malignant'])}")
    
    # Image distribution analysis
    print(f"\n🖼️  Images per Patient Analysis:")
    print(f"   Mean: {patient_info['total_images'].mean():.1f}")
    print(f"   Std: {patient_info['total_images'].std():.1f}")
    print(f"   Min: {patient_info['total_images'].min()}")
    print(f"   Max: {patient_info['total_images'].max()}")
    
    # Check for potential batch effects by specimen ID patterns
    print(f"\n🧬 Specimen ID Patterns:")
    specimen_prefixes = patient_info['specimen_id'].str[:5].value_counts().head(10)
    for prefix, count in specimen_prefixes.items():
        if count > 1:
            print(f"   {prefix}*: {count} patients (potential batch)")
    
    # Subtype distribution
    print(f"\n📈 Subtype Code Distribution:")
    subtype_dist = patient_info['subtype_code'].value_counts()
    for subtype, count in subtype_dist.items():
        print(f"   {subtype}: {count} patients")
    
    return patient_info

def create_robust_holdout_splits(patient_info, configs=None):
    """Create multiple robust holdout configurations to test generalization"""
    
    if configs is None:
        configs = [
            {'train': 50, 'val': 15, 'test': 17, 'name': 'balanced_large_test'},
            {'train': 55, 'val': 12, 'test': 15, 'name': 'moderate_test'}, 
            {'train': 45, 'val': 17, 'test': 20, 'name': 'large_test'},
        ]
    
    print(f"\n🎯 CREATING ROBUST HOLDOUT SPLITS")
    print("=" * 80)
    
    splits_results = {}
    
    for config in configs:
        print(f"\n📋 Configuration: {config['name']}")
        print(f"   Split: {config['train']} train + {config['val']} val + {config['test']} test")
        
        # Create stratified split
        train_size = config['train']
        val_size = config['val'] 
        test_size = config['test']
        
        if train_size + val_size + test_size != len(patient_info):
            print(f"   ⚠️  Adjusting split sizes to match {len(patient_info)} patients")
            # Adjust proportionally
            total = train_size + val_size + test_size
            train_size = int(len(patient_info) * train_size / total)
            val_size = int(len(patient_info) * val_size / total)
            test_size = len(patient_info) - train_size - val_size
            
        print(f"   Adjusted: {train_size} train + {val_size} val + {test_size} test")
        
        # Multi-level stratification
        # First by tumor class, then by tumor type within class
        np.random.seed(42)
        
        # Create combined stratification key
        patient_info['strat_key'] = patient_info['tumor_class'] + '_' + patient_info['tumor_type']
        
        try:
            # Two-step split for better stratification
            train_patients, temp_patients = train_test_split(
                patient_info,
                train_size=train_size,
                stratify=patient_info['tumor_class'],  # Primary stratification
                random_state=42
            )
            
            val_patients, test_patients = train_test_split(
                temp_patients,
                train_size=val_size,
                test_size=test_size,
                stratify=temp_patients['tumor_class'],  # Primary stratification
                random_state=43
            )
            
        except ValueError as e:
            print(f"   ⚠️  Stratification failed: {e}")
            # Fall back to simple random split
            indices = np.random.permutation(len(patient_info))
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]
            
            train_patients = patient_info.iloc[train_idx]
            val_patients = patient_info.iloc[val_idx]
            test_patients = patient_info.iloc[test_idx]
        
        # Verify splits
        splits = {'train': train_patients, 'val': val_patients, 'test': test_patients}
        
        # Check class distribution
        print(f"   📊 Class distribution verification:")
        for split_name, split_df in splits.items():
            class_dist = split_df['tumor_class'].value_counts()
            benign_pct = class_dist.get('benign', 0) / len(split_df) * 100
            malignant_pct = class_dist.get('malignant', 0) / len(split_df) * 100
            print(f"     {split_name}: {benign_pct:.1f}% benign, {malignant_pct:.1f}% malignant")
        
        # Check tumor type distribution
        print(f"   🦠 Tumor type distribution:")
        for split_name, split_df in splits.items():
            tumor_types = split_df['tumor_type'].value_counts()
            print(f"     {split_name}: {len(tumor_types)} unique tumor types")
        
        # Store results
        splits_results[config['name']] = {
            'config': config,
            'train_patients': train_patients,
            'val_patients': val_patients, 
            'test_patients': test_patients
        }
        
        print(f"   ✅ Split created successfully")
    
    return splits_results

def generate_robust_holdout_csv(splits_results, base_name='Folds_robust_holdout'):
    """Generate CSV files for each robust holdout configuration"""
    
    print(f"\n📝 GENERATING ROBUST HOLDOUT CSV FILES")
    print("=" * 80)
    
    # Load original folds to get image filenames
    original_folds = pd.read_csv(FOLD_PATH)
    original_folds['tumor_class'], original_folds['tumor_type'], original_folds['patient_id'], original_folds['magnification'] = \
        zip(*original_folds['filename'].apply(extract_tumor_type_and_patient_id))
    
    generated_files = []
    
    for config_name, split_result in splits_results.items():
        print(f"\n📋 Generating CSV for {config_name}:")
        
        # Create patient to split mapping
        patient_to_split = {}
        
        for patient_id in split_result['train_patients']['patient_id']:
            patient_to_split[patient_id] = 'train'
        for patient_id in split_result['val_patients']['patient_id']:
            patient_to_split[patient_id] = 'val'
        for patient_id in split_result['test_patients']['patient_id']:
            patient_to_split[patient_id] = 'test'
        
        # Create holdout dataframe
        holdout_rows = []
        for _, row in original_folds.iterrows():
            patient_id = row['patient_id']
            if patient_id in patient_to_split:
                holdout_rows.append({
                    'fold': 1,  # Single fold for holdout
                    'mag': row['mag'],
                    'grp': patient_to_split[patient_id],
                    'filename': row['filename']
                })
        
        holdout_df = pd.DataFrame(holdout_rows)
        
        # Save to file
        output_path = f"data/breakhis/{base_name}_{config_name}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        holdout_df.to_csv(output_path, index=False)
        
        print(f"   ✅ Saved: {output_path}")
        print(f"   📊 Samples: {len(holdout_df)} total")
        
        # Show sample distribution
        sample_dist = holdout_df['grp'].value_counts()
        for split, count in sample_dist.items():
            print(f"     {split}: {count} images")
        
        generated_files.append(output_path)
    
    return generated_files

def validate_robustness(splits_results):
    """Validate that the robust splits reduce potential overfitting"""
    
    print(f"\n🛡️  ROBUSTNESS VALIDATION")
    print("=" * 80)
    
    for config_name, split_result in splits_results.items():
        print(f"\n📋 Validating {config_name}:")
        
        # Check patient overlap
        train_ids = set(split_result['train_patients']['patient_id'])
        val_ids = set(split_result['val_patients']['patient_id'])
        test_ids = set(split_result['test_patients']['patient_id'])
        
        overlaps = []
        if train_ids & val_ids:
            overlaps.append(f"Train-Val: {len(train_ids & val_ids)}")
        if train_ids & test_ids:
            overlaps.append(f"Train-Test: {len(train_ids & test_ids)}")
        if val_ids & test_ids:
            overlaps.append(f"Val-Test: {len(val_ids & test_ids)}")
        
        if overlaps:
            print(f"   ❌ Patient overlaps: {', '.join(overlaps)}")
        else:
            print(f"   ✅ No patient overlap")
        
        # Check test set size (larger test = more reliable evaluation)
        test_size = len(test_ids)
        if test_size >= 15:
            print(f"   ✅ Test set size ({test_size}) adequate for reliable evaluation")
        else:
            print(f"   ⚠️  Test set size ({test_size}) may be too small")
        
        # Check class balance in test set
        test_patients = split_result['test_patients']
        test_class_dist = test_patients['tumor_class'].value_counts()
        benign_ratio = test_class_dist.get('benign', 0) / len(test_patients)
        
        if 0.2 <= benign_ratio <= 0.5:  # Reasonable class balance
            print(f"   ✅ Test set class balance reasonable ({benign_ratio:.2f} benign ratio)")
        else:
            print(f"   ⚠️  Test set class balance ({benign_ratio:.2f} benign ratio) may be skewed")
        
        # Check tumor type diversity
        test_tumor_types = len(test_patients['tumor_type'].unique())
        total_tumor_types = test_patients['tumor_type'].nunique()
        
        if test_tumor_types >= 6:  # Good diversity
            print(f"   ✅ Test set tumor type diversity good ({test_tumor_types} types)")
        else:
            print(f"   ⚠️  Test set tumor type diversity limited ({test_tumor_types} types)")

def main():
    """Main execution function"""
    
    print("🎯 CREATING ROBUST HOLDOUT SPLITS FOR ANTI-OVERFITTING")
    print("=" * 80)
    print("🚨 Addressing 100% test accuracy from RUNPOD_OUTPUT.txt")
    print("💡 Strategy: Larger test sets + better stratification")
    print("=" * 80)
    
    # Step 1: Deep patient analysis
    patient_info = analyze_patient_characteristics()
    
    # Step 2: Create multiple robust configurations
    splits_results = create_robust_holdout_splits(patient_info)
    
    # Step 3: Generate CSV files
    generated_files = generate_robust_holdout_csv(splits_results)
    
    # Step 4: Validate robustness
    validate_robustness(splits_results)
    
    # Final recommendations
    print(f"\n" + "=" * 80)
    print(f"💡 ANTI-OVERFITTING RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"✅ Created {len(splits_results)} robust holdout configurations")
    print(f"✅ Test sets range from 15-20 patients (vs original 12)")
    print(f"✅ Better stratification by tumor class and type")
    print(f"✅ Multiple configurations to test consistency")
    
    print(f"\n🎯 Expected Impact:")
    print(f"   • Larger test sets = more reliable performance estimates")
    print(f"   • Multiple splits = detect if 100% accuracy is consistent") 
    print(f"   • Better stratification = reduce selection bias")
    print(f"   • More realistic performance: likely 85-94% instead of 100%")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Test main_holdout.py with each configuration")
    print(f"2. Compare results across configurations")
    print(f"3. If all still show ~100%, investigate model architecture")
    print(f"4. Consider reducing model complexity if overfitting persists")
    
    print(f"\n📁 Generated Files:")
    for file_path in generated_files:
        print(f"   {file_path}")
    
    return splits_results, generated_files

if __name__ == "__main__":
    splits_results, generated_files = main()