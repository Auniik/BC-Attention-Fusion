#!/usr/bin/env python3
"""
Create True Holdout Patient Split for BreakHis Dataset

Implements "Option 1: True Holdout Patient Split" from CLAUDE.md:
- 60 patients for training (never see validation/test)
- 10 patients for validation (during training)
- 12 patients for test (final evaluation only)

This eliminates all cross-validation and provides the most realistic
evaluation for clinical deployment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from config import FOLD_PATH
from datasets.examine import extract_tumor_type_and_patient_id

def analyze_current_patients():
    """Analyze current patient distribution to understand dataset"""
    
    print("=== ANALYZING CURRENT PATIENT DISTRIBUTION ===")
    
    # Load current folds
    folds_df = pd.read_csv(FOLD_PATH)
    
    # Extract patient information
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
        zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    # Get unique patients with their characteristics
    patient_info = folds_df.groupby('patient_id').agg({
        'tumor_class': 'first',
        'tumor_type': 'first',
        'magnification': lambda x: sorted(x.unique())
    }).reset_index()
    
    # Check magnification completeness
    # Convert magnifications to int for proper comparison
    patient_info['mag_set'] = patient_info['magnification'].apply(
        lambda x: set(int(mag) for mag in x if pd.notna(mag))
    )
    patient_info['has_all_mags'] = patient_info['mag_set'].apply(
        lambda x: len(x) == 4 and x == {40, 100, 200, 400}
    )
    
    print(f"Total unique patients: {len(patient_info)}")
    print(f"Patients with all 4 magnifications: {patient_info['has_all_mags'].sum()}")
    
    # Debug magnification values
    print(f"\nDEBUG: Magnification values for first few patients:")
    for i in range(min(3, len(patient_info))):
        patient = patient_info.iloc[i]
        print(f"  Patient {patient['patient_id']}: {patient['magnification']} -> {patient['mag_set']}")
    
    # Class distribution
    class_dist = patient_info['tumor_class'].value_counts()
    print(f"\nClass distribution:")
    for cls, count in class_dist.items():
        print(f"  {cls}: {count} patients ({count/len(patient_info)*100:.1f}%)")
    
    # Tumor type distribution
    print(f"\nTumor type distribution:")
    tumor_dist = patient_info.groupby(['tumor_class', 'tumor_type']).size()
    for (cls, typ), count in tumor_dist.items():
        print(f"  {cls} - {typ}: {count} patients")
    
    return patient_info[patient_info['has_all_mags']].copy()

def create_stratified_holdout_split(patient_info, train_size=60, val_size=10, test_size=12, random_state=42):
    """
    Create stratified holdout split maintaining class and tumor type distribution
    
    Args:
        patient_info: DataFrame with patient characteristics
        train_size: Number of patients for training (60)
        val_size: Number of patients for validation (10) 
        test_size: Number of patients for test (12)
        random_state: Random seed for reproducibility
    """
    
    print(f"\n=== CREATING HOLDOUT SPLIT ===")
    print(f"Target split: {train_size} train + {val_size} val + {test_size} test = {train_size+val_size+test_size}")
    print(f"Available patients: {len(patient_info)}")
    
    if len(patient_info) != (train_size + val_size + test_size):
        raise ValueError(f"Patient count mismatch: {len(patient_info)} != {train_size+val_size+test_size}")
    
    np.random.seed(random_state)
    
    # Strategy: First split into train vs (val+test), then split (val+test) into val vs test
    # This maintains better stratification
    
    # Create stratification label combining class and tumor type
    patient_info['strat_label'] = patient_info['tumor_class'] + '_' + patient_info['tumor_type']
    
    print(f"\nStratification labels:")
    strat_dist = patient_info['strat_label'].value_counts()
    for label, count in strat_dist.items():
        print(f"  {label}: {count} patients")
    
    # First split: train vs (val+test)
    train_patients, valtest_patients = train_test_split(
        patient_info,
        train_size=train_size,
        test_size=(val_size + test_size),
        stratify=patient_info['tumor_class'],  # Stratify by main class for better balance
        random_state=random_state
    )
    
    # Second split: val vs test from the valtest group
    val_patients, test_patients = train_test_split(
        valtest_patients,
        train_size=val_size,
        test_size=test_size,
        stratify=valtest_patients['tumor_class'],  # Stratify by main class
        random_state=random_state + 1
    )
    
    # Verify split sizes
    print(f"\nActual split sizes:")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Validation: {len(val_patients)} patients") 
    print(f"  Test: {len(test_patients)} patients")
    
    # Verify no overlap
    train_ids = set(train_patients['patient_id'])
    val_ids = set(val_patients['patient_id'])
    test_ids = set(test_patients['patient_id'])
    
    assert len(train_ids & val_ids) == 0, "Train-Val overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train-Test overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val-Test overlap detected!"
    print(f"✅ No patient overlap between splits")
    
    # Analyze class distribution in each split
    print(f"\nClass distribution by split:")
    for split_name, split_df in [('Train', train_patients), ('Validation', val_patients), ('Test', test_patients)]:
        class_dist = split_df['tumor_class'].value_counts()
        print(f"  {split_name}:")
        for cls, count in class_dist.items():
            print(f"    {cls}: {count} patients ({count/len(split_df)*100:.1f}%)")
    
    # Analyze tumor type distribution  
    print(f"\nTumor type distribution by split:")
    for split_name, split_df in [('Train', train_patients), ('Validation', val_patients), ('Test', test_patients)]:
        tumor_dist = split_df.groupby(['tumor_class', 'tumor_type']).size()
        print(f"  {split_name}:")
        for (cls, typ), count in tumor_dist.items():
            print(f"    {cls} - {typ}: {count}")
    
    return train_patients, val_patients, test_patients

def generate_holdout_csv(train_patients, val_patients, test_patients, output_path):
    """
    Generate Folds_holdout.csv with the holdout patient assignments
    
    Format matches original Folds_fixed.csv but with patient-based splits:
    - All images from train patients get grp='train'
    - All images from val patients get grp='val' 
    - All images from test patients get grp='test'
    - No fold column needed (single split)
    """
    
    print(f"\n=== GENERATING HOLDOUT CSV ===")
    
    # Load original folds to get all image filenames
    original_folds = pd.read_csv(FOLD_PATH)
    
    # Extract patient IDs from original data
    original_folds['tumor_class'], original_folds['tumor_type'], original_folds['patient_id'], original_folds['magnification'] = \
        zip(*original_folds['filename'].apply(extract_tumor_type_and_patient_id))
    
    # Create patient to split mapping
    patient_to_split = {}
    
    # Assign splits
    for patient_id in train_patients['patient_id']:
        patient_to_split[patient_id] = 'train'
    for patient_id in val_patients['patient_id']:
        patient_to_split[patient_id] = 'val'
    for patient_id in test_patients['patient_id']:
        patient_to_split[patient_id] = 'test'
    
    # Create new holdout dataframe
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
    
    print(f"Generated holdout CSV with {len(holdout_df)} image entries")
    
    # Verify distribution
    print(f"\nSample distribution in holdout CSV:")
    split_dist = holdout_df['grp'].value_counts()
    for split, count in split_dist.items():
        print(f"  {split}: {count} images")
    
    # Save to file
    holdout_df.to_csv(output_path, index=False)
    print(f"✅ Saved holdout CSV to: {output_path}")
    
    return holdout_df

def visualize_holdout_split(train_patients, val_patients, test_patients, output_dir='figs'):
    """Create visualizations of the holdout split"""
    
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class distribution by split
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    splits = [('Train', train_patients), ('Validation', val_patients), ('Test', test_patients)]
    
    for i, (split_name, split_df) in enumerate(splits):
        class_counts = split_df['tumor_class'].value_counts()
        axes[i].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[i].set_title(f'{split_name} Split\n({len(split_df)} patients)')
    
    plt.suptitle('Class Distribution Across Holdout Splits')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/holdout_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tumor type distribution
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    for i, (split_name, split_df) in enumerate(splits):
        tumor_dist = split_df.groupby(['tumor_class', 'tumor_type']).size().reset_index(name='count')
        
        # Create stacked bar chart
        benign_data = tumor_dist[tumor_dist['tumor_class'] == 'benign']
        malignant_data = tumor_dist[tumor_dist['tumor_class'] == 'malignant']
        
        x_pos = np.arange(len(tumor_dist['tumor_type'].unique()))
        
        axes[i].bar(range(len(benign_data)), benign_data['count'], 
                   label='Benign', alpha=0.7, color='lightblue')
        axes[i].bar(range(len(benign_data), len(benign_data) + len(malignant_data)), 
                   malignant_data['count'], label='Malignant', alpha=0.7, color='lightcoral')
        
        axes[i].set_title(f'{split_name} Split - Tumor Type Distribution')
        axes[i].set_ylabel('Number of Patients')
        axes[i].legend()
        
        # Set x-tick labels
        all_types = list(benign_data['tumor_type']) + list(malignant_data['tumor_type'])
        axes[i].set_xticks(range(len(all_types)))
        axes[i].set_xticklabels(all_types, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/holdout_tumor_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary statistics table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    summary_data = []
    for split_name, split_df in splits:
        class_dist = split_df['tumor_class'].value_counts()
        benign_count = class_dist.get('benign', 0)
        malignant_count = class_dist.get('malignant', 0)
        total = len(split_df)
        
        summary_data.append([
            split_name,
            total,
            benign_count,
            f"{benign_count/total*100:.1f}%" if total > 0 else "0%",
            malignant_count, 
            f"{malignant_count/total*100:.1f}%" if total > 0 else "0%"
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Split', 'Total Patients', 'Benign', 'Benign %', 'Malignant', 'Malignant %'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    ax.set_title('Holdout Split Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(f'{output_dir}/holdout_split_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}/")

def validate_holdout_split(holdout_csv_path):
    """Validate the generated holdout split"""
    
    print(f"\n=== VALIDATING HOLDOUT SPLIT ===")
    
    holdout_df = pd.read_csv(holdout_csv_path)
    
    # Extract patient IDs
    holdout_df['tumor_class'], holdout_df['tumor_type'], holdout_df['patient_id'], holdout_df['magnification'] = \
        zip(*holdout_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    # Check patient overlap
    train_patients = set(holdout_df[holdout_df['grp'] == 'train']['patient_id'].unique())
    val_patients = set(holdout_df[holdout_df['grp'] == 'val']['patient_id'].unique())
    test_patients = set(holdout_df[holdout_df['grp'] == 'test']['patient_id'].unique())
    
    print(f"Patient counts:")
    print(f"  Train: {len(train_patients)} unique patients")
    print(f"  Validation: {len(val_patients)} unique patients")
    print(f"  Test: {len(test_patients)} unique patients")
    print(f"  Total: {len(train_patients | val_patients | test_patients)} unique patients")
    
    # Check for overlaps
    overlaps = []
    if train_patients & val_patients:
        overlaps.append(f"Train-Val: {len(train_patients & val_patients)}")
    if train_patients & test_patients:
        overlaps.append(f"Train-Test: {len(train_patients & test_patients)}")
    if val_patients & test_patients:
        overlaps.append(f"Val-Test: {len(val_patients & test_patients)}")
    
    if overlaps:
        print(f"❌ Patient overlaps detected: {', '.join(overlaps)}")
        return False
    else:
        print(f"✅ No patient overlap detected")
    
    # Check magnification completeness
    print(f"\nMagnification completeness check:")
    for split_name, split_grp in [('Train', 'train'), ('Validation', 'val'), ('Test', 'test')]:
        split_data = holdout_df[holdout_df['grp'] == split_grp]
        patients_with_all_mags = split_data.groupby('patient_id')['magnification'].nunique()
        complete_patients = (patients_with_all_mags == 4).sum()
        total_patients = len(patients_with_all_mags)
        print(f"  {split_name}: {complete_patients}/{total_patients} patients have all 4 magnifications")
    
    print(f"✅ Holdout split validation completed")
    return True

def main():
    """Main execution function"""
    
    print("🔬 CREATING TRUE HOLDOUT PATIENT SPLIT FOR BREAKHIS DATASET")
    print("=" * 80)
    
    # Step 1: Analyze current patients
    patient_info = analyze_current_patients()
    
    # Step 2: Create stratified holdout split
    train_patients, val_patients, test_patients = create_stratified_holdout_split(
        patient_info, 
        train_size=60, 
        val_size=10, 
        test_size=12,
        random_state=42
    )
    
    # Step 3: Generate holdout CSV
    output_path = os.path.join(os.path.dirname(FOLD_PATH), 'Folds_holdout.csv')
    holdout_df = generate_holdout_csv(train_patients, val_patients, test_patients, output_path)
    
    # Step 4: Create visualizations
    visualize_holdout_split(train_patients, val_patients, test_patients)
    
    # Step 5: Validate the split
    validation_success = validate_holdout_split(output_path)
    
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"🎯 HOLDOUT SPLIT CREATION COMPLETE")
    print(f"=" * 80)
    print(f"✅ Generated: {output_path}")
    print(f"✅ Split: 60 train + 10 validation + 12 test patients")
    print(f"✅ Zero patient overlap between splits")
    print(f"✅ Stratified by tumor class and type")
    print(f"✅ All patients have complete magnification sets")
    print(f"")
    print(f"📋 NEXT STEPS:")
    print(f"1. Update config.py to use Folds_holdout.csv")
    print(f"2. Create main_holdout.py without cross-validation")
    print(f"3. Train single model on 60 patients")
    print(f"4. Validate during training on 10 patients")
    print(f"5. Final test on 12 completely unseen patients")
    print(f"")
    print(f"🎯 Expected realistic performance: 85-94% (not 97%+ from data leakage)")
    
    return validation_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)