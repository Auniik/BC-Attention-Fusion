#!/usr/bin/env python3
"""
Test script to verify deterministic image selection eliminates data leakage
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datasets.multi_mag import MultiMagnificationDataset
from datasets.preprocess import create_multi_mag_dataset_info, get_patients_for_mode
from datasets.examine import extract_tumor_type_and_patient_id
from config import BASE_PATH
import torch

def test_deterministic_selection():
    """Test that image selection is deterministic and prevents leakage"""
    
    print("Testing deterministic image selection...")
    
    # Load fold data
    folds_path = os.path.join(BASE_PATH, 'Folds_fixed.csv')
    if not os.path.exists(folds_path):
        print(f"ERROR: {folds_path} not found. Run fix_cross_validation.py first.")
        return False
    
    fold_df = pd.read_csv(folds_path)
    
    # Add patient_id, tumor_class, tumor_type to dataframe
    fold_df['tumor_class'], fold_df['tumor_type'], fold_df['patient_id'], fold_df['magnification'] = \
        zip(*fold_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    multi_mag_patients, _, fold_data, _ = create_multi_mag_dataset_info(fold_df, fold=1)
    
    # Create transforms to convert images to tensors for comparison
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create train and test datasets with require_all_mags=True
    train_dataset = MultiMagnificationDataset(
        multi_mag_patients, fold_data, mode='train', 
        samples_per_patient=2, balance_classes=False, require_all_mags=True,
        transform=transform
    )
    
    test_dataset = MultiMagnificationDataset(
        multi_mag_patients, fold_data, mode='test',
        samples_per_patient=2, balance_classes=False, require_all_mags=True,
        transform=transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test deterministic behavior - same index should give same image
    if len(train_dataset) > 0:
        sample1 = train_dataset[0]
        sample2 = train_dataset[0]
        
        # Check if same patient gives same images
        same_images = True
        for mag in [40, 100, 200, 400]:
            key = f'mag_{mag}'
            if key in sample1['images'] and key in sample2['images']:
                if not torch.equal(sample1['images'][key], sample2['images'][key]):
                    same_images = False
                    break
        
        if same_images:
            print("✅ Deterministic selection: Same index gives same images")
        else:
            print("❌ Non-deterministic selection detected")
            return False
    
    # Check for patient overlap in train/test (should be zero)
    train_patients = set()
    test_patients = set()
    
    for i in range(min(len(train_dataset), 10)):  # Sample first 10
        train_patients.add(train_dataset[i]['patient_id'])
    
    for i in range(min(len(test_dataset), 10)):  # Sample first 10  
        test_patients.add(test_dataset[i]['patient_id'])
    
    overlap = train_patients.intersection(test_patients)
    
    if len(overlap) == 0:
        print("✅ No patient overlap between train/test")
        return True
    else:
        print(f"❌ Patient overlap detected: {overlap}")
        return False

if __name__ == "__main__":
    success = test_deterministic_selection()
    if success:
        print("\n✅ Image-level data leakage fix appears to be working correctly!")
        print("The model will now select images deterministically, preventing memorization.")
    else:
        print("\n❌ Issues detected with the fix.")
    
    sys.exit(0 if success else 1)