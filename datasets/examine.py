import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from config import FOLD_PATH, SLIDES_PATH
from datasets.preprocess import create_balanced_sampling_weights
from utils.helpers import get_base_path

# Set up paths
# BASE_PATH = '/kaggle/input/breakhis'


# First, let's understand the dataset structure
def explore_dataset():
    stats = defaultdict(lambda: defaultdict(int))
    
    # Count images per category
    for class_type in ['benign', 'malignant']:
        class_path = os.path.join(SLIDES_PATH, class_type, 'SOB')

        if os.path.exists(class_path):
            for tumor_type in os.listdir(class_path):
                tumor_path = os.path.join(class_path, tumor_type)

                for patient_id in os.listdir(tumor_path):
                    patient_path = os.path.join(tumor_path, patient_id)

                    for mag in ['40X', '100X', '200X', '400X']:
                        mag_path = os.path.join(patient_path, mag)
                        if os.path.exists(mag_path):
                            num_images = len([f for f in os.listdir(mag_path) if f.endswith('.png')])
                            stats[f'{class_type}_{tumor_type}_{mag}']['images'] += num_images
                            stats[f'{class_type}_{tumor_type}_{mag}']['patients'] += 1

    # Display statistics
    print("Dataset Statistics:")
    print("-" * 80)
    print(f"{'Category':<40} {'Images':<10} {'Patients':<10}")
    print("-" * 80)

    for key, value in sorted(stats.items()):
        print(f"{key:<40} {value['images']:<10} {value['patients']:<10}")

    return stats

def examine_folds():
    folds_df = pd.read_csv(FOLD_PATH)
    print("\nFolds.csv Info:")
    print(f"Shape: {folds_df.shape}")
    print(f"\nColumns: {list(folds_df.columns)}")
    print(f"\nFirst few rows:")
    print(folds_df.head())

    # Check fold distribution
    print("\nFold distribution:")
    print(folds_df['fold'].value_counts().sort_index())

    # Check magnification distribution
    print("\nMagnification distribution:")
    print(folds_df['mag'].value_counts())

    # Check train/test split
    print("\nTrain/Test distribution:")
    print(folds_df['grp'].value_counts())

    # Plot fold distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=folds_df, x='fold', order=sorted(folds_df['fold'].unique()))
    plt.title('Sample Distribution by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    # plt.show()
    # save the figure if needed
    plt.savefig('figs/fold_distribution.png')
    plt.close()

    # Plot magnification distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=folds_df, x='mag', order=sorted(folds_df['mag'].unique()))
    plt.title('Sample Distribution by Magnification')
    plt.xlabel('Magnification')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    # plt.show()
    # save the figure if needed
    plt.savefig('figs/magnification_distribution.png')
    plt.close()

    # Plot train/test distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=folds_df, x='grp')
    plt.title('Train/Test Distribution')
    plt.xlabel('Group')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    # plt.show()
    # save the figure if needed
    plt.savefig('figs/train_test_distribution.png')
    plt.close()

    return folds_df


# Fix the patient ID extraction
def extract_tumor_type_and_patient_id(filename):
    """Extract tumor type and patient ID from filename"""
    parts = filename.split('/')
    
    # Expected format: BreaKHis_v1/histology_slides/breast/[class]/[tumor_type]/[patient_id]/[magnification]/[image.png]
    if len(parts) >= 8:
        tumor_class = parts[3]  # benign or malignant
        tumor_type = parts[5]   # adenosis, fibroadenoma, etc.
        patient_id = parts[6]   # SOB_B_A_14-22549AB, etc.
        magnification = parts[7].replace('X', '')  # 40, 100, 200, 400
        
        return tumor_class, tumor_type, patient_id, magnification
    return None, None, None, None




# Run exploration
explore_dataset()
folds_df = examine_folds()

print(folds_df)



folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
    zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))

folds_df['magnification'] = pd.to_numeric(folds_df['magnification'], errors='coerce')

print("Dataset structure analysis:")
print(f"Unique patients: {folds_df['patient_id'].nunique()}")
print(f"\nTumor types distribution:")
print(folds_df.groupby(['tumor_class', 'tumor_type']).size())

# Check patient-magnification availability
patient_mag_availability = folds_df.groupby('patient_id')['magnification'].apply(set).apply(len)
print(f"\nPatients by number of magnifications available:")
print(patient_mag_availability.value_counts().sort_index())

# Show distribution of images per patient-magnification combination
print("\nImages per patient-magnification (sample):")
sample_counts = folds_df.groupby(['patient_id', 'magnification']).size().head(20)
print(sample_counts)

# Add barplot for tumor types by class
plt.figure(figsize=(10, 6))
sns.countplot(data=folds_df, y='tumor_type', hue='tumor_class', order=folds_df['tumor_type'].value_counts().index)
plt.title('Tumor Type Distribution by Class')
plt.xlabel('Number of Samples')
plt.ylabel('Tumor Type')
plt.legend(title='Tumor Class')
plt.tight_layout()
# plt.show()
plt.savefig('figs/tumor_type_distribution.png')
plt.close()



def analyze_data_balance(folds_df, fold=1):
    """Analyze data balance and distribution issues"""
    
    # Get fold data
    fold_df = folds_df[folds_df['fold'] == fold].copy()
    
    # 1. Class distribution analysis
    print("=== CLASS DISTRIBUTION ANALYSIS ===")
    class_dist = fold_df.groupby(['grp', 'tumor_class']).size().unstack(fill_value=0)
    print("\nClass distribution by split:")
    print(class_dist)
    print(f"\nClass ratio (benign:malignant) in train: {class_dist.loc['train', 'benign']:.0f}:{class_dist.loc['train', 'malignant']:.0f}")
    
    # 2. Tumor type distribution
    print("\n=== TUMOR TYPE DISTRIBUTION ===")
    tumor_dist = fold_df[fold_df['grp'] == 'train'].groupby(['tumor_class', 'tumor_type']).size()
    print(tumor_dist)
    
    # 3. Patient-level analysis
    print("\n=== PATIENT-LEVEL ANALYSIS ===")
    patient_class = fold_df.groupby(['patient_id', 'tumor_class']).size().reset_index()[['patient_id', 'tumor_class']]
    patient_class_dist = patient_class['tumor_class'].value_counts()
    print(f"Unique patients - Benign: {patient_class_dist.get('benign', 0)}, Malignant: {patient_class_dist.get('malignant', 0)}")
    
    # 4. Images per patient analysis
    imgs_per_patient = fold_df[fold_df['grp'] == 'train'].groupby(['patient_id', 'tumor_class']).size()
    print(f"\nImages per patient stats:")
    print(f"Mean: {imgs_per_patient.mean():.1f}, Std: {imgs_per_patient.std():.1f}")
    
    # 5. Magnification balance
    print("\n=== MAGNIFICATION BALANCE ===")
    mag_dist = fold_df[fold_df['grp'] == 'train'].groupby(['tumor_class', 'mag']).size().unstack(fill_value=0)
    print(mag_dist)
    
    return fold_df

# Add this to your fold processing
for fold in range(1, 6):  # Just fold 1 for analysis
    print(f"\n{'='*60}")
    print(f"FOLD {fold} ANALYSIS")
    print(f"{'='*60}")
    
    # Analyze balance
    fold_df = analyze_data_balance(folds_df, fold)
    
    # Get sampling weights
    class_weights, tumor_weights, patient_weights = create_balanced_sampling_weights(fold_df)
    
    # Get tensor weights for loss function
    # tumor_class_weights = get_balanced_tumor_weights(fold_df, device)
    
    # Visualize class distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    fold_df[fold_df['grp'] == 'train']['tumor_class'].value_counts().plot(kind='bar')
    plt.title('Class Distribution (Train)')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 3, 2)
    fold_df[fold_df['grp'] == 'train']['tumor_type'].value_counts().plot(kind='bar')
    plt.title('Tumor Type Distribution (Train)')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 3, 3)
    fold_df[fold_df['grp'] == 'train'].groupby(['mag', 'tumor_class']).size().unstack().plot(kind='bar')
    plt.title('Magnification by Class')
    plt.xlabel('Magnification')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('figs/magnification_by_class.png')
    plt.close()

    break
    