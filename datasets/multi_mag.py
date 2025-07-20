import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

from config import BASE_PATH


class MultiMagnificationDataset(Dataset):    
    def __init__(self, patient_data, fold_df, mode='train', mags=[40, 100, 200, 400], 
                 samples_per_patient=4, transform=None, balance_classes=True):
        self.patient_data = patient_data
        self.fold_df = fold_df
        self.mode = mode
        self.mags = mags
        self.samples_per_patient = samples_per_patient
        self.transform = transform
        self.base_path = BASE_PATH + '/BreaKHis_v1'
        self.rng = np.random.default_rng(seed=42)

        # Filter patients based on mode
        self.samples = []
        for patient in patient_data:
            patient_id = patient['patient_id']

            # Get all images for this patient in current mode
            patient_fold_data = fold_df[
                (fold_df['patient_id'] == patient_id) & 
                (fold_df['grp'] == mode)
            ]

            if len(patient_fold_data) > 0:
                # Create sample entries
                for _ in range(self.samples_per_patient):
                    self.samples.append({
                        'patient_id': patient_id,
                        'tumor_class': patient['tumor_class'],
                        'tumor_type': patient['tumor_type'],
                        'images': patient['images']
                    })

        if balance_classes and mode == 'train':
            self._create_balanced_samples()

        # Create label mapping
        self.class_to_idx = {'benign': 0, 'malignant': 1}

        # Create tumor type mapping for fine-grained classification
        tumor_types = sorted(fold_df['tumor_type'].unique())
        self.tumor_type_to_idx = {t: i for i, t in enumerate(tumor_types)}

        print(f"Dataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Patients: {len(patient_data)}")
        print(f"  Samples per epoch: {len(self.samples)}")
        print(f"  Magnifications: {self.mags}")

    def _create_balanced_samples(self):
        """Balance samples by oversampling minority class"""
        benign_samples = [s for s in self.samples if s['tumor_class'] == 'benign']
        malignant_samples = [s for s in self.samples if s['tumor_class'] == 'malignant']
        
        # Calculate how many times to repeat benign samples
        repeat_factor = len(malignant_samples) // len(benign_samples)
        remainder = len(malignant_samples) % len(benign_samples)
        
        # Oversample benign class
        balanced_samples = malignant_samples.copy()
        balanced_samples.extend(benign_samples * repeat_factor)
        if remainder > 0:
            balanced_samples.extend(self.rng.choice(benign_samples, size=remainder, replace=False).tolist())
        
        # Shuffle
        self.rng.shuffle(balanced_samples)
        self.samples = balanced_samples
        
        # Fixed print statement
        total_benign = len(benign_samples) * repeat_factor + remainder
        print(f"Balanced dataset: {total_benign} benign, {len(malignant_samples)} malignant")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Dictionary to store images from different magnifications
        images = {}

        # For each magnification, randomly select one image
        for mag in self.mags:
            if mag in sample['images'] and sample['images'][mag]:
                img_path = self.rng.choice(sample['images'][mag])
                full_path = os.path.join(self.base_path, img_path)

                try:
                    image = Image.open(full_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading: {full_path} ({e})")
                    image = Image.new('RGB', (700, 460), color='white')

                if self.transform:
                    image = self.transform(image)

                images[f'mag_{mag}'] = image
            else:
                print(f"Missing magnification {mag} for patient {sample['patient_id']}")
                images[f'mag_{mag}'] = torch.zeros((3, 224, 224))  # dummy tensor

        class_label = self.class_to_idx[sample['tumor_class']]
        tumor_type_label = self.tumor_type_to_idx[sample['tumor_type']]

        return {
            'images': images,
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'tumor_type_label': torch.tensor(tumor_type_label, dtype=torch.long),
            'patient_id': sample['patient_id']
        }