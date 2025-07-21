import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

from config import BASE_PATH


class MultiMagnificationDataset(Dataset):    
    def __init__(self, patient_data, fold_df, mode='train', mags=[40, 100, 200, 400], 
                 samples_per_patient=4, transform=None, balance_classes=True, 
                 require_all_mags=True):
        self.patient_data = patient_data
        self.fold_df = fold_df
        self.mode = mode
        self.mags = mags
        self.samples_per_patient = samples_per_patient
        self.transform = transform
        self.require_all_mags = require_all_mags
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
                # Filter images to only include those in current mode/fold
                mode_images = {}
                for mag in self.mags:
                    # Compare with both int and string versions to handle type variations
                    mag_files = patient_fold_data[
                        (patient_fold_data['magnification'] == mag) | 
                        (patient_fold_data['magnification'] == str(mag))
                    ]['filename'].tolist()
                    if mag_files:
                        mode_images[mag] = mag_files
                
                # Check if patient meets magnification requirements
                if self.require_all_mags:
                    # Only include if patient has ALL required magnifications
                    if set(mode_images.keys()) == set(self.mags):
                        # Create sample entries
                        for _ in range(self.samples_per_patient):
                            self.samples.append({
                                'patient_id': patient_id,
                                'tumor_class': patient['tumor_class'],
                                'tumor_type': patient['tumor_type'],
                                'images': mode_images
                            })
                else:
                    # Include if patient has at least some magnifications
                    if mode_images:
                        # Create sample entries
                        for _ in range(self.samples_per_patient):
                            self.samples.append({
                                'patient_id': patient_id,
                                'tumor_class': patient['tumor_class'],
                                'tumor_type': patient['tumor_type'],
                                'images': mode_images
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
        
        # Handle case where one class is missing
        if len(benign_samples) == 0:
            print(f"No benign samples found in {self.mode} mode - using only malignant samples")
            self.samples = malignant_samples
            return
        
        if len(malignant_samples) == 0:
            print(f"No malignant samples found in {self.mode} mode - using only benign samples")
            self.samples = benign_samples
            return
        
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

        # For each magnification, deterministically select image to prevent data leakage
        for mag in self.mags:
            if mag in sample['images'] and sample['images'][mag]:
                # Sort images for deterministic selection, then use modulo for consistent selection
                sorted_images = sorted(sample['images'][mag])
                img_idx = (idx + hash(sample['patient_id'])) % len(sorted_images)
                img_path = sorted_images[img_idx]
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
                if not self.require_all_mags:
                    # Use fallback strategy: try to use closest available magnification
                    available_mags = list(sample['images'].keys())
                    if available_mags:
                        # Find closest magnification
                        closest_mag = min(available_mags, key=lambda x: abs(x - mag))
                        print(f"Missing magnification {mag} for patient {sample['patient_id']}, using {closest_mag}x as fallback")
                        # Use deterministic selection for fallback too
                        sorted_images = sorted(sample['images'][closest_mag])
                        img_idx = (idx + hash(sample['patient_id'])) % len(sorted_images)
                        img_path = sorted_images[img_idx]
                        full_path = os.path.join(self.base_path, img_path)
                        
                        try:
                            image = Image.open(full_path).convert('RGB')
                            if self.transform:
                                image = self.transform(image)
                            images[f'mag_{mag}'] = image
                        except Exception as e:
                            print(f"Error loading fallback image: {full_path} ({e})")
                            images[f'mag_{mag}'] = torch.zeros((3, 224, 224))
                    else:
                        print(f"No images available for patient {sample['patient_id']}")
                        images[f'mag_{mag}'] = torch.zeros((3, 224, 224))
                else:
                    # This shouldn't happen if require_all_mags=True and filtering worked correctly
                    print(f"ERROR: Missing magnification {mag} for patient {sample['patient_id']} (require_all_mags=True)")
                    images[f'mag_{mag}'] = torch.zeros((3, 224, 224))

        class_label = self.class_to_idx[sample['tumor_class']]
        tumor_type_label = self.tumor_type_to_idx[sample['tumor_type']]

        return {
            'images': images,
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'tumor_type_label': torch.tensor(tumor_type_label, dtype=torch.long),
            'patient_id': sample['patient_id']
        }