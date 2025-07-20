from torchvision import transforms
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random

class MedicalAugmentation:
    """Advanced medical image augmentations for histology"""
    
    @staticmethod
    def elastic_transform(image, alpha=1, sigma=50, alpha_affine=50):
        """Apply elastic deformation"""
        if random.random() < 0.3:  # 30% chance
            image_array = np.array(image)
            shape = image_array.shape
            shape_size = shape[:2]
            
            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
            pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            
            # Apply slight deformation
            if len(image_array.shape) == 3:
                deformed = image_array + np.random.normal(0, alpha, image_array.shape) * 0.1
                deformed = np.clip(deformed, 0, 255).astype(np.uint8)
                return Image.fromarray(deformed)
        return image
    
    @staticmethod
    def add_gaussian_noise(image, std=0.02):
        """Add Gaussian noise"""
        if random.random() < 0.4:  # 40% chance
            image_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, std, image_array.shape)
            noisy = image_array + noise
            noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy)
        return image
    
    @staticmethod
    def histogram_equalization(image):
        """Apply histogram equalization with probability"""
        if random.random() < 0.25:  # 25% chance
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(random.uniform(0.8, 1.3))
        return image
    
    @staticmethod
    def random_erasing(image, probability=0.3, sl=0.02, sh=0.08, r1=0.3):
        """Random erasing augmentation"""
        if random.random() > probability:
            return image
        
        image_array = np.array(image)
        area = image_array.shape[0] * image_array.shape[1]
        
        for _ in range(100):
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < image_array.shape[1] and h < image_array.shape[0]:
                x1 = random.randint(0, image_array.shape[0] - h)
                y1 = random.randint(0, image_array.shape[1] - w)
                
                # Fill with random color
                image_array[x1:x1+h, y1:y1+w] = np.random.randint(0, 255, (h, w, 3))
                break
        
        return Image.fromarray(image_array)

def get_transforms(mode='train', img_size=224):
    """Get enhanced image transforms for medical imaging"""
    
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),  # Resize larger for random crop
            transforms.RandomCrop((img_size, img_size)),
            
            # Medical-specific augmentations
            transforms.Lambda(MedicalAugmentation.elastic_transform),
            transforms.Lambda(MedicalAugmentation.add_gaussian_noise),
            transforms.Lambda(MedicalAugmentation.histogram_equalization),
            
            # Standard augmentations with more variety
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=255),  # White fill for medical images
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1), 
                shear=5, 
                fill=255
            ),
            
            # Color augmentations more suitable for histology
            transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.2, 
                hue=0.05
            ),
            
            # Random perspective for 3D-like effects
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=255),
            
            transforms.Lambda(MedicalAugmentation.random_erasing),
            
            transforms.ToTensor(),
            
            # Stain normalization can be added here if available
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Random erasing after normalization
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
        ])
    else:
        # Test-time augmentation for validation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def get_tta_transforms(img_size=224):
    """Get test-time augmentation transforms"""
    tta_transforms = []
    
    # Original
    tta_transforms.append(transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Horizontal flip
    tta_transforms.append(transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Vertical flip
    tta_transforms.append(transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Slight rotation
    tta_transforms.append(transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    return tta_transforms

#/kaggle/input/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-027.png