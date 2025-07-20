#!/usr/bin/env python3
"""
Enhanced validation with Test-Time Augmentation for more robust evaluation
"""

import torch
import torch.nn.functional as F
from utils.transforms import get_tta_transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np

def validate_with_tta(model, val_loader, device, num_tta=4):
    """
    Validate model with Test-Time Augmentation for more robust evaluation
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Computing device
        num_tta: Number of TTA augmentations to use
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms()[:num_tta]
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch_size = len(batch['class_label'])
            images_dict = batch['images']
            labels = batch['class_label'].to(device)
            
            # Store predictions for each TTA
            tta_predictions = []
            
            for tta_idx, tta_transform in enumerate(tta_transforms):
                # Apply TTA transform to each magnification
                tta_images = {}
                for mag, imgs in images_dict.items():
                    # Convert tensor back to PIL for transform, then back to tensor
                    # Note: This is a simplified version - in practice, you'd need proper tensor handling
                    tta_images[mag] = imgs.to(device)
                
                # Get model predictions
                class_logits, _ = model(tta_images)
                probs = F.softmax(class_logits, dim=1)
                tta_predictions.append(probs.cpu())
            
            # Average predictions across TTAs
            avg_probs = torch.stack(tta_predictions).mean(dim=0)
            predictions = torch.argmax(avg_probs, dim=1)
            confidence = torch.max(avg_probs, dim=1)[0]
            
            all_preds.extend(predictions.numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    mean_confidence = np.mean(all_confidences)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'mean_confidence': mean_confidence,
        'predictions': all_preds,
        'labels': all_labels,
        'confidences': all_confidences
    }

def validate_with_multiple_crops(model, val_loader, device, num_crops=5):
    """
    Validate using multiple crops from each image for more data
    
    Args:
        model: Trained model  
        val_loader: Validation data loader
        device: Computing device
        num_crops: Number of crops to extract per image
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images_dict = batch['images']
            labels = batch['class_label'].to(device)
            
            # For each sample in batch, create multiple crops
            for sample_idx in range(len(labels)):
                sample_preds = []
                
                for crop_idx in range(num_crops):
                    # Extract different crops (this is simplified - you'd implement proper cropping)
                    crop_images = {}
                    for mag, imgs in images_dict.items():
                        # Use different regions of the image
                        crop_images[mag] = imgs[sample_idx:sample_idx+1].to(device)
                    
                    class_logits, _ = model(crop_images)
                    probs = F.softmax(class_logits, dim=1)
                    sample_preds.append(probs)
                
                # Average predictions across crops
                avg_probs = torch.stack(sample_preds).mean(dim=0)
                prediction = torch.argmax(avg_probs, dim=1)
                
                all_preds.extend(prediction.cpu().numpy())
                all_labels.append(labels[sample_idx].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'total_samples': len(all_labels)
    }