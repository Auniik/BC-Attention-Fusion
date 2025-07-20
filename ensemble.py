import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
import torch.nn.functional as F

def predict_with_tta(model, dataloader, device, tta_transforms=None):
    """Make predictions with test-time augmentation"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='TTA Prediction'):
            images_dict = {k: v.to(device, non_blocking=True) for k, v in batch['images'].items()}
            class_labels = batch['class_label'].to(device, non_blocking=True)
            
            if tta_transforms and len(tta_transforms) > 1:
                # Apply multiple TTA transforms
                tta_predictions = []
                
                for transform in tta_transforms:
                    # Apply transform to each magnification
                    transformed_images = {}
                    for mag_key, mag_imgs in images_dict.items():
                        # Convert back to PIL, apply transform, convert to tensor
                        transformed_batch = []
                        for img_tensor in mag_imgs:
                            # Denormalize
                            img_denorm = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                            img_denorm += torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                            img_denorm = torch.clamp(img_denorm, 0, 1)
                            
                            # Convert to PIL and apply transform
                            img_pil = transforms.ToPILImage()(img_denorm.cpu())
                            img_transformed = transform(img_pil)
                            transformed_batch.append(img_transformed)
                        
                        transformed_images[mag_key] = torch.stack(transformed_batch).to(device)
                    
                    # Get prediction
                    class_logits, _ = model(transformed_images)
                    tta_predictions.append(F.softmax(class_logits, dim=1))
                
                # Average TTA predictions
                predictions = torch.stack(tta_predictions).mean(dim=0)
            else:
                # Standard prediction
                class_logits, _ = model(images_dict)
                predictions = F.softmax(class_logits, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(class_labels.cpu())
    
    return torch.cat(all_predictions), torch.cat(all_labels)

def ensemble_predictions(fold_predictions, fold_weights=None):
    """Combine predictions from multiple folds"""
    if fold_weights is None:
        fold_weights = [1.0] * len(fold_predictions)
    
    # Normalize weights
    fold_weights = np.array(fold_weights)
    fold_weights = fold_weights / fold_weights.sum()
    
    # Weighted average of predictions
    ensemble_pred = None
    for i, (pred, weight) in enumerate(zip(fold_predictions, fold_weights)):
        if ensemble_pred is None:
            ensemble_pred = pred * weight
        else:
            ensemble_pred += pred * weight
    
    return ensemble_pred

def evaluate_ensemble(ensemble_predictions, true_labels, confidence_threshold=None):
    """Evaluate ensemble predictions"""
    # Convert to numpy
    if isinstance(ensemble_predictions, torch.Tensor):
        ensemble_predictions = ensemble_predictions.numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.numpy()
    
    # Get predicted classes
    predicted_classes = np.argmax(ensemble_predictions, axis=1)
    prediction_confidence = np.max(ensemble_predictions, axis=1)
    
    # Apply confidence threshold if specified
    if confidence_threshold is not None:
        confident_mask = prediction_confidence >= confidence_threshold
        predicted_classes = predicted_classes[confident_mask]
        true_labels = true_labels[confident_mask]
        
        print(f"Confidence threshold {confidence_threshold}: {confident_mask.sum()}/{len(confident_mask)} samples retained")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_classes)
    balanced_acc = balanced_accuracy_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes, average='weighted')
    
    print(f"Ensemble Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes, 
                              target_names=['Benign', 'Malignant']))
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'predictions': predicted_classes,
        'confidence': prediction_confidence if confidence_threshold is None else prediction_confidence[confident_mask]
    }

def run_ensemble_evaluation(models, val_loader, device, fold_weights=None):
    """Run complete ensemble evaluation pipeline"""
    print("🔥 Running Multi-Fold Ensemble Evaluation")
    print("=" * 60)
    
    fold_predictions = []
    true_labels = None
    
    # Get predictions from each fold
    for fold_idx, model in enumerate(models):
        print(f"\n📊 Evaluating Fold {fold_idx + 1}")
        predictions, labels = predict_with_tta(model, val_loader, device)
        fold_predictions.append(predictions)
        
        if true_labels is None:
            true_labels = labels
    
    # Ensemble predictions
    print(f"\n🎯 Combining {len(fold_predictions)} fold predictions...")
    ensemble_pred = ensemble_predictions(fold_predictions, fold_weights)
    
    # Evaluate ensemble
    results = evaluate_ensemble(ensemble_pred, true_labels)
    
    # Also evaluate with confidence thresholding
    print(f"\n🎯 Confidence-based results:")
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        print(f"\n--- Confidence >= {threshold} ---")
        evaluate_ensemble(ensemble_pred, true_labels, confidence_threshold=threshold)
    
    return results