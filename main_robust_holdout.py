#!/usr/bin/env python3
"""
Robust Holdout Training Script for BreakHis Anti-Overfitting Evaluation

This script tests multiple holdout configurations to address the 100% test accuracy
issue seen in RUNPOD_OUTPUT.txt by using:

1. Larger test sets (15-20 patients vs 12)
2. Multiple configurations to test consistency 
3. Better stratification to reduce selection bias
4. Prediction confidence analysis to detect overconfidence

Expected: More realistic performance (85-94% instead of 100%)
"""

import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from analyze.analyze import analyze_predictions
from gradcam import plot_and_save_gradcam
from datasets.multi_mag import MultiMagnificationDataset
from plotting import plot_training_metrics, print_fold_metrics
from train import train_model
from datasets.preprocess import create_multi_mag_dataset_info, get_patients_for_mode
from config import get_training_config, TRAINING_CONFIG

from utils.transforms import get_transforms

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, classification_report
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backbones import get_all_backbones

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(42)

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

def get_device():
    """Get device info with tensor core optimization"""
    config = get_training_config()
    device = config['device']
    num_gpus = config['num_gpus']
    
    print(f"🖥️  Device: {device}")
    if device == 'cuda':
        print(f"🚀 CUDA GPU: {torch.cuda.get_device_name()}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"⚡ Tensor Cores: Available")
        # Use AMP and tensor cores for optimal training
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    elif device == 'mps':
        print(f"🍎 Apple Silicon GPU (MPS)")
    
    print(f"🔧 Number of workers: {config['num_workers']}")
    
    return device, num_gpus

def load_robust_holdout_folds(config_name):
    """Load one of the robust holdout split CSVs"""
    
    holdout_path = f"data/breakhis/Folds_robust_holdout_{config_name}.csv"
    
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(
            f"Robust holdout CSV not found at {holdout_path}. "
            f"Please run 'python datasets/create_robust_holdout_split.py' first."
        )
    
    folds_df = pd.read_csv(holdout_path)
    print(f"📂 Loaded robust holdout split ({config_name}): {len(folds_df)} image samples")
    print(f"   Train: {len(folds_df[folds_df['grp'] == 'train'])} images")
    print(f"   Validation: {len(folds_df[folds_df['grp'] == 'val'])} images") 
    print(f"   Test: {len(folds_df[folds_df['grp'] == 'test'])} images")
    
    return folds_df

def analyze_prediction_confidence(model, data_loader, device, split_name):
    """Analyze prediction confidence to detect overconfident models"""
    
    print(f"\n🔍 PREDICTION CONFIDENCE ANALYSIS - {split_name.upper()}")
    print("=" * 60)
    
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images_dict = {}
            for mag in TRAINING_CONFIG['magnifications']:
                images_dict[f'mag_{mag}'] = batch['images'][f'mag_{mag}'].to(device, non_blocking=True)
            
            class_labels = batch['class_label'].to(device, non_blocking=True)
            
            class_logits, _ = model(images_dict)
            probs = F.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(class_labels.cpu())
    
    # Concatenate all results
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Calculate confidence metrics
    max_probs = np.max(all_probs, axis=1)  # Highest probability for each prediction
    correct_mask = all_preds == all_labels
    
    print(f"📊 Confidence Statistics:")
    print(f"   Mean confidence: {max_probs.mean():.4f}")
    print(f"   Median confidence: {np.median(max_probs):.4f}")
    print(f"   Min confidence: {max_probs.min():.4f}")
    print(f"   Max confidence: {max_probs.max():.4f}")
    
    # Overconfidence analysis
    high_confidence = max_probs > 0.95
    perfect_confidence = max_probs > 0.99
    
    print(f"\n🚨 Overconfidence Detection:")
    print(f"   Predictions with >95% confidence: {high_confidence.sum()}/{len(max_probs)} ({high_confidence.mean()*100:.1f}%)")
    print(f"   Predictions with >99% confidence: {perfect_confidence.sum()}/{len(max_probs)} ({perfect_confidence.mean()*100:.1f}%)")
    
    if high_confidence.mean() > 0.8:
        print(f"   ⚠️  HIGH overconfidence detected - likely overfitting!")
    elif high_confidence.mean() > 0.6:
        print(f"   ⚠️  Moderate overconfidence - check generalization")
    else:
        print(f"   ✅ Confidence levels appear reasonable")
    
    # Confidence vs accuracy correlation
    correct_high_conf = correct_mask[high_confidence].mean() if high_confidence.sum() > 0 else 0
    correct_low_conf = correct_mask[~high_confidence].mean() if (~high_confidence).sum() > 0 else 0
    
    print(f"\n📈 Confidence-Accuracy Correlation:")
    print(f"   Accuracy on high-confidence predictions: {correct_high_conf:.4f}")
    print(f"   Accuracy on lower-confidence predictions: {correct_low_conf:.4f}")
    
    if abs(correct_high_conf - correct_low_conf) < 0.1:
        print(f"   ✅ Good calibration - confidence matches accuracy")
    else:
        print(f"   ⚠️  Poor calibration - confidence doesn't match accuracy")
    
    return {
        'mean_confidence': max_probs.mean(),
        'high_confidence_rate': high_confidence.mean(),
        'perfect_confidence_rate': perfect_confidence.mean(),
        'high_conf_accuracy': correct_high_conf,
        'low_conf_accuracy': correct_low_conf
    }

def run_single_configuration(config_name):
    """Run holdout training for a single robust configuration"""
    
    print(f"\n" + "="*100)
    print(f"🎯 TESTING ROBUST CONFIGURATION: {config_name.upper()}")
    print("="*100)
    
    # Initialize device and configuration
    device, num_gpus = get_device()
    config = get_training_config()
    
    # Load robust holdout split
    folds_df = load_robust_holdout_folds(config_name)
    
    # Extract patient information from filenames
    from datasets.examine import extract_tumor_type_and_patient_id
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
        zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    # Create dataset info for holdout split
    multi_mag_patients, _, fold_df, fold_statistics = create_multi_mag_dataset_info(folds_df, fold=1)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total patients with all magnifications: {len(multi_mag_patients)}")
    print(f"   Training samples: {fold_statistics['train_samples']}")
    print(f"   Test samples: {fold_statistics['test_samples']}")
    
    # Get transforms
    train_transform = get_transforms('train', img_size=TRAINING_CONFIG['img_size'])
    val_transform = get_transforms('val', img_size=TRAINING_CONFIG['img_size'])
    
    # Create patient splits 
    train_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='train')
    val_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='val')
    test_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='test')
    
    print(f"\n🎯 Patient Splits:")
    print(f"   Train: {len(train_patients)} patients")
    print(f"   Validation: {len(val_patients)} patients")
    print(f"   Test: {len(test_patients)} patients")
    
    # Create datasets
    print(f"\n🏗️  Creating datasets...")
    
    train_dataset = MultiMagnificationDataset(
        train_patients, 
        fold_df,
        mode='train',
        mags=TRAINING_CONFIG['magnifications'],
        samples_per_patient=TRAINING_CONFIG['samples_per_patient'],
        transform=train_transform,
        balance_classes=True,
        require_all_mags=True
    )
    
    val_dataset = MultiMagnificationDataset(
        val_patients,
        fold_df,
        mode='val',
        mags=TRAINING_CONFIG['magnifications'],
        samples_per_patient=TRAINING_CONFIG['samples_per_patient_val'],
        transform=val_transform,
        balance_classes=False,
        require_all_mags=True
    )
    
    test_dataset = MultiMagnificationDataset(
        test_patients,
        fold_df,
        mode='test',
        mags=TRAINING_CONFIG['magnifications'],
        samples_per_patient=TRAINING_CONFIG['samples_per_patient_val'],
        transform=val_transform,
        balance_classes=False,
        require_all_mags=True
    )
    
    print(f"\n📊 Dataset Sizes:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples") 
    print(f"   Test: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Train the model
    print(f"\n🚀 Starting training...")
    print(f"   Model: {TRAINING_CONFIG['model_name']}")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    
    # Get model
    from backbones.our.model import LightweightMultiMagNet
    model = LightweightMultiMagNet(
        magnifications=TRAINING_CONFIG['magnifications'],
        num_classes=2,
        num_tumor_types=8
    ).to(device)
    
    history, all_preds_np, all_labels_np = train_model(
        model,
        train_loader, 
        val_loader, 
        fold_df,
        fold=1,
        num_epochs=TRAINING_CONFIG['epochs'],
        device=device
    )
    
    # Load best checkpoint for final evaluation
    checkpoint_path = f"output/model_fold_1_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Loading best checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
    
    # Validation evaluation with confidence analysis
    print(f"\n" + "=" * 80)
    print(f"📊 VALIDATION EVALUATION")
    print(f"=" * 80)
    
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images_dict = {}
            for mag in TRAINING_CONFIG['magnifications']:
                images_dict[f'mag_{mag}'] = batch['images'][f'mag_{mag}'].to(device, non_blocking=True)
            
            class_labels = batch['class_label'].to(device, non_blocking=True)
            
            class_logits, _ = model(images_dict)
            preds = torch.argmax(class_logits, dim=1)
            
            val_preds.append(preds.cpu())
            val_labels.append(class_labels.cpu())
    
    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    
    # Validation metrics
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='binary')
    val_recall = recall_score(val_labels, val_preds, average='binary')
    val_f1 = f1_score(val_labels, val_preds, average='binary')
    
    print(f"📈 Validation Results:")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   Balanced Accuracy: {val_balanced_acc:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    
    # Validation confidence analysis
    val_confidence = analyze_prediction_confidence(model, val_loader, device, "Validation")
    
    # FINAL TEST EVALUATION
    print(f"\n" + "=" * 80)
    print(f"🧪 FINAL TEST EVALUATION")
    print(f"=" * 80)
    print(f"🚨 These patients were NEVER seen during training or validation")
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images_dict = {}
            for mag in TRAINING_CONFIG['magnifications']:
                images_dict[f'mag_{mag}'] = batch['images'][f'mag_{mag}'].to(device, non_blocking=True)
            
            class_labels = batch['class_label'].to(device, non_blocking=True)
            
            class_logits, _ = model(images_dict)
            preds = torch.argmax(class_logits, dim=1)
            
            test_preds.append(preds.cpu())
            test_labels.append(class_labels.cpu())
    
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    
    # Final test metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_balanced_acc = balanced_accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='binary')
    test_recall = recall_score(test_labels, test_preds, average='binary')
    test_f1 = f1_score(test_labels, test_preds, average='binary')
    
    print(f"🎯 FINAL TEST RESULTS:")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   F1-Score: {test_f1:.4f}")
    
    # Test confusion matrix
    test_cm = confusion_matrix(test_labels, test_preds)
    print(f"\n🔍 Final Test Confusion Matrix:")
    print(f"   [[TN={test_cm[0,0]}, FP={test_cm[0,1]}],")
    print(f"    [FN={test_cm[1,0]}, TP={test_cm[1,1]}]]")
    
    # Test confidence analysis
    test_confidence = analyze_prediction_confidence(model, test_loader, device, "Test")
    
    # Classification report
    print(f"\n📋 Final Test Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Benign', 'Malignant']))
    
    # Return results for comparison
    return {
        'config_name': config_name,
        'train_patients': len(train_patients),
        'val_patients': len(val_patients),
        'test_patients': len(test_patients),
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'val_confidence': val_confidence,
        'test_accuracy': test_accuracy,
        'test_balanced_accuracy': test_balanced_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_confidence': test_confidence
    }

def main():
    """Main execution - test all robust configurations"""
    
    print("🎯 ROBUST HOLDOUT ANTI-OVERFITTING EVALUATION")
    print("=" * 100)
    print("🚨 Testing multiple configurations to address 100% test accuracy issue")
    print("💡 Expected: More realistic performance across all configurations")
    print("=" * 100)
    
    # Test configurations
    configurations = [
        'balanced_large_test',  # 50+15+17 patients
        'moderate_test',        # 55+12+15 patients  
        'large_test'            # 45+17+20 patients
    ]
    
    results_summary = []
    
    for config_name in configurations:
        try:
            result = run_single_configuration(config_name)
            results_summary.append(result)
        except Exception as e:
            print(f"\n❌ Error with configuration {config_name}: {e}")
            continue
    
    # Compare results across configurations
    print(f"\n" + "="*100)
    print(f"📊 CROSS-CONFIGURATION COMPARISON")
    print("="*100)
    
    results_df = pd.DataFrame(results_summary)
    
    if len(results_df) > 0:
        print(f"\n📈 Test Accuracy Across Configurations:")
        for _, row in results_df.iterrows():
            print(f"   {row['config_name']}: {row['test_accuracy']:.4f} ({row['test_patients']} patients)")
        
        print(f"\n🔍 Consistency Analysis:")
        accuracy_std = results_df['test_accuracy'].std()
        accuracy_mean = results_df['test_accuracy'].mean()
        
        print(f"   Mean test accuracy: {accuracy_mean:.4f}")
        print(f"   Standard deviation: {accuracy_std:.4f}")
        
        if accuracy_std < 0.05:
            print(f"   ✅ Results are consistent across configurations")
        else:
            print(f"   ⚠️  High variability - may indicate dataset issues")
        
        if accuracy_mean > 0.95:
            print(f"   🚨 STILL HIGH ACCURACY - Overfitting likely remains!")
            print(f"   💡 Consider: Reduce model complexity, increase regularization")
        elif accuracy_mean >= 0.85:
            print(f"   ✅ Realistic medical AI performance achieved")
        else:
            print(f"   📉 Performance may be too low - check model/data issues")
        
        # Save comparison results
        results_df.to_csv('robust_holdout_comparison.csv', index=False)
        print(f"\n📁 Detailed results saved to: robust_holdout_comparison.csv")
    
    return results_summary

if __name__ == "__main__":
    results = main()
    print("\n🎉 Robust holdout evaluation completed!")