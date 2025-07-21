#!/usr/bin/env python3
"""
True Holdout Training Script for BreakHis Multi-Magnification Classification

Implements "Option 1: True Holdout Patient Split" from CLAUDE.md:
- Train on 60 patients (never see validation/test)
- Validate on 10 patients (for hyperparameter tuning and early stopping)
- Test on 12 patients (final evaluation only - completely unseen)

This provides the most realistic evaluation for clinical deployment
by eliminating all forms of data leakage.
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

def load_holdout_folds():
    """Load the holdout split CSV"""
    from config import HOLDOUT_PATH
    holdout_path = HOLDOUT_PATH
    
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(
            f"Holdout CSV not found at {holdout_path}. "
            f"Please run 'python datasets/create_holdout_split.py' first."
        )
    
    folds_df = pd.read_csv(holdout_path)
    print(f"📂 Loaded holdout split: {len(folds_df)} image samples")
    print(f"   Train: {len(folds_df[folds_df['grp'] == 'train'])} images")
    print(f"   Validation: {len(folds_df[folds_df['grp'] == 'val'])} images") 
    print(f"   Test: {len(folds_df[folds_df['grp'] == 'test'])} images")
    
    return folds_df

def main():
    """Main holdout training pipeline"""
    
    print("🎯 BREAKHIS TRUE HOLDOUT PATIENT SPLIT TRAINING")
    print("=" * 80)
    print("📊 Training without cross-validation for realistic medical AI evaluation")
    print("🏥 60 train + 10 validation + 12 test patients (zero overlap)")
    print("=" * 80)
    
    # Initialize device and configuration
    device, num_gpus = get_device()
    config = get_training_config()
    
    # Load holdout split
    folds_df = load_holdout_folds()
    
    # Extract patient information from filenames
    from datasets.examine import extract_tumor_type_and_patient_id
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
        zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    # Create dataset info for holdout split (use fold=1 as dummy since we only have one split)
    multi_mag_patients, _, fold_df, fold_statistics = create_multi_mag_dataset_info(folds_df, fold=1)
    
    print(f"\\n📈 Dataset Statistics:")
    print(f"   Total patients with all magnifications: {len(multi_mag_patients)}")
    print(f"   Training samples: {fold_statistics['train_samples']}")
    print(f"   Test samples: {fold_statistics['test_samples']}")  # This includes both val and test
    
    # Get transforms
    train_transform = get_transforms('train', img_size=TRAINING_CONFIG['img_size'])
    val_transform = get_transforms('val', img_size=TRAINING_CONFIG['img_size'])
    
    # Create patient splits 
    train_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='train')
    val_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='val')
    test_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='test')
    
    print(f"\\n🎯 Patient Splits:")
    print(f"   Train: {len(train_patients)} patients")
    print(f"   Validation: {len(val_patients)} patients")
    print(f"   Test: {len(test_patients)} patients")
    
    # Create datasets
    print(f"\\n🏗️  Creating datasets...")
    
    train_dataset = MultiMagnificationDataset(
        train_patients, 
        fold_df,
        mode='train',
        mags=TRAINING_CONFIG['magnifications'],
        samples_per_patient=TRAINING_CONFIG['samples_per_patient'],
        transform=train_transform,
        balance_classes=True,  # Balance for training
        require_all_mags=True  # Only patients with all magnifications
    )
    
    val_dataset = MultiMagnificationDataset(
        val_patients,
        fold_df,
        mode='val',
        mags=TRAINING_CONFIG['magnifications'],
        samples_per_patient=TRAINING_CONFIG['samples_per_patient_val'],
        transform=val_transform,
        balance_classes=False,  # No balancing for validation
        require_all_mags=True
    )
    
    test_dataset = MultiMagnificationDataset(
        test_patients,
        fold_df,
        mode='test',
        mags=TRAINING_CONFIG['magnifications'],
        samples_per_patient=TRAINING_CONFIG['samples_per_patient_val'],
        transform=val_transform,
        balance_classes=False,  # No balancing for test
        require_all_mags=True
    )
    
    print(f"\\n📊 Dataset Sizes:")
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
    print(f"\\n🚀 Starting training on holdout split...")
    print(f"   Model: {TRAINING_CONFIG['model_name']}")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    
    # Get model first
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
        fold=1,  # Dummy fold for compatibility
        num_epochs=TRAINING_CONFIG['epochs'],
        device=device
    )
    
    # Load best checkpoint for final evaluation
    checkpoint_path = f"output/model_fold_1_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"\\n📂 Loading best checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
    else:
        print(f"\\n⚠️  Checkpoint not found, using current model state")
    
    # Validation evaluation (on the 10 validation patients)
    print(f"\\n" + "=" * 80)
    print(f"📊 VALIDATION EVALUATION (10 patients)")
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
    
    print(f"\\n📈 Validation Results:")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   Balanced Accuracy: {val_balanced_acc:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    
    # Validation confusion matrix
    val_cm = confusion_matrix(val_labels, val_preds)
    print(f"\\n🔍 Validation Confusion Matrix:")
    print(f"   [[TN={val_cm[0,0]}, FP={val_cm[0,1]}],")
    print(f"    [FN={val_cm[1,0]}, TP={val_cm[1,1]}]]")
    
    # Classification report
    print(f"\\n📋 Validation Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Benign', 'Malignant']))
    
    # FINAL TEST EVALUATION (on the 12 completely unseen test patients)
    print(f"\\n" + "=" * 80)
    print(f"🧪 FINAL TEST EVALUATION (12 unseen patients)")
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
    
    print(f"\\n🎯 FINAL TEST RESULTS (Clinical Deployment Performance):")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   F1-Score: {test_f1:.4f}")
    
    # Test confusion matrix
    test_cm = confusion_matrix(test_labels, test_preds)
    print(f"\\n🔍 Final Test Confusion Matrix:")
    print(f"   [[TN={test_cm[0,0]}, FP={test_cm[0,1]}],")
    print(f"    [FN={test_cm[1,0]}, TP={test_cm[1,1]}]]")
    
    # Final classification report
    print(f"\\n📋 Final Test Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Benign', 'Malignant']))
    
    # Analyze predictions
    print(f"\\n🔬 Analyzing test predictions...")
    analyze_predictions(test_labels, test_preds, test_loader)
    
    # Generate visualizations
    print(f"\\n🎨 Generating visualizations...")
    
    # Training metrics plot
    plot_training_metrics(history, fold=1)
    
    # Grad-CAM visualization on test set
    plot_and_save_gradcam(model, test_loader, device, 1)
    
    # Save final results
    results_summary = {
        'training_setup': 'true_holdout_split',
        'train_patients': len(train_patients),
        'val_patients': len(val_patients),
        'test_patients': len(test_patients),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'validation_accuracy': val_accuracy,
        'validation_balanced_accuracy': val_balanced_acc,
        'validation_f1': val_f1,
        'test_accuracy': test_accuracy,
        'test_balanced_accuracy': test_balanced_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'model': TRAINING_CONFIG['model_name'],
        'epochs': TRAINING_CONFIG['epochs'],
        'batch_size': TRAINING_CONFIG['batch_size'],
        'learning_rate': TRAINING_CONFIG['learning_rate']
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv('holdout_results.csv', index=False)
    
    # Final summary
    print(f"\\n" + "=" * 80)
    print(f"🎯 HOLDOUT TRAINING COMPLETE")
    print(f"=" * 80)
    print(f"✅ Training: Complete on {len(train_patients)} patients")
    print(f"✅ Validation: {val_accuracy:.3f} accuracy on {len(val_patients)} patients")
    print(f"✅ Final Test: {test_accuracy:.3f} accuracy on {len(test_patients)} patients")
    print(f"")
    print(f"🏥 CLINICAL DEPLOYMENT READINESS:")
    print(f"   Final Test Accuracy: {test_accuracy:.1%}")
    print(f"   Balanced Accuracy: {test_balanced_acc:.1%}")
    print(f"   F1-Score: {test_f1:.1%}")
    print(f"")
    print(f"📊 Expected vs Actual Performance:")
    if test_accuracy > 0.95:
        print(f"   ⚠️  Accuracy {test_accuracy:.1%} seems high - verify no remaining data leakage")
    elif test_accuracy >= 0.85:
        print(f"   ✅ Accuracy {test_accuracy:.1%} is realistic for medical imaging")
    else:
        print(f"   📉 Accuracy {test_accuracy:.1%} may need model improvements")
    print(f"")
    print(f"📁 Results saved to: holdout_results.csv")
    print(f"🎨 Visualizations saved to: figs/")
    
    return results_summary

if __name__ == "__main__":
    results = main()
    print("\\n🎉 Holdout training pipeline completed successfully!")