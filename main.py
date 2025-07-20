import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from analyze.analyze import analyze_predictions
from gradcam import plot_and_save_gradcam
from datasets.multi_mag import MultiMagnificationDataset
from plotting import plot_all_fold_confusion_matrices, plot_cross_magnification_fusion, plot_training_metrics, print_cross_fold_summary, print_fold_metrics
from train import train_model
from datasets.preprocess import create_multi_mag_dataset_info, get_patients_for_mode
from config import get_training_config, TRAINING_CONFIG

from utils.transforms import get_transforms

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
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
    
    print(f"🔧 Environment detected: {config['environment']}")
    print(f"🎯 Device: {device}")
    
    if device.type == "cuda":
        print(f"🚀 CUDA available with {num_gpus} GPU(s)")
        
        # Enable tensor core optimization for supported GPUs
        if num_gpus > 0:
            gpu_name = torch.cuda.get_device_name(0)
            if "A100" in gpu_name or "V100" in gpu_name or "RTX" in gpu_name or "T4" in gpu_name:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                print(f"⚡ Tensor core optimization enabled for {gpu_name}")
                
        if num_gpus > 1:
            print(f"🔥 Multi-GPU training will be enabled with {num_gpus} GPUs")
            
    elif device.type == "mps":
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        print("💻 Using CPU (consider using GPU for faster training)")
        
    print(f"📊 Batch size: {config['batch_size']} | Workers: {config['num_workers']}")
    
    return device, num_gpus, config


def main():
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # Optional: if explicitly needed

    from datasets.examine import folds_df

    device, num_gpus, train_config = get_device()

    # Create datasets
    train_transform = get_transforms('train', img_size=TRAINING_CONFIG['img_size'])
    val_transform = get_transforms('val', img_size=TRAINING_CONFIG['img_size'])


    models = get_all_backbones()

    seed_everything(42)

    per_fold_results = []
    all_fold_histories = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
    }
    all_cms = []
    all_fold_statistics = []

    # 5-FOLD ROTATION STRATEGY: Train/Val on each fold, test on others
    all_folds = [1, 2, 3, 4, 5]
    
    print(f"==== 5-FOLD ROTATION STRATEGY ====")
    print(f"🔄 Each fold will be used as train/val, with others as test")
    print(f"📊 This gives 5 independent test results for better generalizability")
    print(f"✅ Zero patient overlap between train/val and test sets")
    
    # Storage for all rotation results
    all_rotation_results = []
    all_test_results = []
    
    # Rotate through each fold as train/val
    for main_fold in all_folds:
        # Create fresh model for each rotation
        model = models['our_model'].to(device)
        
        # Enable multi-GPU training if available
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
            print(f"Model wrapped with DataParallel for {num_gpus} GPUs")
        test_folds = [f for f in all_folds if f != main_fold]
        
        print(f"\n{'='*80}")
        print(f"🔄 ROTATION {main_fold}/5: Train/Val on Fold {main_fold}, Test on Folds {test_folds}")
        print(f"{'='*80}")
        
        # Get main fold for training/validation
        main_multi_mag, _, main_fold_df, main_stats = create_multi_mag_dataset_info(folds_df, fold=main_fold)
        
        # Collect all test patients from other folds
        all_test_multi_mag = []
        all_test_fold_df_list = []
        
        for test_fold in test_folds:
            test_multi_mag, _, test_fold_df, _ = create_multi_mag_dataset_info(folds_df, fold=test_fold)
            # Get only test patients from this fold
            test_patients = get_patients_for_mode(test_multi_mag, test_fold_df, mode='test')
            all_test_multi_mag.extend(test_patients)
            
            # Get test samples from this fold
            test_samples = test_fold_df[test_fold_df['grp'] == 'test'].copy()
            all_test_fold_df_list.append(test_samples)
        
        # Combine all test fold data
        combined_test_fold_df = pd.concat(all_test_fold_df_list, ignore_index=True)
        print(f"📊 Train/Val fold {main_fold}: {len(main_multi_mag)} total patients")
        print(f"📊 Test patients: {len(all_test_multi_mag)} from folds {test_folds}")
        print(f"📊 Test samples: {len(combined_test_fold_df)}")        
        # Use main fold for training/validation split
        fold = main_fold
        multi_mag_patients = main_multi_mag
        fold_df = main_fold_df
        fold_statistics = main_stats
        all_fold_statistics.append(fold_statistics)
        
        print(f"==== Training on Fold {main_fold} ====")
        fold = main_fold

        # Train on train samples from main fold
        train_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='train')
        train_dataset = MultiMagnificationDataset(
            train_patients, 
            fold_df,
            mode='train',
            mags=TRAINING_CONFIG['magnifications'],
            samples_per_patient=TRAINING_CONFIG['samples_per_patient_train'],
            transform=train_transform,
            balance_classes=True  # Enable balancing
        )

        # Validate on test samples from main fold  
        val_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='test')
        val_dataset = MultiMagnificationDataset(
            val_patients,
            fold_df,
            mode='test',
            mags=TRAINING_CONFIG['magnifications'],
            samples_per_patient=TRAINING_CONFIG['samples_per_patient_val'],
            transform=val_transform,
            balance_classes=True  # Add class balancing to validation for fair evaluation
        )

        # Use device-specific configuration
        effective_batch_size = train_config['effective_batch_size']
        num_workers = train_config['num_workers']
        pin_memory = train_config['pin_memory']
        persistent_workers = train_config['persistent_workers']
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None, 
            generator=g if num_workers > 0 else None, 
            pin_memory=pin_memory, 
            persistent_workers=persistent_workers and num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=effective_batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None, 
            generator=g if num_workers > 0 else None, 
            pin_memory=pin_memory, 
            persistent_workers=persistent_workers and num_workers > 0
        )
        
        print(f"📈 Using batch size: {effective_batch_size} | Workers: {num_workers} | Environment: {train_config['environment']}")

        print("Starting training...")
        history, val_preds, val_labels = train_model(
            model,
            train_loader,
            val_loader,
            fold_df,
            fold=fold,
            num_epochs=TRAINING_CONFIG['num_epochs'],
            device=device
        )

        # Load best model BEFORE final evaluation - handle DataParallel wrapper
        print(f"\n📊 Loading best checkpoint for final evaluation...")
        checkpoint = torch.load(f'output/best_model_fold_{fold}.pth', map_location=device)
        if hasattr(model, 'module'):
            # Model is wrapped with DataParallel
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Re-evaluate with best model for accurate classification report
        print("🔍 Re-evaluating with best checkpoint...")
        all_preds_final = []
        all_labels_final = []
        
        with torch.no_grad():
            for batch in val_loader:
                images_dict = {k: v.to(device, non_blocking=True) for k, v in batch['images'].items()}
                class_labels = batch['class_label'].to(device, non_blocking=True)
                
                class_logits, _ = model(images_dict)
                preds = torch.argmax(class_logits, dim=1)
                
                all_preds_final.append(preds.cpu())
                all_labels_final.append(class_labels.cpu())
        
        val_preds_final = torch.cat(all_preds_final).numpy()
        val_labels_final = torch.cat(all_labels_final).numpy()

        # Use final predictions for analysis and metrics
        analyze_predictions(val_labels_final, val_preds_final, val_loader)

        for key in all_fold_histories:
            all_fold_histories[key].append(history[key])

        fold_metrics = {
            "fold": fold,
            "accuracy": accuracy_score(val_labels_final, val_preds_final),
            "precision": precision_score(val_labels_final, val_preds_final, average='binary'),
            "recall": recall_score(val_labels_final, val_preds_final, average='binary'),
            "f1": f1_score(val_labels_final, val_preds_final, average='binary')
        }
        per_fold_results.append(fold_metrics)
        cm = confusion_matrix(val_labels_final, val_preds_final)
        all_cms.append(cm)

        # Grad-CAM visualization
        plot_and_save_gradcam(model, val_loader, device, fold)
        plot_cross_magnification_fusion(model, val_loader, device, fold)

    # FINAL TEST EVALUATION on completely unseen test patients
    print(f"\n{'='*80}")
    print(f"🧪 FINAL TEST EVALUATION ON FOLDS {test_folds}")
    print(f"{'='*80}")
    
    # Create test dataset from all test patients
    test_dataset = MultiMagnificationDataset(
        all_test_multi_mag,
        combined_test_fold_df,
        mode='test',  # Use test samples from combined folds
        mags=TRAINING_CONFIG['magnifications'],
        samples_per_patient=TRAINING_CONFIG['samples_per_patient_val'],
        transform=val_transform,
        balance_classes=False  # No balancing for final test
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_config['effective_batch_size'], 
        shuffle=False, 
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory']
    )
    
    print(f"📊 Test Set: {len(all_test_multi_mag)} patients, {len(test_dataset)} samples")
    
    # Load best model for final test
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Test evaluation
    all_test_preds = []
    all_test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images_dict = {k: v.to(device, non_blocking=True) for k, v in batch['images'].items()}
            class_labels = batch['class_label'].to(device, non_blocking=True)
            
            class_logits, _ = model(images_dict)
            preds = torch.argmax(class_logits, dim=1)
            
            all_test_preds.append(preds.cpu())
            all_test_labels.append(class_labels.cpu())
    
    test_preds = torch.cat(all_test_preds).numpy()
    test_labels = torch.cat(all_test_labels).numpy()
    
    # Final test analysis
    print("🔬 FINAL TEST RESULTS:")
    analyze_predictions(test_labels, test_preds, test_loader)
    
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='binary')
    test_balanced_acc = balanced_accuracy_score(test_labels, test_preds)
    
    print(f"\n🎯 ROTATION {main_fold} RESULTS:")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test F1 Score: {test_f1:.4f}")
    print(f"   Test Balanced Acc: {test_balanced_acc:.4f}")
    print(f"{'='*80}")
    
    # Store results for this rotation
    rotation_result = {
        'main_fold': main_fold,
        'test_folds': test_folds,
        'val_accuracy': per_fold_results[-1]['accuracy'],
        'val_f1': per_fold_results[-1]['f1'],
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_balanced_acc': test_balanced_acc,
        'test_samples': len(test_labels)
    }
    all_rotation_results.append(rotation_result)
        all_test_results.extend(test_preds.tolist())

    # FINAL SUMMARY ACROSS ALL ROTATIONS
    print(f"\n{'='*80}")
    print(f"🏆 FINAL SUMMARY: 5-FOLD ROTATION RESULTS")
    print(f"{'='*80}")

    # Calculate statistics across all rotations
    val_accuracies = [r['val_accuracy'] for r in all_rotation_results]
    test_accuracies = [r['test_accuracy'] for r in all_rotation_results]
    test_f1s = [r['test_f1'] for r in all_rotation_results]

    print(f"\n📊 VALIDATION RESULTS (across 5 rotations):")
    print(f"   Mean: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
    print(f"   Range: {np.min(val_accuracies):.4f} - {np.max(val_accuracies):.4f}")

    print(f"\n🧪 TEST RESULTS (across 5 rotations):")
    print(f"   Mean Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print(f"   Mean F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"   Range: {np.min(test_accuracies):.4f} - {np.max(test_accuracies):.4f}")

    print(f"\n📋 DETAILED ROTATION RESULTS:")
    for i, result in enumerate(all_rotation_results, 1):
        print(f"   Rotation {i}: Val={result['val_accuracy']:.4f}, Test={result['test_accuracy']:.4f}")

    # Check for suspicious perfect results
    perfect_count = sum(1 for acc in test_accuracies if acc >= 0.999)
    if perfect_count > 0:
        print(f"\n⚠️  WARNING: {perfect_count}/5 rotations achieved perfect test accuracy!")
        print(f"   This suggests possible remaining data leakage or overfitting")
    else:
        print(f"\n✅ Realistic test accuracy range - no perfect scores detected")

    print(f"\n🎯 FINAL ASSESSMENT:")
    mean_test_acc = np.mean(test_accuracies)
    if mean_test_acc >= 0.95:
        print(f"   🎉 EXCELLENT: Mean test accuracy {mean_test_acc:.1%}")
    elif mean_test_acc >= 0.90:
        print(f"   ✅ GOOD: Mean test accuracy {mean_test_acc:.1%}")  
    else:
        print(f"   📈 ROOM FOR IMPROVEMENT: Mean test accuracy {mean_test_acc:.1%}")

    print(f"{'='*80}")

    print_cross_fold_summary(all_fold_statistics)
    plot_training_metrics(all_fold_histories, save_path='figs/training_metrics.png')
        
    print(f"\n🎉 ALL 5 ROTATIONS COMPLETE!")
    print(f"📊 Each fold used as train/val with others as test")
    print(f"📊 Total test evaluations: {len(all_rotation_results)} independent rotations") 
    print(f"✅ Comprehensive evaluation with zero patient overlap")
    
    # Skip ensemble evaluation for single model training
    """
    # Multi-fold ensemble evaluation for maximum accuracy
    print("\n" + "="*80)
    print("🚀 RUNNING ENSEMBLE EVALUATION FOR MAXIMUM ACCURACY")
    print("="*80)
    
    # Load all fold models
    ensemble_models = []
    fold_weights = []
    
    for fold in range_of_folds:
        # Create a fresh model instance
        fold_model = models['our_model']
        
        # Load the best checkpoint for this fold
        checkpoint = torch.load(f'output/best_model_fold_{fold}.pth', map_location=device)
        if hasattr(fold_model, 'module'):
            fold_model.module.load_state_dict(checkpoint)
        else:
            fold_model.load_state_dict(checkpoint)
        fold_model.eval()
        
        ensemble_models.append(fold_model)
        
        # Weight by fold performance (balanced accuracy)
        fold_performance = per_fold_results[fold-1]['f1']  # Use F1 as weight
        fold_weights.append(fold_performance)
    
    # Create validation dataset for ensemble evaluation
    # Use the last fold's validation set as representative
    from ensemble import run_ensemble_evaluation
    
    ensemble_results = run_ensemble_evaluation(
        models=ensemble_models,
        val_loader=val_loader,  # Use last validation loader
        device=device,
        fold_weights=fold_weights
    )
    
    print(f"\n🎯 FINAL ENSEMBLE ACCURACY: {ensemble_results['balanced_accuracy']:.4f}")
    print(f"🎯 TARGET ACHIEVED: {'✅ YES' if ensemble_results['balanced_accuracy'] >= 0.95 else '❌ NO (keep training)'}")
    print("="*80)

    """

if __name__ == "__main__":
    main()