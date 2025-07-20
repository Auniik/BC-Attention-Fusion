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

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
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
    model = models['our_model'].to(device)
    
    # Enable multi-GPU training if available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel for {num_gpus} GPUs")

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

    range_of_folds = range(1, 4)  # 3-fold CV for larger validation sets
    for fold in range_of_folds:  # 3-fold CV
        print(f"==== Fold {fold} ====")
        multi_mag_patients, single_mag_patients, fold_df, fold_statistics = create_multi_mag_dataset_info(folds_df, fold=fold)
        all_fold_statistics.append(fold_statistics)

        # Filter patients for proper cross-validation
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

        # Filter patients for proper cross-validation
        test_patients = get_patients_for_mode(multi_mag_patients, fold_df, mode='test')
        
        val_dataset = MultiMagnificationDataset(
            test_patients,
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

    print_cross_fold_summary(all_fold_statistics)
    print_fold_metrics(per_fold_results)
    plot_all_fold_confusion_matrices(all_cms, save_path='figs/all_fold_confusion_matrices.png')
    plot_training_metrics(all_fold_histories, save_path='figs/training_metrics.png')
    
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


if __name__ == "__main__":
    main()