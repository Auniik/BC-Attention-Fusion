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
from datasets.preprocess import create_multi_mag_dataset_info

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

device = torch.device("cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

print(f"Using device: {device}")

def main():
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # Optional: if explicitly needed

    from datasets.examine import folds_df

    # Create datasets
    train_transform = get_transforms('train', img_size=224)
    val_transform = get_transforms('val', img_size=224)


    models = get_all_backbones()
    model = models['our_model'].to(device)

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

    range_of_folds = range(1, 6)
    for fold in range_of_folds:  # 5-fold CV
        print(f"==== Fold {fold} ====")
        multi_mag_patients, single_mag_patients, fold_df, fold_statistics = create_multi_mag_dataset_info(folds_df, fold=fold)
        all_fold_statistics.append(fold_statistics)

        train_dataset = MultiMagnificationDataset(
            multi_mag_patients, 
            fold_df,
            mode='train',
            mags=[40, 100, 200, 400],
            samples_per_patient=8,  # Increase for better epoch coverage
            transform=train_transform,
            balance_classes=True  # Enable balancing
        )

        val_dataset = MultiMagnificationDataset(
            multi_mag_patients,
            fold_df,
            mode='test',
            mags=[40, 100, 200, 400],
            samples_per_patient=2,
            transform=val_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4,
                                worker_init_fn=seed_worker, generator=g, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4,
                                worker_init_fn=seed_worker, generator=g, pin_memory=True)

        print("Starting training...")
        history, val_preds, val_labels = train_model(
            model,
            train_loader,
            val_loader,
            fold_df,
            fold=fold,
            num_epochs=5,
            device=device
        )

        analyze_predictions(val_labels, val_preds, val_loader)

        for key in all_fold_histories:
            all_fold_histories[key].append(history[key])

        fold_metrics = {
            "fold": fold,
            "accuracy": accuracy_score(val_labels, val_preds),
            "precision": precision_score(val_labels, val_preds, average='binary'),
            "recall": recall_score(val_labels, val_preds, average='binary'),
            "f1": f1_score(val_labels, val_preds, average='binary')
        }
        per_fold_results.append(fold_metrics)
        cm = confusion_matrix(val_labels, val_preds)
        all_cms.append(cm)

        # Load best model
        model.load_state_dict(torch.load(f'output/best_model_fold_{fold}.pth'))
        model.eval()

        # Grad-CAM visualization
        plot_and_save_gradcam(model, val_loader, device, fold)
        plot_cross_magnification_fusion(model, val_loader, device, fold)

    print_cross_fold_summary(all_fold_statistics)
    print_fold_metrics(per_fold_results)
    plot_all_fold_confusion_matrices(all_cms, save_path='figs/all_fold_confusion_matrices.png')
    plot_training_metrics(all_fold_histories, save_path='figs/training_metrics.png')


if __name__ == "__main__":
    main()