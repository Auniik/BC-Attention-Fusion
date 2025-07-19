import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from analyze import analyze_predictions
from gradcam import GradCAM, visualize_gradcam
from multimag_dataset import MultiMagnificationDataset
from train import get_loss_weights, train_model

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


def main():


    from dataset import create_multi_mag_dataset_info, folds_df


    from utils.transforms import get_transforms

    # Create datasets
    train_transform = get_transforms('train', img_size=224)
    val_transform = get_transforms('val', img_size=224)


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    from backbones import get_all_backbones

    models = get_all_backbones()
    model = models['our_model'].to(device)



    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # Optional: if explicitly needed
    
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

    range_of_folds = [1] #range(1, 6)
    for fold in range_of_folds:  # 5-fold CV
        print(f"==== Fold {fold} ====")
        multi_mag_patients, single_mag_patients, fold_df, fold_statistics = create_multi_mag_dataset_info(folds_df, fold=fold)

        all_fold_statistics.append(fold_statistics)

        class_weights, tumor_weights = get_loss_weights(fold_df, device)


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

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2,
                                worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2,
                                worker_init_fn=seed_worker, generator=g)

        print("Starting training...")
        history, val_preds, val_labels = train_model(
            model,
            train_loader,
            val_loader,
            fold_df,
            fold=fold,
            num_epochs=10,
            device=device
        )

        # Store metrics for final combined plot
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

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        print(cm)
        all_cms.append(cm)

        analyze_predictions(val_labels, val_preds, val_loader)

        # Load best model
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
        model.eval()

        # Grad-CAM visualization
        gradcam = GradCAM(model)
        for i, batch in enumerate(val_loader):
            if i >= 3: break
            sample_images = {k: v[0:1].to(device) for k, v in batch['images'].items()}
            true_label = batch['class_label'][0].item()
            with torch.no_grad():
                logits, _ = model(sample_images)
                pred_label = logits.argmax(dim=1).item()
        
            cams = gradcam.get_cam(sample_images, target_class=pred_label)
            visualize_gradcam(
                cams,
                sample_images,
                true_label=true_label,
                pred_label=pred_label,
                save_path=f'gradcam_outputs/fold_{fold}_sample_{i}.png',
                show=True
            )

        print("Analyzing cross-magnification feature importance...")
        mag_importance = {'40x': [], '100x': [], '200x': [], '400x': []}
        correct_preds = 0
        total_preds = 0

        all_mags = ['40', '100', '200', '400']

        for batch in val_loader:
            batch_images = {k: v.to(device) for k, v in batch['images'].items()}
            batch_labels = batch['class_label'].to(device)

            with torch.no_grad():
                class_logits_full, _ = model(batch_images)
                conf_full = F.softmax(class_logits_full, dim=1).max(dim=1)[0]

                preds = class_logits_full.argmax(dim=1)
                correct_preds += (preds == batch_labels).sum().item()
                total_preds += len(batch_labels)

                for mag in all_mags:
                    masked_batch = {}
                    for m in all_mags:
                        mag_key = f'mag_{m}'
                        if m == mag:
                            masked_batch[mag_key] = torch.zeros_like(batch_images[mag_key])
                        else:
                            masked_batch[mag_key] = batch_images[mag_key]

                    logits_masked, _ = model(masked_batch)
                    conf_masked = F.softmax(logits_masked, dim=1).max(dim=1)[0]
                    contribution = (conf_full - conf_masked).cpu().numpy()
                    mag_importance[f'{mag}x'].extend(contribution)

        final_acc = correct_preds / total_preds
        print(f"\nFinal Validation Accuracy: {final_acc:.4f}")


        print("\nMagnification Importance Analysis:")
        for mag in all_mags:
            values = mag_importance[f'{mag}x']
            print(f"{mag}x contribution: {np.mean(values):.4f} ± {np.std(values):.4f}")

        # plt.figure(figsize=(10, 6))
        # plt.boxplot(
        #     [mag_importance[f'{mag}x'] for mag in all_mags],
        #     labels=[f'{mag}x' for mag in all_mags]
        # )
        # plt.title('Magnification Contribution to Model Confidence')
        # plt.ylabel('Confidence Contribution')
        # plt.grid(True, alpha=0.3)
        # plt.show()


if __name__ == "__main__":
    main()