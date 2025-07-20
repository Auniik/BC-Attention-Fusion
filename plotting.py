import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
import torch
import torch.nn.functional as F


def plot_all_fold_confusion_matrices(confusion_matrices, class_names=['Benign', 'Malignant'], save_path=None):
    num_folds = len(confusion_matrices)
    cols = 3 if num_folds > 3 else 2
    rows = int(np.ceil(num_folds / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for i, cm in enumerate(confusion_matrices):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Fold {i+1}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    for j in range(num_folds, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 13,
#     "font.weight": "bold",
#     "axes.titlesize": 14,
#     "axes.labelsize": 13,
#     "legend.fontsize": 11,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "figure.dpi": 150
# })

def plot_training_metrics(history_dict, save_path=None):
    custom_rc = {
        "font.family": "serif",
        "font.size": 13,
        "font.weight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150
    }

    with plt.rc_context(rc=custom_rc):
        color_map = plt.get_cmap('tab20')  # or 'Set1' or 'Dark2'
        colors = color_map(np.linspace(0, 1, 10))  # 5 folds × 2 lines (train/val)
        
        num_folds = len(history_dict['train_loss'])
        epochs = range(1, len(history_dict['train_loss'][0]) + 1)
        
        fig, axs = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
        fig.suptitle('Per-Fold Training Performance', fontsize=14, fontweight='bold', y=0.95)
        
        fig.subplots_adjust(hspace=0.35)
        
        # Use one unique color per fold, with dashed (train) vs solid (val)
        for i in range(num_folds):
            axs[0].plot(epochs, history_dict['train_loss'][i], linestyle='--', color=colors[i], label=f'Train Fold {i+1}')
            axs[0].plot(epochs, history_dict['val_loss'][i], linestyle='-',  color=colors[i], label=f'Val Fold {i+1}')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('(a) Loss')
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].legend(loc='upper right', ncol=2)
        
        for i in range(num_folds):
            axs[1].plot(epochs, history_dict['train_acc'][i], linestyle='--', color=colors[i], label=f'Train Fold {i+1}')
            axs[1].plot(epochs, history_dict['val_acc'][i], linestyle='-',  color=colors[i], label=f'Val Fold {i+1}')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('(b) Accuracy')
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].legend(loc='lower right', ncol=2)
        
        for i in range(num_folds):
            axs[2].plot(epochs, history_dict['train_f1'][i], linestyle='--', color=colors[i], label=f'Train Fold {i+1}')
            axs[2].plot(epochs, history_dict['val_f1'][i], linestyle='-',  color=colors[i], label=f'Val Fold {i+1}')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('F1-Score')
        axs[2].set_title('(c) F1-Score')
        axs[2].grid(True, linestyle='--', alpha=0.6)
        axs[2].legend(loc='lower right', ncol=2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # plt.show()




def print_fold_metrics(per_fold_results, file_path='output/tab_per_fold_metrics.csv'):
    results_df = pd.DataFrame(per_fold_results)
    results_df.index = [f'Fold {i}' for i in results_df['fold']]
    results_df = results_df.drop(columns='fold')
    print("\n=== Per-Fold Results ===")
    print(tabulate(results_df, headers='keys', tablefmt='github', floatfmt=".4f"))
    results_df.to_csv(file_path)
    # results_df.to_latex("table1_foldwise_results.tex", float_format="%.4f")


def print_cross_fold_summary(all_fold_statistics, file_path='output/tab_cross_fold_summary.csv'):
    summary_df = pd.DataFrame(all_fold_statistics)

    summary_display_df = summary_df.drop(columns=['multi_mag_patients', 'class_distribution'])

    print("\n📊 Cross-Fold Summary Statistics:")
    print(summary_display_df.to_string(index=False))
    summary_display_df.to_csv(file_path)
    # summary_display_df.to_latex("table1_cross_fold_summary.tex", float_format="%.4f")



def plot_cross_magnification_fusion(model, val_loader, device, fold, save_path='figs/cross_magnification_fusion.png'):
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

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [mag_importance[f'{mag}x'] for mag in all_mags],
        labels=[f'{mag}x' for mag in all_mags]
    )
    plt.title('Magnification Contribution to Model Confidence')
    plt.ylabel('Confidence Contribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()