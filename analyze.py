# 1. FIRST - Analyze what's really happening
import numpy as np


def analyze_predictions(val_labels, val_preds, val_loader):
    """Detailed analysis of model predictions"""
    
    # Convert to numpy if needed
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    
    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(val_labels, val_preds, 
                              target_names=['benign', 'malignant'], 
                              digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("\n=== CONFUSION MATRIX ===")
    print("Predicted:  benign  malignant")
    print(f"benign:      {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"malignant:   {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Calculate key metrics
    total_benign = cm[0].sum()
    total_malignant = cm[1].sum()
    benign_recall = cm[0,0] / total_benign if total_benign > 0 else 0
    malignant_recall = cm[1,1] / total_malignant if total_malignant > 0 else 0
    
    print(f"\n=== KEY INSIGHTS ===")
    print(f"Benign Recall: {benign_recall:.4f} ({cm[0,0]}/{total_benign})")
    print(f"Malignant Recall: {malignant_recall:.4f} ({cm[1,1]}/{total_malignant})")
    
    # Check if model is biased
    pred_distribution = np.bincount(val_preds)
    print(f"\nPrediction distribution: benign={pred_distribution[0]}, malignant={pred_distribution[1]}")
    
    if malignant_recall > 0.95 and benign_recall < 0.7:
        print("\n⚠️  Model is heavily biased towards malignant class!")
    
    return cm, benign_recall, malignant_recall