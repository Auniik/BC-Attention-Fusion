import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, balanced_accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_weights(fold_df, device):
    train_df = fold_df[fold_df['grp'] == 'train']
    
    # Binary classification weights
    class_counts = train_df['tumor_class'].value_counts()
    total = len(train_df)
    class_weights = torch.tensor([
        total / (2 * class_counts['benign']),
        total / (2 * class_counts['malignant'])
    ], dtype=torch.float32).to(device)
    
    # Tumor type weights
    tumor_types = sorted(fold_df['tumor_type'].unique())
    tumor_counts = train_df['tumor_type'].value_counts()
    
    tumor_weights = []
    for tumor in tumor_types:
        count = tumor_counts.get(tumor, 1)
        weight = np.sqrt(total / (len(tumor_types) * count))  # Square root for less aggressive weighting
        tumor_weights.append(weight)
    
    tumor_weights = torch.tensor(tumor_weights, dtype=torch.float32).to(device)
    tumor_weights = tumor_weights / tumor_weights.mean()  # Normalize
    
    return class_weights, tumor_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = self.weight
        if weight is not None:
            weight = weight.unsqueeze(0)
            
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        if weight is not None:
            nll_loss = nll_loss * weight.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
            
        smooth_loss = -log_prob.mean(dim=-1)
        if weight is not None:
            smooth_loss = smooth_loss * weight.mean()
            
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create a learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, train_loader, val_loader, fold_df, fold, num_epochs, device):

    class_weights, tumor_weights = get_loss_weights(fold_df, device)
    
    # Enhanced loss functions - Label smoothing for better generalization
    criterion_class = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    criterion_tumor = nn.CrossEntropyLoss(weight=tumor_weights, label_smoothing=0.05)
    
    # Optimizer with reduced learning rate and better weight decay
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3, betas=(0.9, 0.999))
    
    # Advanced learning rate scheduler with warmup
    warmup_epochs = 3
    scheduler = get_warmup_scheduler(optimizer, warmup_epochs, num_epochs)
    
    # Early stopping parameters
    best_balanced_acc = 0.0
    patience = 8
    patience_counter = 0
    
    
    # Training history with additional metrics
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_balanced_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_balanced_acc': [], 'val_tumor_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    balanced_acc = 0.0
    
    # Initialize variables for final return
    all_preds_np = np.array([])
    all_labels_np = np.array([])
    
    print(f"Starting {num_epochs} epoch training with warmup for {warmup_epochs} epochs")
    print(f"Early stopping patience: {patience} epochs")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            images_dict = {k: v.to(device, non_blocking=True) for k, v in batch['images'].items()}
            class_labels = batch['class_label'].to(device, non_blocking=True)
            tumor_labels = batch['tumor_type_label'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, tumor_logits = model(images_dict)
            
            # Calculate losses
            loss_class = criterion_class(class_logits, class_labels)
            loss_tumor = criterion_tumor(tumor_logits, tumor_labels)
            
            # Combined loss
            loss = 0.9 * loss_class + 0.1 * loss_tumor
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics - accumulate on GPU to reduce transfers
            running_loss += loss.item()
            preds = torch.argmax(class_logits, dim=1)
            all_preds.append(preds.detach())  # Detach to avoid keeping computation graph
            all_labels.append(class_labels.detach())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Convert accumulated tensors to numpy once at the end
        all_preds_np = torch.cat(all_preds).cpu().numpy()
        all_labels_np = torch.cat(all_labels).cpu().numpy()
        
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels_np, all_preds_np)
        train_f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
        train_balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
        
        # Validation
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_tumor_preds = []
        all_tumor_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                images_dict = {k: v.to(device, non_blocking=True) for k, v in batch['images'].items()}
                class_labels = batch['class_label'].to(device, non_blocking=True)
                tumor_labels = batch['tumor_type_label'].to(device, non_blocking=True)
                
                # Forward pass
                class_logits, tumor_logits = model(images_dict)
                
                # Calculate losses
                loss_class = criterion_class(class_logits, class_labels)
                loss_tumor = criterion_tumor(tumor_logits, tumor_labels)
                loss = 0.7 * loss_class + 0.3 * loss_tumor
                
                # Statistics - accumulate on GPU
                running_loss += loss.item()
                preds = torch.argmax(class_logits, dim=1)
                tumor_preds = torch.argmax(tumor_logits, dim=1)
                
                all_preds.append(preds.detach())
                all_labels.append(class_labels.detach())
                all_tumor_preds.append(tumor_preds.detach())
                all_tumor_labels.append(tumor_labels.detach())
        
        # Convert accumulated tensors to numpy once at the end
        all_preds_np = torch.cat(all_preds).cpu().numpy()
        all_labels_np = torch.cat(all_labels).cpu().numpy()
        all_tumor_preds_np = torch.cat(all_tumor_preds).cpu().numpy()
        all_tumor_labels_np = torch.cat(all_tumor_labels).cpu().numpy()
        
        # Calculate balanced accuracy
        benign_mask = all_labels_np == 0
        malignant_mask = all_labels_np == 1
        benign_acc = accuracy_score(all_labels_np[benign_mask], all_preds_np[benign_mask]) if benign_mask.any() else 0
        malignant_acc = accuracy_score(all_labels_np[malignant_mask], all_preds_np[malignant_mask]) if malignant_mask.any() else 0
        balanced_acc = (benign_acc + malignant_acc) / 2
        
        val_loss = running_loss / len(val_loader)
        val_acc = accuracy_score(all_labels_np, all_preds_np)
        val_f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
        val_balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
        tumor_acc = accuracy_score(all_tumor_labels_np, all_tumor_preds_np)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_balanced_acc'].append(train_balanced_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_balanced_acc'].append(val_balanced_acc)
        history['val_tumor_acc'].append(tumor_acc)
        history['learning_rate'].append(current_lr)
        
        # Print results
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Bal_Acc: {train_balanced_acc:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Bal_Acc: {val_balanced_acc:.4f}")
        print(f"Detailed Bal_Acc: {balanced_acc:.4f} (B: {benign_acc:.4f}, M: {malignant_acc:.4f})")
        print(f"Tumor Acc: {tumor_acc:.4f}, LR: {current_lr:.2e}")
        
        # Early stopping based on balanced accuracy
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            patience_counter = 0
            
            # Save best model - handle DataParallel wrapper
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), f'output/best_model_fold_{fold}.pth')
            else:
                torch.save(model.state_dict(), f'output/best_model_fold_{fold}.pth')
            print(f"🎯 New best balanced accuracy: {best_balanced_acc:.4f} - Model saved!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            print(f"Best balanced accuracy: {best_balanced_acc:.4f}")
            break
    
    return history, all_preds_np, all_labels_np