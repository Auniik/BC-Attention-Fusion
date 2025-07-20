import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


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


def train_model(model, train_loader, val_loader, fold_df, fold, num_epochs, device):

    class_weights, tumor_weights = get_loss_weights(fold_df, device)
    
    # Loss functions
    # criterion_class = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    criterion_class = FocalLoss(alpha=2.5, gamma=2) 
    criterion_tumor = nn.CrossEntropyLoss(weight=tumor_weights, label_smoothing=0.05)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Mixed precision training for tensor cores
    use_amp = device.type == 'cuda' and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for tensor core optimization")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_tumor_acc': []
    }
    
    best_val_acc = 0.0
    balanced_acc = 0.0
    
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
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    class_logits, tumor_logits = model(images_dict)
                    
                    # Calculate losses
                    loss_class = criterion_class(class_logits, class_labels)
                    loss_tumor = criterion_tumor(tumor_logits, tumor_labels)
                    
                    # Combined loss
                    loss = 0.9 * loss_class + 0.1 * loss_tumor
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
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
            all_preds.append(preds)
            all_labels.append(class_labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Convert accumulated tensors to numpy once at the end
        all_preds_np = torch.cat(all_preds).cpu().numpy()
        all_labels_np = torch.cat(all_labels).cpu().numpy()
        
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels_np, all_preds_np)
        train_f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
        
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
                
                # Forward pass with mixed precision
                if use_amp:
                    with autocast():
                        class_logits, tumor_logits = model(images_dict)
                        
                        # Calculate losses
                        loss_class = criterion_class(class_logits, class_labels)
                        loss_tumor = criterion_tumor(tumor_logits, tumor_labels)
                        loss = 0.7 * loss_class + 0.3 * loss_tumor
                else:
                    # Standard forward pass
                    class_logits, tumor_logits = model(images_dict)
                    
                    # Calculate losses
                    loss_class = criterion_class(class_logits, class_labels)
                    loss_tumor = criterion_tumor(tumor_logits, tumor_labels)
                    loss = 0.7 * loss_class + 0.3 * loss_tumor
                
                # Statistics - accumulate on GPU
                running_loss += loss.item()
                preds = torch.argmax(class_logits, dim=1)
                tumor_preds = torch.argmax(tumor_logits, dim=1)
                
                all_preds.append(preds)
                all_labels.append(class_labels)
                all_tumor_preds.append(tumor_preds)
                all_tumor_labels.append(tumor_labels)
        
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
        tumor_acc = accuracy_score(all_tumor_labels_np, all_tumor_preds_np)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_tumor_acc'].append(tumor_acc)
        
        # Print results
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Tumor Acc: {tumor_acc:.4f}")
        print(f"Balanced Acc: {balanced_acc:.4f} (B: {benign_acc:.4f}, M: {malignant_acc:.4f})")
        
        # Save best model - handle DataParallel wrapper
        if balanced_acc > best_val_acc: 
            best_val_acc = balanced_acc
            if hasattr(model, 'module'):
                # Model is wrapped with DataParallel
                torch.save(model.module.state_dict(), f'output/best_model_fold_{fold}.pth')
            else:
                torch.save(model.state_dict(), f'output/best_model_fold_{fold}.pth')
            print(f"Saved best model of Fold-{fold} with validation accuracy: {best_val_acc:.4f}")
    
    return history, all_preds_np, all_labels_np