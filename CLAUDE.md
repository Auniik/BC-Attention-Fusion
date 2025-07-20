# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical imaging research project focused on multi-magnification histology analysis for breast cancer classification using the BreakHis dataset. The project implements a lightweight multi-magnification neural network that processes histology slides at multiple magnifications (40x, 100x, 200x, 400x) to classify breast cancer tumors as benign or malignant, with additional fine-grained tumor type classification.

## Key Commands

### Running the Main Experiment
```bash
python main.py
```
This runs the complete 5-fold cross-validation experiment including training, evaluation, and visualization generation.

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Single Components
- **Training only**: Use functions from `train.py`
- **Analysis only**: Use functions from `analyze/analyze.py`
- **Visualization**: Use functions from `gradcam.py` and `plotting.py`

## Architecture Overview

### Core Components

1. **Multi-Magnification Model** (`backbones/our/model.py`):
   - `LightweightMultiMagNet`: Main model class that processes multiple magnifications simultaneously
   - Uses EfficientNet-B0 as feature extractors for each magnification
   - Cross-magnification fusion through concatenation and linear layers
   - Dual classification heads: binary (benign/malignant) + tumor type (8 classes)

2. **Dataset Management** (`datasets/`):
   - `MultiMagnificationDataset`: Custom PyTorch dataset for handling multi-mag images
   - `preprocess.py`: Dataset creation and fold generation utilities
   - `examine.py`: Dataset analysis and statistics

3. **Training Pipeline** (`train.py`):
   - 5-fold cross-validation setup
   - Focal loss for class imbalance handling
   - Combined loss function: 90% classification + 10% tumor type
   - Balanced accuracy as primary metric

4. **Analysis & Visualization**:
   - `gradcam.py`: Grad-CAM visualization for model interpretability
   - `plotting.py`: Training metrics and confusion matrix visualization
   - `analyze/analyze.py`: Prediction analysis and statistics

### Data Flow

1. **Input**: Multi-magnification histology images from BreakHis dataset
2. **Processing**: Each magnification processed by separate EfficientNet extractors
3. **Fusion**: Features concatenated and passed through fusion layers
4. **Output**: Binary classification + tumor type predictions

### Model Backbones

- **Primary Model**: `LightweightMultiMagNet` (recommended)
- **Baseline**: `BaselineConcatNet` for comparison
- Configurable through `backbones/__init__.py`

## Configuration

### Key Files
- `config.py`: Dataset paths and base configuration
- `utils/helpers.py`: Path utilities and helper functions
- `utils/transforms.py`: Image augmentation pipelines

### Important Settings
- **Magnifications**: [40, 100, 200, 400] (configured in multiple files)
- **Image Size**: 224x224 pixels
- **Batch Size**: 8 (due to multi-mag memory requirements)
- **Device**: Auto-detection (CUDA > MPS > CPU)

## Development Notes

### Training Considerations
- Uses deterministic seeding (seed=42) for reproducibility
- Class balancing implemented in dataset loader
- Focal loss with α=2.5, γ=2 for handling class imbalance
- Cosine annealing learning rate scheduler

### Memory Management
- Multi-magnification processing is memory-intensive
- Batch size limited to 8 for most GPUs
- num_workers=0 recommended to avoid multiprocessing issues

### Output Structure
- `output/`: Model checkpoints for each fold
- `figs/`: Generated visualizations and plots
- Automated CSV reports for cross-fold analysis

### Dataset Structure
The BreakHis dataset should be placed in `data/breakhis/` with the standard directory structure including benign/malignant subdirectories and magnification-specific folders.

## ✅ CRITICAL ISSUES RESOLVED

### **Fixed: Cross-Validation Data Leakage** 
- **Issue**: All 82 patients appeared in all 5 folds, causing massive data leakage
- **Root Cause**: Original `Folds.csv` had image-level splits, not patient-level splits
- **Solution**: Created `datasets/fix_cross_validation.py` that implements proper patient-level stratified cross-validation
- **Files Changed**: 
  - `config.py`: Updated to use `Folds_fixed.csv`
  - Created `data/breakhis/Folds_fixed.csv` with proper patient splits
- **Validation**: `test_fixes.py` confirms zero patient overlap between train/test within folds

### **Fixed: Classification Report Evaluation**
- **Issue**: Training reports 97%+ accuracy but final classification reports showed 93-94%
- **Root Cause**: Final evaluation used last epoch model instead of best saved checkpoint
- **Solution**: Modified `main.py` to load best checkpoint before final evaluation and re-run predictions
- **Files Changed**: `main.py` lines 179-223 - moved checkpoint loading before analysis

### **Fixed: PyTorch Compatibility**
- **Issue**: `weights_only=True` parameter caused errors on older PyTorch versions
- **Solution**: Removed `weights_only=True` from all `torch.load()` calls
- **Files Changed**: `main.py`, `train.py`

## Expected Realistic Performance (Post-Fix)
- **Individual Fold Accuracy**: 85-95% (realistic for medical imaging with proper CV)
- **Ensemble Accuracy**: 90-96% (modest improvement, NOT 100%)
- **Per-class Requirements**: Both benign and malignant ≥85% accuracy
- **Classification Reports**: Should now match training metrics

## Deployment Instructions

### Local Development
```bash
# Test cross-validation fixes
python test_fixes.py

# Run full training with fixes
python main.py
```

### RunPod Deployment
```bash
# 1. First-time setup (automatically applies cross-validation fix)
python setup_runpod.py

# 2. Run training after setup
python main.py
```

### Manual Cross-Validation Fix (if needed)
```bash
# Apply cross-validation fix manually
python datasets/fix_cross_validation.py

# Validate the fix worked
python test_fixes.py
```

## RunPod Setup Requirements
1. **Upload BreakHis dataset** to `/workspace/breakhis/`
2. **Ensure file structure**:
   ```
   /workspace/breakhis/
   ├── Folds.csv                    # Original folds (will be fixed)
   ├── Folds_fixed.csv             # Generated by setup script
   └── BreaKHis_v1/                # Image dataset
       └── BreaKHis_v1/
           └── histology_slides/
   ```
3. **Run setup script**: `python setup_runpod.py`
4. **Start training**: `python main.py`

## Expected Results (Post-Fix)
- **Environment Detection**: Automatic (RunPod/Local/Kaggle)
- **Cross-Validation**: Proper patient-level splits
- **Individual Fold Accuracy**: 85-95% (realistic)
- **Ensemble Accuracy**: 90-96% (NOT 100%)
- **Training Time**: ~30 minutes on RTX 4090

**✅ STATUS**: All critical data leakage issues have been resolved. The ensemble should now show realistic performance (90-96%) instead of perfect 100% accuracy.