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