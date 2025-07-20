#!/bin/bash

echo "🚀 Setting up RunPod environment for BC-Attention-Fusion..."
echo "=================================================="

# Check if we're running as root (common in RunPod)
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Running as root - this is normal for RunPod"
fi

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq

# Check NVIDIA driver and CUDA
echo "🔍 Checking GPU and CUDA..."
nvidia-smi
echo ""

# Check current PyTorch installation (should be 2.1.0 with CUDA 11.8)
echo "🔍 Checking base container PyTorch..."
python -c "import torch; print('Base PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None')" || echo "PyTorch has issues"
echo ""

# Skip PyTorch reinstallation - use base container version (2.1.0 + CUDA 11.8)
echo "⚡ Using base container PyTorch 2.1.0 with CUDA 11.8..."
echo "   (Skipping PyTorch reinstallation to save time)"

# Fix NumPy compatibility with PyTorch 2.1.0
echo "🔧 Fixing NumPy compatibility..."
echo "Current NumPy version:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy import failed"
echo "Downgrading to NumPy 1.x for PyTorch 2.1.0 compatibility..."
pip install "numpy==1.24.3" --force-reinstall --no-deps

# Verify NumPy downgrade worked
echo "Verifying NumPy downgrade:"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" || echo "❌ NumPy still has issues"

# Install only missing dependencies
echo "📚 Installing additional project dependencies..."
pip install -r requirements.txt.runpod

# Verify CUDA installation
echo ""
echo "✅ Verifying CUDA installation..."
python -c "
import torch
print('=' * 50)
print('🔥 PYTORCH & CUDA VERIFICATION')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # Test tensor operations
    print('Testing GPU tensor operations...')
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('✅ GPU tensor operations working!')
else:
    print('❌ CUDA not available - check installation')
print('=' * 50)
"

# Create output directory
echo ""
echo "📁 Creating output directories..."
mkdir -p output
mkdir -p figs

# Set proper permissions
echo "🔐 Setting permissions..."
chmod -R 755 .

echo ""
echo "🎉 Setup complete! Ready to run training."
echo "📋 To start training: python main.py"
echo "🔍 To monitor GPU: watch -n 1 nvidia-smi"