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

# Check current PyTorch installation
echo "🔍 Checking current PyTorch installation..."
python -c "import torch; print('Current PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None')" || echo "PyTorch not installed or has issues"
echo ""

# Uninstall existing PyTorch installations
echo "🗑️  Removing existing PyTorch installations..."
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.4)
echo "⚡ Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies with numpy compatibility fix
echo "📚 Installing project dependencies..."
pip install numpy==1.24.4  # Force compatible numpy version
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