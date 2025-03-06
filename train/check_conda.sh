#!/bin/bash
set -e

echo "Checking conda installation..."

# Check if conda command is available
if command -v conda &> /dev/null; then
    echo "✅ Conda is installed and in PATH"
    conda --version
else
    echo "❌ Conda command not found in PATH"
    
    # Check if conda is installed in common locations
    for dir in "/workspace/miniconda" "/opt/conda" "/usr/local/miniconda" "$HOME/miniconda" "$HOME/anaconda3"; do
        if [ -d "$dir" ]; then
            echo "Found conda installation at: $dir"
            echo "Adding to PATH..."
            export PATH="$dir/bin:$PATH"
            
            if command -v conda &> /dev/null; then
                echo "✅ Successfully added conda to PATH"
                conda --version
                break
            else
                echo "❌ Failed to add conda to PATH from $dir"
            fi
        fi
    done
    
    if ! command -v conda &> /dev/null; then
        echo "No conda installation found. Please run the setup script:"
        echo "bash /workspace/recreategoods/train/runpod_setup.sh"
        exit 1
    fi
fi

# Check if train-model environment exists
if conda env list | grep -q "train-model"; then
    echo "✅ train-model environment exists"
else
    echo "❌ train-model environment not found"
    echo "Creating environment..."
    conda env create -f /workspace/recreategoods/train/environment.yaml -n train-model
fi

# Try to activate the environment
echo "Activating train-model environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda activate train-model; then
    echo "✅ Successfully activated train-model environment"
    
    # Check for key packages
    echo "Checking for key packages..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if torch.cuda.is_available():
        python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
        python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0)}')"
    
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    python -c "import diffusers; print(f'Diffusers version: {diffusers.__version__}')"
    python -c "import pytorch_lightning; print(f'PyTorch Lightning version: {pytorch_lightning.__version__}')"
else
    echo "❌ Failed to activate train-model environment"
    exit 1
fi

echo "Conda check complete!" 