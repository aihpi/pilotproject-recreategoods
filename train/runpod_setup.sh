#!/bin/bash
set -e

# Create workspace directory
mkdir -p /workspace
cd /workspace

# Check if conda is installed or if the miniconda directory exists
if ! command -v conda &> /dev/null; then
    if [ -d "/workspace/miniconda" ]; then
        echo "Miniconda directory exists but conda command not found. Adding to PATH..."
        export PATH="/workspace/miniconda/bin:$PATH"
        
        # Initialize conda for bash if not already done
        if ! grep -q "conda initialize" ~/.bashrc; then
            echo "Initializing conda in .bashrc..."
            /workspace/miniconda/bin/conda init bash
        fi
    else
        echo "Conda not found. Installing Miniconda..."
        
        # Download and install Miniconda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p /workspace/miniconda
        
        # Add conda initialization to .bashrc
        /workspace/miniconda/bin/conda init bash
        
        echo "Conda installed successfully."
    fi
    
    echo ""
    echo "============================================================"
    echo "IMPORTANT: To use conda, you need to either:"
    echo "  1. Start a new terminal session, or"
    echo "  2. Run: source ~/.bashrc"
    echo "============================================================"
    echo ""
    
    # Export PATH for the current script
    export PATH="/workspace/miniconda/bin:$PATH"
else
    echo "Conda is already installed."
fi

# Clone the repository
if [ ! -d "/workspace/recreategoods" ]; then
    echo "Cloning repository..."
    git clone https://github.com/aihpi/recreategoods.git -b macos-runpod-train
    cd recreategoods
else
    echo "Repository already exists, updating..."
    cd recreategoods
    git pull
fi

# Ensure conda is in PATH for this script
export PATH="/workspace/miniconda/bin:$PATH"
source /workspace/miniconda/etc/profile.d/conda.sh

# Create conda environment
echo "Setting up conda environment..."
conda env create -f train/environment.yaml -n train-model || conda env update -f train/environment.yaml -n train-model
conda activate train-model

# Install additional dependencies for smaller GPU
pip install --upgrade deepspeed
pip install --upgrade accelerate
pip install --upgrade psutil

# Create dataset directory
mkdir -p /workspace/dataset

# Set up disk space management
echo "Setting up disk space monitoring..."
cat > /workspace/monitor_disk.sh << 'EOL'
#!/bin/bash
export PATH="/workspace/miniconda/bin:$PATH"
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate train-model

while true; do
    # Check disk usage of the workspace filesystem
    DISK_USAGE=$(df -h /workspace | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 85 ]; then
        echo "WARNING: Workspace disk usage is at $DISK_USAGE%. Cleaning up..."
        # Clean up old checkpoints, keeping only the latest 3
        if [ -d "/workspace/recreategoods/train/checkpoints" ]; then
            find /workspace/recreategoods/train/checkpoints -type d -name "epoch=*" | sort | head -n -3 | xargs rm -rf 2>/dev/null || true
        fi
    fi
    sleep 300  # Check every 5 minutes
done
EOL

chmod +x /workspace/monitor_disk.sh
nohup /workspace/monitor_disk.sh > /workspace/disk_monitor.log 2>&1 &

echo ""
echo "============================================================"
echo "Setup complete! To use conda and start training:"
echo ""
echo "1. First, activate conda in your shell:"
echo "   source ~/.bashrc"
echo "   conda activate train-model"
echo ""
echo "2. Then, start training:"
echo "   cd /workspace/recreategoods"
echo "   python train/main.py --config_path train/config/runpod.yaml"
echo "============================================================" 