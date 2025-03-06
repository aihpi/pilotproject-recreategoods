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

monitor_disk_space() {
    echo "===== Disk Space Check ====="
    
    # Check root filesystem
    ROOT_AVAIL=$(df -h / | awk 'NR==2 {print $4}')
    ROOT_TOTAL=$(df -h / | awk 'NR==2 {print $2}')
    ROOT_USAGE=$(df -h / | awk 'NR==2 {print $5}')
    ROOT_USAGE_PCT=$(echo $ROOT_USAGE | sed 's/%//')
    
    # Check workspace filesystem
    WORKSPACE_AVAIL=$(df -h /workspace | awk 'NR==2 {print $4}')
    WORKSPACE_TOTAL=$(df -h /workspace | awk 'NR==2 {print $2}')
    WORKSPACE_USAGE=$(df -h /workspace | awk 'NR==2 {print $5}')
    WORKSPACE_USAGE_PCT=$(echo $WORKSPACE_USAGE | sed 's/%//')
    
    echo "$(date) - Root: $ROOT_AVAIL available of $ROOT_TOTAL ($ROOT_USAGE used)"
    echo "$(date) - Workspace: $WORKSPACE_AVAIL available of $WORKSPACE_TOTAL ($WORKSPACE_USAGE used)"
    
    # Clean up if workspace is getting full
    if [ "$WORKSPACE_USAGE_PCT" -gt 85 ]; then
        echo "$(date) - WARNING: Workspace disk usage is high ($WORKSPACE_USAGE). Cleaning up..."
        if [ -d "/workspace/recreategoods/train/checkpoints" ]; then
            find /workspace/recreategoods/train/checkpoints -type d -name "epoch=*" | sort | head -n -3 | xargs rm -rf 2>/dev/null || true
        fi
    fi
}

while true; do
    monitor_disk_space >> /workspace/disk_monitor.log
    sleep 300  # Check every 5 minutes
done
EOL

chmod +x /workspace/monitor_disk.sh
nohup /workspace/monitor_disk.sh > /dev/null 2>&1 &

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