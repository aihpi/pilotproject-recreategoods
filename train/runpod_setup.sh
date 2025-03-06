#!/bin/bash
set -e

# Create workspace directory
mkdir -p /workspace
cd /workspace

# Check if conda is installed, if not install it
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /workspace/miniconda
    
    # Add conda to path
    export PATH="/workspace/miniconda/bin:$PATH"
    
    # Initialize conda for bash
    /workspace/miniconda/bin/conda init bash
    
    # Create a symlink to make conda available system-wide
    ln -sf /workspace/miniconda/bin/conda /usr/local/bin/conda
    
    echo "Conda installed successfully."
    
    # Source bashrc to get conda working in current session
    source ~/.bashrc
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

# Create conda environment
echo "Setting up conda environment..."
conda env create -f train/environment.yaml -n train-model || conda env update -f train/environment.yaml -n train-model
source $(conda info --base)/etc/profile.d/conda.sh
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
while true; do
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        echo "WARNING: Disk usage is at $DISK_USAGE%. Cleaning up..."
        # Clean up old checkpoints, keeping only the latest 3
        find /workspace/recreategoods/train/checkpoints -type d -name "epoch=*" | sort | head -n -3 | xargs rm -rf
    fi
    sleep 300  # Check every 5 minutes
done
EOL

chmod +x /workspace/monitor_disk.sh
nohup /workspace/monitor_disk.sh > /workspace/disk_monitor.log 2>&1 &

# Add conda initialization to .bashrc if not already there
if ! grep -q "conda initialize" ~/.bashrc; then
    echo "Adding conda initialization to .bashrc..."
    conda init bash
    source ~/.bashrc
fi

echo "Setup complete! You can now run training with:"
echo "cd /workspace/recreategoods && conda activate train-model && python train/main.py --config_path train/config/runpod.yaml" 