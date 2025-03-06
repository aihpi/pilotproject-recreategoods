#!/bin/bash
set -e

# Check if conda is installed, if not run the setup script
if [ ! -d "/workspace/miniconda" ]; then
    echo "Conda not found. Running setup script..."
    bash train/runpod_setup.sh
    echo "Please start a new shell session or run 'source ~/.bashrc' and then run this script again."
    exit 0
fi

# Check if we're in the conda environment
if [[ "$CONDA_DEFAULT_ENV" != "train-model" ]]; then
    echo "Conda environment 'train-model' is not activated."
    echo "Please run: conda activate train-model"
    echo "Then run this script again."
    exit 0
fi

# Create a comprehensive disk monitoring function
monitor_disk_space() {
    echo "===== Disk Space Information ====="
    
    # Check root filesystem
    ROOT_AVAIL=$(df -h / | awk 'NR==2 {print $4}')
    ROOT_TOTAL=$(df -h / | awk 'NR==2 {print $2}')
    ROOT_USAGE=$(df -h / | awk 'NR==2 {print $5}')
    echo "Root filesystem (/): $ROOT_AVAIL available of $ROOT_TOTAL ($ROOT_USAGE used)"
    
    # Check workspace filesystem
    if [ -d "/workspace" ]; then
        WORKSPACE_AVAIL=$(df -h /workspace | awk 'NR==2 {print $4}')
        WORKSPACE_TOTAL=$(df -h /workspace | awk 'NR==2 {print $2}')
        WORKSPACE_USAGE=$(df -h /workspace | awk 'NR==2 {print $5}')
        echo "Workspace filesystem (/workspace): $WORKSPACE_AVAIL available of $WORKSPACE_TOTAL ($WORKSPACE_USAGE used)"
        
        # Check if workspace is getting full
        WORKSPACE_USAGE_PCT=$(echo $WORKSPACE_USAGE | sed 's/%//')
        if [ "$WORKSPACE_USAGE_PCT" -gt 85 ]; then
            echo "WARNING: Workspace disk usage is high ($WORKSPACE_USAGE). Consider cleaning up old files."
        fi
    else
        echo "Workspace directory not found."
    fi
    
    echo "=================================="
}

# Run disk space check
monitor_disk_space

# Check available GPU memory and adjust batch size if needed
AVAILABLE_GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
echo "Available GPU memory: ${AVAILABLE_GPU_MEM}MB"

# Copy the RunPod config to a temporary file for potential modification
cp train/config/runpod.yaml train/config/runpod_temp.yaml

# If GPU memory is less than 16GB, reduce batch size and increase gradient accumulation
if [ "$AVAILABLE_GPU_MEM" -lt 16000 ]; then
    echo "Small GPU detected (< 16GB). Adjusting training parameters..."
    sed -i 's/batch_size: 4/batch_size: 2/' train/config/runpod_temp.yaml
    sed -i 's/val_batch_size: 4/val_batch_size: 2/' train/config/runpod_temp.yaml
    sed -i 's/accumulate_grad_batches: 16/accumulate_grad_batches: 32/' train/config/runpod_temp.yaml
    sed -i 's/strategy: "deepspeed_stage_2"/strategy: "deepspeed_stage_3"/' train/config/runpod_temp.yaml
fi

# If GPU memory is less than 8GB, further reduce batch size and use more aggressive memory optimization
if [ "$AVAILABLE_GPU_MEM" -lt 8000 ]; then
    echo "Very small GPU detected (< 8GB). Using more aggressive memory optimization..."
    sed -i 's/batch_size: 2/batch_size: 1/' train/config/runpod_temp.yaml
    sed -i 's/val_batch_size: 2/val_batch_size: 1/' train/config/runpod_temp.yaml
    sed -i 's/accumulate_grad_batches: 32/accumulate_grad_batches: 64/' train/config/runpod_temp.yaml
    sed -i 's/resize_res: \[256, 256\]/resize_res: \[224, 224\]/' train/config/runpod_temp.yaml
fi

# Check for existing checkpoints to resume training
LATEST_CHECKPOINT=$(find train/checkpoints -name "last.ckpt" -type f 2>/dev/null | sort -r | head -n 1)
RESUME_ARG=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT. Resuming training..."
    RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
fi

# Start training
echo "Starting training..."
python train/main.py --config_path train/config/runpod_temp.yaml $RESUME_ARG

# Keep the script running to maintain the pod
echo "Training complete. Pod will remain active for manual inspection."
tail -f /dev/null 