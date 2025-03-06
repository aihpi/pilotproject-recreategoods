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

# Start disk space monitoring in the background
echo "Starting disk space monitoring..."
python train/disk_management.py --threshold 85 --interval 300 &
DISK_MONITOR_PID=$!

# Function to clean up on exit
cleanup() {
    echo "Cleaning up..."
    if [ -n "$DISK_MONITOR_PID" ]; then
        kill $DISK_MONITOR_PID
    fi
    exit 0
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

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
LATEST_CHECKPOINT=$(find train/checkpoints -name "last.ckpt" -type f | sort -r | head -n 1)
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