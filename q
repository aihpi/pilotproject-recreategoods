#!/bin/bash
#SBATCH --job-name=qwen-segment-wandb
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h100:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/segment_wandb_train_%j.out
#SBATCH --error=logs/segment_wandb_train_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export NCCL_DEBUG=warn

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    pip install --break-system-packages uv
fi

# Set up virtual environment with UV
VENV_DIR="/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with UV..."
    uv venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
echo "Virtual environment activated"

# Install all required dependencies from requirements.txt
uv pip install -r DiffSynth-Studio/requirements.txt

# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

# Ensure logs/output dirs exist
mkdir -p ./models/train logs

# Launch WandB-enhanced training
echo "Starting WandB-enhanced training..."
python train_launcher_wandb.py