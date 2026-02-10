#!/bin/bash
#SBATCH --job-name=qwen-curriculum
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=26:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/curriculum_%j.out
#SBATCH --error=logs/curriculum_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export NCCL_DEBUG=warn

# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

# Activate the curriculum environment
source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio/curriculum_env/bin/activate

# Ensure logs/output dirs exist
mkdir -p ./models/curriculum logs

# Adaptive Curriculum Training - phases automatically determined by validation LPIPS
# The training will automatically:
# - Start with easy samples (tier 0 only)
# - Advance phases when validation LPIPS stops improving
# - Early stop on final phase when convergence is reached

echo "Starting production curriculum training with 8 GPUs..."
echo "Using config: configs/day_long_curriculum.yaml"

# Multi-GPU launch (8 processes = 8 GPUs) using config file
accelerate launch --num_processes 8 --mixed_precision bf16 DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py --config configs/day_long_curriculum.yaml

echo "Adaptive curriculum training completed!"
echo "Output saved to: ./models/curriculum/Qwen-Image-Edit_adaptive_curriculum_lora_rank32_lr1e4"
