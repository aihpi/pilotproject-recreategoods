#!/bin/bash
#SBATCH --job-name=qwen-validate
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err

# Add DiffSynth to Python path
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create output directory
mkdir -p validation_results logs

# Run validation with uv
uv run python validate_training.py