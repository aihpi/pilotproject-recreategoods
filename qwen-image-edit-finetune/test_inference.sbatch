#!/bin/bash
#SBATCH --job-name=qwen-human-eval
#SBATCH --partition=aisc
#SBATCH --account=aisc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1

# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

# Ensure logs/output dirs exist
mkdir -p ./test_outputs logs

# Use conda environment that has required packages
source /sc/home/felix.boelter/miniconda3/etc/profile.d/conda.sh
conda activate base

# Install only missing packages
pip install pandas --quiet

# Run the inference test for multiple checkpoints
python test_models.py --model_config configs/config_qwen_curriculum.yaml