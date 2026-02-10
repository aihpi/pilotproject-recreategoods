#!/bin/bash
#SBATCH --job-name=qwen-curriculum-test
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/curriculum_test_%j.out
#SBATCH --error=logs/curriculum_test_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export NCCL_DEBUG=info

# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

# Activate the curriculum environment
source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio/curriculum_env/bin/activate

# Ensure logs/output dirs exist
mkdir -p ./models/test_curriculum logs

# Adaptive Curriculum Test - phases automatically determined by validation LPIPS
# The training will test the adaptive system with:
# - Automatic phase advancement based on validation performance
# - Early stopping when convergence is detected
# - Reduced resources for quick testing (2 GPUs, smaller batches)

echo "Starting adaptive curriculum test with 5 GPUs..."
echo "Testing: Automatic phase advancement based on validation LPIPS"
echo "Using config: configs/test_curriculum.yaml"

# Multi-GPU launch (5 processes = 5 GPUs) using config file
accelerate launch --num_processes 5 --mixed_precision bf16 DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py \
  --config configs/test_curriculum.yaml

echo "Adaptive curriculum test completed!"
echo "Check logs/curriculum_test_*.out for phase advancement details and LPIPS scores"
echo "Output saved to: ./models/test_curriculum/Qwen-Image-Edit_adaptive_curriculum_test"
echo ""
echo "Starting smaller 2-GPU test..."
echo "Using config: configs/test_curriculum_small.yaml"
accelerate launch --num_processes 2 --mixed_precision bf16 DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py \
  --config configs/test_curriculum_small.yaml

echo "Small curriculum test completed!"
echo "Output saved to: ./models/Qwen-Image-Edit_adaptive_fixed"