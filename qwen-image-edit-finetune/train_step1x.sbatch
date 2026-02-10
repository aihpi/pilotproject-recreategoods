#!/bin/bash
#SBATCH --job-name=step1x-lora
#SBATCH --partition=aisc
#SBATCH --account=aisc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/train_step1x_%j.out
#SBATCH --error=logs/train_step1x_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export NCCL_DEBUG=warn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

# Optional: activate your env if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your-env>

# Ensure logs/output dirs exist
mkdir -p ./models/train logs

# Hard-disable DeepSpeed import to avoid nvcc checks during checkpointing
python - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("deepspeed"):
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "deepspeed"], check=False)
    except Exception:
        pass
PY
export PYTHONNOUSERSITE=1

# Multi-GPU launch (8 processes = 8 GPUs)
accelerate launch --num_processes 5 DiffSynth-Studio/examples/flux/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_step1x.csv \
  --data_file_keys "image,step1x_reference_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen2.5-VL-7B-Instruct:,stepfun-ai/Step1X-Edit:step1x-edit-i1258.safetensors,stepfun-ai/Step1X-Edit:vae.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Step1X-Edit_lora-rank32-1e4-wd0" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --extra_inputs "step1x_reference_image" \
  --align_to_opensource_format \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.0 \
  --dataset_num_workers 12 \
  --save_steps 1000 \
  --find_unused_parameters