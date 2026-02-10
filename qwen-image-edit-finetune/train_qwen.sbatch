#!/bin/bash
#SBATCH --job-name=qwen-lora
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export NCCL_DEBUG=warn

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
accelerate launch --num_processes 6 DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_edit.csv \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 5e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit_lora_rank16_lr5e5_wd002_correct" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 16 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.02 \
  --dataset_num_workers 12 \
  --save_steps 1000 \
  --find_unused_parameters