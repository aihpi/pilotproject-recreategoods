#!/bin/bash
#SBATCH --job-name=qwen-lora-multinode
#SBATCH --partition=aisc
#SBATCH --account=aisc-staff
#SBATCH --nodes=2                     # 2 nodes
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --gres=gpu:6                  # 6 GPUs per node = 12 total
#SBATCH --cpus-per-task=96            # 16 CPU cores per GPU (good for 12 workers/rank)
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1         # free speedup on H100
export NCCL_DEBUG=warn

# Multi-node communication setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# (Optional) Weights & Biases
export WANDB_ENTITY=felixboelter
export WANDB_PROJECT=qwen-image-edit-lora
export WANDB_MODE=online

# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

mkdir -p ./models/train logs

# Multi-node launch with 12 total GPUs (6 per node × 2 nodes)
uv run accelerate launch \
  --num_processes 12 \
  --num_machines 2 \
  --machine_rank $SLURM_PROCID \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  DiffSynth-Studio/examples/qwen_image/model_training/train_with_wandb.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_edit.csv \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --dataset_num_workers 12 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --find_unused_parameters