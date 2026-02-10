#!/bin/bash
#SBATCH --job-name=qwen-simple-test
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/simple_test_%j.out
#SBATCH --error=logs/simple_test_%j.err

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export HF_HUB_OFFLINE=1

# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

# Activate the curriculum environment
source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio/curriculum_env/bin/activate

# Ensure logs/output dirs exist
mkdir -p ./models/test_simple logs

echo "Starting simple test - NO validation, single GPU, 1 step"

# Single GPU, no validation
python DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/validation.csv \
  --num_epochs 1 \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "./models/Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,./models/Qwen/Qwen-Image:text_encoder/model*.safetensors,./models/Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 5e-15 \
  --weight_decay 1e-2 \
  --save_steps 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/test_simple/Qwen-Image-Edit_simple_test" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 1 \
  --use_gradient_checkpointing \
  --dataset_num_workers 1 \
  --gradient_accumulation_steps 1 \
  --task "sft"

echo "Simple test completed!"
echo "Check for checkpoints in: ./models/test_simple/Qwen-Image-Edit_simple_test"