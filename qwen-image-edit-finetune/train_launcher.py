#!/usr/bin/env python3
import sys
import os
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Multi-GPU training script
def launch_training():
    """Launch the training script with proper arguments"""
    script_path = "DiffSynth-Studio/examples/qwen_image/model_training/train_with_segmentation.py"
    
    # Base arguments
    args = [
        script_path,
        "--dataset_base_path", "data/example_image_dataset",
        "--dataset_metadata_path", "data/example_image_dataset/metadata_edit.csv",
        "--data_file_keys", "image,edit_image",
        "--extra_inputs", "edit_image",
        "--max_pixels", "1048576",
        "--dataset_repeat", "1",
        "--model_id_with_origin_paths", "Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors",
        "--learning_rate", "5e-5",
        "--num_epochs", "2",
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--output_path", "./models/train/Qwen-Image-Edit_lora_segmentation_rank16_lr5e5_wd002",
        "--lora_base_model", "dit",
        "--lora_target_modules", "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",
        "--lora_rank", "16",
        "--use_gradient_checkpointing",
        "--gradient_accumulation_steps", "4",
        "--weight_decay", "0.02",
        "--dataset_num_workers", "12",
        "--save_steps", "1000",
        "--find_unused_parameters",
        "--mask_loss_weight", "1.0",
        "--similarity_threshold", "0.5",
        "--enable_dynamic_filtering"
    ]
    
    # Try different launch methods
    methods = [
        ["torchrun", "--nproc_per_node=6"] + args,
        ["python", "-m", "torch.distributed.run", "--nproc_per_node=6"] + args,
        ["python"] + args
    ]
    
    for method in methods:
        try:
            print(f"Trying launch method: {' '.join(method[:3])}")
            result = subprocess.run(method, check=True, cwd=project_root)
            return result.returncode
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Method failed: {e}")
            continue
    
    print("All launch methods failed, running single GPU training")
    return subprocess.run(["python"] + args, cwd=project_root).returncode

if __name__ == "__main__":
    sys.exit(launch_training())
