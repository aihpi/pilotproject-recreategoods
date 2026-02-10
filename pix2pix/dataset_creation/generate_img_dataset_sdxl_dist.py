import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import argparse
import json
import sys
sys.path.append("./")
from typing import Dict, Optional, Union
import logging, traceback
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import random

# Removed flux-specific imports and replaced with diffusers pipeline
from utils.prompt_to_prompt_sdxl import Prompt2PromptPipeline
from metrics.clip_similarity import ClipSimilarity
from diffusers import DiffusionPipeline

def setup_logging(rank, log_dir="../logs"):
    # Create the log directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join(log_dir, timestamp)
    os.makedirs(log_folder, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(log_folder, f'gpu_{rank}_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [Rank {rank}] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_distributed(local_rank: int):
    """Initialize distributed training environment."""
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}")

def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def save_results(results: Dict, prompt_dir: Path, opt: argparse.Namespace):
    """Save generated images and metadata."""
    metadata = [
        (result["clip_sim_dir"], seed)
        for seed, result in results.items()
        if result["clip_sim_image"] >= opt.clip_img_threshold
        and result["clip_sim_dir"] >= opt.clip_dir_threshold
        and result["clip_sim_0"] >= opt.clip_threshold
        and result["clip_sim_1"] >= opt.clip_threshold
    ]
    
    metadata.sort(reverse=True)
    for _, seed in metadata[: opt.max_out_samples]:
        result = results[seed]
        image_0 = result.pop("image_0")
        image_1 = result.pop("image_1")
        image_0.save(prompt_dir.joinpath(f"{seed}_0.jpg"), quality=100)
        image_1.save(prompt_dir.joinpath(f"{seed}_1.jpg"), quality=100)
        with open(prompt_dir.joinpath(f"metadata.jsonl"), "a") as fp:
            fp.write(f"{json.dumps(dict(seed=seed, **result))}\n")

def get_global_prompt_idx(local_prompt_idx: int, local_rank: int, world_size: int, all_prompts_length: int) -> int:
    """Calculate the global prompt index from local index and rank."""
    if world_size > 1:
        # Calculate how many prompts each GPU handles
        prompts_per_gpu = all_prompts_length // world_size
        # Global index is (prompts_per_gpu * rank) + local_index
        return (prompts_per_gpu * local_rank) + local_prompt_idx
    return local_prompt_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-out-samples", type=int, default=4)
    parser.add_argument("--clip-threshold", type=float, default=0.15)
    parser.add_argument("--clip-dir-threshold", type=float, default=0.15)
    parser.add_argument("--clip-img-threshold", type=float, default=0.9)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--min-cfg", type=float, default=7.5, help="Min classifier free guidance scale.")
    parser.add_argument("--max-cfg", type=float, default=15, help="Max classifier free guidance scale.")
    opt = parser.parse_args()

    # Set up distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        device = setup_distributed(local_rank)
        is_main_process = local_rank == 0
    else:
        device = torch.device("cuda")
        is_main_process = True

    logger = setup_logging(local_rank)
    logger.info(f"Starting process on GPU {local_rank} of {world_size}")

    # Load and prepare prompts
    with open(opt.prompts_file, 'r') as fp:
        all_prompts = json.load(fp)
    
    total_samples = opt.n_samples * len(all_prompts)
    
    if world_size > 1:
        # Calculate samples per GPU
        samples_per_gpu = total_samples // world_size
        if is_main_process:
            logger.info(f"Total samples: {total_samples}, Samples per GPU: {samples_per_gpu}")
        
        # Distribute prompts
        prompts_with_samples = []
        current_samples = 0
        current_prompt_samples = []
        
        for prompt in all_prompts:
            samples_for_prompt = opt.n_samples
            if current_samples + samples_for_prompt <= samples_per_gpu:
                current_prompt_samples.append((prompt, samples_for_prompt))
                current_samples += samples_for_prompt
            else:
                remaining = samples_per_gpu - current_samples
                if remaining > 0:
                    current_prompt_samples.append((prompt, remaining))
                prompts_with_samples.append(current_prompt_samples)
                current_prompt_samples = []
                if samples_for_prompt - remaining > 0:
                    current_prompt_samples.append((prompt, samples_for_prompt - remaining))
                current_samples = samples_for_prompt - remaining
        
        if current_prompt_samples:
            prompts_with_samples.append(current_prompt_samples)
        
        while len(prompts_with_samples) < world_size:
            prompts_with_samples.append([])
        
        my_prompts = prompts_with_samples[local_rank]
    else:
        my_prompts = [(prompt, opt.n_samples) for prompt in all_prompts]

    logger.info(f"GPU {local_rank} assigned {len(my_prompts)} prompts")

    # Initialize the SDXL pipeline
    # Make sure you have the correct repo and model name. For example:
    # "stabilityai/stable-diffusion-xl-base-1.0"
    # pipeline = Prompt2PromptPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", 
    #     torch_dtype=torch.float16
    # ).to(device)

    base = Prompt2PromptPipeline.from_pretrained(
    "/home/felix.boelter/recreategoods/diffusers/examples/text_to_image/sd-fashion-full-sdxl", torch_dtype=torch.float16
    ).to(device)
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    refiner.set_progress_bar_config(disable=True)

    # pipeline.load_lora_weights('/home/felix.boelter/recreategoods/diffusers/examples/text_to_image/sd-fashion-lora-sdxl/pytorch_lora_weights.safetensors')
    # Optionally enable torch.autocast if mixed-precision was requested
    # SDXL pipeline by default uses float16 if set
    if opt.mixed_precision:
        torch.set_default_dtype(torch.float16)

    # Create output directory
    out_dir = Path(opt.out_dir)
    if is_main_process:
        out_dir.mkdir(exist_ok=True, parents=True)
    if world_size > 1:
        dist.barrier()

    clip_similarity = ClipSimilarity().cuda()

    # Process prompts
    with torch.no_grad():
        for local_prompt_idx, (prompt, n_samples) in enumerate(my_prompts):
            all_prompts_length = len(all_prompts)
            global_prompt_idx = get_global_prompt_idx(local_prompt_idx, local_rank, world_size, all_prompts_length)
            logger.info(f"Processing prompt {global_prompt_idx + 1}/{all_prompts_length}")
            
            prompt_dir = out_dir.joinpath(f"{global_prompt_idx:07d}")
            prompt_dir.mkdir(exist_ok=True)

            prompt_info = {
                "prompt": prompt,
                "global_index": global_prompt_idx,
                "local_index": local_prompt_idx,
                "gpu_rank": local_rank if world_size > 1 else 0
            }
            with open(prompt_dir.joinpath("prompt.json"), "w") as fp:
                json.dump(prompt_info, fp, indent=2)

            results = {}
            pbar = tqdm(total=n_samples, desc="Samples", disable=not is_main_process)
            
            edit_types = ["refine"]
            for _ in range(n_samples):
                try:
                    torch.cuda.empty_cache()
                    seed = torch.randint(1 << 32, ()).item()
                    if seed in results:
                        continue
                    
                    generator = torch.Generator(device=device)
                    generator = generator.manual_seed(seed)
                    
                    chosen_edit_type = random.choice(edit_types)
                    n_self_replace_val = random.uniform(0.01, 0.6)
                    n_cross_replace_val = random.uniform(0.01, 0.6)
                    cfg_scale = opt.min_cfg + torch.rand(()).item() * (opt.max_cfg - opt.min_cfg)
                    cross_attention_kwargs = {"edit_type": chosen_edit_type,
                              "n_self_replace": n_self_replace_val,
                              "n_cross_replace": n_cross_replace_val,
                              }
                    images = base(
                        prompt=[prompt["original_caption"], prompt["resulting_caption"]],
                        negative_prompt="human, person, people, portrait, zoomed-in, close-up, closeup, face, head, eyes, mouth, nose, ears, hair, glitch, physically impossible, close-up on details, close-up, low depth of field",
                        num_inference_steps=opt.steps,
                        guidance_scale=cfg_scale,
                        num_images_per_prompt=1,
                        cross_attention_kwargs=cross_attention_kwargs,
                        denoising_end=0.8,
                        output_type="latent",
                        generator=generator
                    ).images  # This returns a list of PIL images
                    x0 = refiner(
                        prompt=prompt["original_caption"],
                        negative_prompt="human, person, people, portrait, zoomed-in, close-up, closeup, face, head, eyes, mouth, nose, ears, hair, glitch, physically impossible, close-up on details, close-up, low depth of field",
                        num_inference_steps=opt.steps,
                        guidance_scale=cfg_scale,
                        num_images_per_prompt=1,
                        denoising_start=0.8,
                        image=images[0]
                    ).images[0]
                    x1 = refiner(
                        prompt=prompt["resulting_caption"],
                        negative_prompt="human, person, people, portrait, zoomed-in, close-up, closeup, face, head, eyes, mouth, nose, ears, hair, glitch, physically impossible, close-up on details, close-up, low depth of field",
                        num_inference_steps=opt.steps,
                        guidance_scale=cfg_scale,
                        num_images_per_prompt=1,
                        denoising_start=0.8,
                        image=images[1]
                    ).images[0]
                    # Assuming we got two images: x0 and x1
                    # x0, x1 = images[0], images[1]

                    # Calculate CLIP similarity
                    # ClipSimilarity expects tensors. Convert PIL to tensor:
                    def pil_to_torch(img):
                        arr = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
                        # Normalize to [-1,1] if needed. The original code had a different range,
                        # but let's keep it simple and just call the similarity function directly.
                        # Adjust if your similarity function expects a certain range.
                        return arr.unsqueeze(0).cuda()

                    clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = clip_similarity(
                        pil_to_torch(x0),
                        pil_to_torch(x1),
                        [prompt["original_caption"]],
                        [prompt["resulting_caption"]]
                    )
                    
                    # Store results
                    results[seed] = dict(
                        image_0=x0,
                        image_1=x1,
                        clip_sim_0=clip_sim_0[0].item(),
                        clip_sim_1=clip_sim_1[0].item(),
                        clip_sim_dir=clip_sim_dir[0].item(),
                        clip_sim_image=clip_sim_image[0].item(),
                    )
                    
                    pbar.update(1)
                    
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"GPU {local_rank} encountered OOM, clearing cache and retrying")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.error(f"GPU {local_rank} encountered error: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise

            pbar.close()

            # Save results
            save_results(results, prompt_dir, opt)
            del results
            torch.cuda.empty_cache()

            logger.info(f"GPU {local_rank} completed prompt {global_prompt_idx + 1}/{all_prompts_length}")

    if world_size > 1:
        cleanup_distributed()
        
    logger.info(f"GPU {local_rank} completed all processing")

if __name__ == "__main__":
    main()