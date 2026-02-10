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

from diffusers import DiffusionPipeline
from metrics.clip_similarity import ClipSimilarity
from utils.flux_processor import extract_key_changes

def setup_logging(rank, log_dir="../logs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join(log_dir, timestamp)
    os.makedirs(log_folder, exist_ok=True)
    
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
        prompts_per_gpu = all_prompts_length // world_size
        return (prompts_per_gpu * local_rank) + local_prompt_idx
    return local_prompt_idx




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-out-samples", type=int, default=4)
    parser.add_argument("--clip-threshold", type=float, default=0.1)
    parser.add_argument("--clip-dir-threshold", type=float, default=0.1)
    parser.add_argument("--clip-img-threshold", type=float, default=0.8)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--min-cfg", type=float, default=2.5, help="Min classifier free guidance scale.")
    parser.add_argument("--max-cfg", type=float, default=8.5, help="Max classifier free guidance scale.")
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

    with open(opt.prompts_file, 'r') as fp:
        all_prompts = json.load(fp)
    
    total_samples = opt.n_samples * len(all_prompts)
    
    if world_size > 1:
        samples_per_gpu = total_samples // world_size
        prompts_per_gpu = int(np.ceil(len(all_prompts) / world_size))
        if is_main_process:
            logger.info(f"Total samples: {total_samples}, Samples per GPU: {samples_per_gpu}")
        
        prompts_with_samples = [[] for _ in range(world_size)]
        current_rank = 0
        
        for i, prompt in enumerate(all_prompts):
            samples_for_prompt = opt.n_samples
            if current_rank >= world_size:
                current_rank = 0
            if len(prompts_with_samples[current_rank]) >= prompts_per_gpu:
                current_rank += 1
            
            prompts_with_samples[current_rank].append((i, prompt, samples_for_prompt))

        # Assign prompts to the current rank
        my_prompts = prompts_with_samples[local_rank]
    else:
        my_prompts = [(prompt, opt.n_samples) for prompt in all_prompts]

    logger.info(f"GPU {local_rank} assigned {len(my_prompts)} prompts")
    # Initialize the Shuttle pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "shuttleai/shuttle-3.1-aesthetic", 
        torch_dtype=torch.bfloat16,
        custom_pipeline='./utils/prompt_to_prompt_shuttle.py',
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_lora_weights('/home/felix.boelter/recreategoods/ai-toolkit/output/flux_lora_v3/flux_lora_v3.safetensors')

    if opt.mixed_precision:
        torch.set_default_dtype(torch.float16)

    out_dir = Path(opt.out_dir)
    if is_main_process:
        out_dir.mkdir(exist_ok=True, parents=True)
    if world_size > 1:
        dist.barrier()

    clip_similarity = ClipSimilarity().cuda()

    with torch.no_grad():
        for local_prompt_idx, (global_prompt_idx, prompt, n_samples) in enumerate(my_prompts):
            all_prompts_length = len(all_prompts)
            # global_prompt_idx = get_global_prompt_idx(local_prompt_idx, local_rank, world_size, all_prompts_length)
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
            
            for _ in range(n_samples):
                try:
                    torch.cuda.empty_cache()
                    seed = torch.randint(1 << 32, ()).item()
                    if seed in results:
                        continue
                    
                    generator = torch.Generator(device=device)
                    generator = generator.manual_seed(seed)
                    
                    cfg_scale = opt.min_cfg + torch.rand(()).item() * (opt.max_cfg - opt.min_cfg)
                    p2p_strength = np.random.uniform(low=0.6, high=1.01)
                    p2p_threshold = np.random.uniform(low=0.6, high=0.9)
                    # Generate two images: one from the original caption and one from the resulting caption
                    # by passing them as a list. Each prompt in the list produces one image.
                    amplify, suppress, shared  = extract_key_changes(prompt["original_caption"], prompt["resulting_caption"])
                    joint_attention_kwargs = {
                        'self_replace_steps': p2p_threshold,
                        'resulting_caption': prompt["resulting_caption"],
                        'words_shared': shared,
                        "words_amplification": amplify,
                        "words_suppression": suppress,
                        "shared_factor": 1.0,
                        "amplification_factor": 1.0,
                        "suppression_factor": 1.0,
                    }
                    images = pipe(
                        [prompt["original_caption"], prompt["resulting_caption"]],
                        num_inference_steps=opt.steps,
                        height=1024,
                        width=1024,
                        guidance_scale=cfg_scale,
                        num_images_per_prompt=1,
                        max_sequence_length=256,
                        generator=generator,
                        joint_attention_kwargs=joint_attention_kwargs
                    ).images

                    x0, x1 = images[0], images[1]

                    def pil_to_torch(img):
                        arr = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
                        return arr.unsqueeze(0).cuda()

                    clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = clip_similarity(
                        pil_to_torch(x0),
                        pil_to_torch(x1),
                        [prompt["original_caption"]],
                        [prompt["resulting_caption"]]
                    )
                    
                    results[seed] = dict(
                        image_0=x0,
                        image_1=x1,
                        clip_sim_0=clip_sim_0[0].item(),
                        clip_sim_1=clip_sim_1[0].item(),
                        clip_sim_dir=clip_sim_dir[0].item(),
                        clip_sim_image=clip_sim_image[0].item(),
                        shared=shared,
                        amplify=amplify,
                        suppress=suppress,
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

            save_results(results, prompt_dir, opt)
            del results
            torch.cuda.empty_cache()

            logger.info(f"GPU {local_rank} completed prompt {global_prompt_idx + 1}/{all_prompts_length}")

    if world_size > 1:
        cleanup_distributed()
        
    logger.info(f"GPU {local_rank} completed all processing")

if __name__ == "__main__":
    main()