import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import argparse
import json
import sys
from typing import Dict, Optional, Union
import logging, traceback
sys.path.append("./")
sys.path.append("./flux")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5
from flux.sampling import get_noise, get_schedule, prepare, unpack
from metrics.clip_similarity import ClipSimilarity
from flux.model import Flux
from torch import Tensor
from flux.modules.layers import DoubleStreamBlock
from datetime import datetime

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

@dataclass
class DistributedModels:
    """Container for distributed models - optimized for VRAM usage."""
    model: nn.Module
    decoder: nn.Module  # Only store decoder instead of full AE
    t5: nn.Module
    clip: nn.Module

def wrap_model_in_ddp(
    model: nn.Module,
    device: torch.device,
    local_rank: Optional[int],
    model_name: str,
    ignore_modules: Optional[list[str]] = None
) -> nn.Module:
    """Utility function to wrap a model in DDP."""
    if local_rank is None:
        return model
        
    try:
        model = model.to(device)
        wrapped_model = DDP(
            model,
            device_ids=[local_rank])
        print(f"Successfully wrapped {model_name} in DDP")
        return wrapped_model
    except Exception as e:
        print(f"Failed to wrap {model_name} in DDP: {str(e)}. Using non-DDP model.")
        return model.to(device)

def setup_distributed_models(
    name: str,
    device: torch.device,
    offload: bool,
    is_schnell: bool,
    local_rank: Optional[int] = None
) -> DistributedModels:
    """Initialize and wrap models in DDP - optimized for VRAM usage."""
    # Initialize models on CPU first
    t5 = load_t5(device=device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu")
    ae = load_ae(name, device=device)
    
    
    if offload:
        device_map = "cpu"
    else:
        # Wrap models in DDP if distributed
        if local_rank is not None:
            print(f"Setting up distributed models on rank {local_rank}")
            # Wrap models in DDP
            model = wrap_model_in_ddp(model, device, local_rank, "Main Model")
        else:
            # Move models to GPU if not distributed
            model = model.to(device)
            ae = ae.to(device)
    
    # Force CUDA cache clear after model loading
    torch.cuda.empty_cache()
    
    return DistributedModels(
        model=model,
        decoder=ae,
        t5=t5,
        clip=clip
    )

def get_base_model(model: nn.Module) -> nn.Module:
    """Get base model from potentially DDP-wrapped model."""
    return model.module if isinstance(model, DDP) else model

def prepare_with_ddp(t5: nn.Module, clip: nn.Module, x: torch.Tensor, prompt: list) -> dict:
    """Prepare inputs handling DDP-wrapped models."""
    t5_base = get_base_model(t5)
    clip_base = get_base_model(clip)
    return prepare(t5_base, clip_base, x, prompt)

def get_ancestral_step(sigma_from, sigma_to, noise_factor=0.3):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = noise_factor * min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up

def to_pil(x: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    x = rearrange(x, "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    return img

def denoise(
    model: nn.Module,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    prompt_to_prompt: bool = False,
    p2p_threshold: float = 0.2,
    noise_factor: float = 0.3,
):
    """Denoise images with proper DDP model handling."""
    with torch.no_grad():
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            
            apply_p2p = prompt_to_prompt and p2p_threshold > i / (len(timesteps) - 1)
            
            if apply_p2p:
                base_model = get_base_model(model)
                for module in base_model.modules():
                    if isinstance(module, DoubleStreamBlock):
                        module.prompt_to_prompt = True
                        module.p2p_strength = 0.6
            
            torch.cuda.empty_cache()
            
            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
            )
            
            if apply_p2p:
                base_model = get_base_model(model)
                for module in base_model.modules():
                    if isinstance(module, DoubleStreamBlock):
                        module.prompt_to_prompt = False
                        module.p2p_strength = None

            sigma_down, sigma_up = get_ancestral_step(t_curr, t_prev, noise_factor=noise_factor)
            dt = sigma_down - t_curr
            
            img = img + dt * pred
            
            if t_prev > 0:
                img = img + torch.randn_like(img) * sigma_up

        return img.detach()

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

def get_global_prompt_idx(local_prompt_idx: int, local_rank: int, world_size: int, all_prompts_length : int) -> int:
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
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-out-samples", type=int, default=4)
    parser.add_argument("--clip-threshold", type=float, default=0.1)
    parser.add_argument("--clip-dir-threshold", type=float, default=0.1)
    parser.add_argument("--clip-img-threshold", type=float, default=0.6)
    parser.add_argument("--min-p2p", type=float, default=0.2)
    parser.add_argument("--max-p2p", type=float, default=0.8)
    parser.add_argument("--noise-factor", type=float, default=0.3)
    parser.add_argument("--mixed-precision", action="store_true")
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
    with open(opt.prompts_file) as fp:
        all_prompts = [json.loads(line) for line in fp]
    
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

    # Initialize models
    distributed_models = setup_distributed_models(
        "flux-schnell",
        device=device,
        offload=False,
        is_schnell=True,
        local_rank=local_rank if world_size > 1 else None
    )
    
    model = distributed_models.model
    decoder = distributed_models.decoder
    t5 = distributed_models.t5
    clip = distributed_models.clip

    # Create output directory
    out_dir = Path(opt.out_dir)
    if is_main_process:
        out_dir.mkdir(exist_ok=True, parents=True)
    if world_size > 1:
        dist.barrier()

    height, width = 512, 512
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
            
            while len(results) < n_samples:
                try:
                    torch.cuda.empty_cache()
                    seed = torch.randint(1 << 32, ()).item()
                    if seed in results:
                        continue
                    
                    # Generate sample
                    x = get_noise(2, height, width, device=device, dtype=torch.bfloat16, seed=seed)
                    
                    inp = prepare_with_ddp(t5, clip, x, prompt=[prompt["input"], prompt["output"]])
                    timesteps = get_schedule(opt.steps, inp["img"].shape[1])

                    p2p_threshold = opt.min_p2p + torch.rand(()).item() * (opt.max_p2p - opt.min_p2p)
                    x = denoise(
                        model, 
                        **inp, 
                        timesteps=timesteps, 
                        prompt_to_prompt=True, 
                        p2p_threshold=p2p_threshold,
                        noise_factor=opt.noise_factor
                    )
                    
                    # Decode
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        x_samples = decoder.decode(unpack(x, height, width))
                        x_samples = torch.clamp(x_samples, -1.0, 1.0)
                    x0, x1 = x_samples[0], x_samples[1]

                    # Calculate CLIP similarity
                    clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = clip_similarity(
                        x0[None], x1[None], [prompt["input"]], [prompt["output"]]
                    )
                    
                    # Clean up intermediate tensors
                    del x, inp, x_samples
                    torch.cuda.empty_cache()
                    
                    # Store results
                    results[seed] = dict(
                        image_0=to_pil(x0),
                        image_1=to_pil(x1),
                        clip_sim_0=clip_sim_0[0].item(),
                        clip_sim_1=clip_sim_1[0].item(),
                        clip_sim_dir=clip_sim_dir[0].item(),
                        clip_sim_image=clip_sim_image[0].item(),
                    )
                    
                    del x0, x1, clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image
                    torch.cuda.empty_cache()
                    
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