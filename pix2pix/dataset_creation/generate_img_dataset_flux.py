from os import environ
environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import sys
sys.path.append("./")
sys.path.append("./flux")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from metrics.clip_similarity import ClipSimilarity
from flux.model import Flux
from torch import Tensor
from flux.modules.layers import DoubleStreamBlock

NSFW_THRESHOLD = 0.85

def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, t5, clip, nsfw_classifier

def to_pil(
    x: torch.Tensor,
) -> Image.Image:
    # bring into PIL format and save
    x = rearrange(x, "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    return img

def get_ancestral_step(sigma_from, sigma_to, noise_factor=0.3):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = noise_factor * min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    prompt_to_prompt: bool = False,
    p2p_threshold: float = 0.2,  # Portion of timesteps to apply P2P
):
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        # Check if the current step is within the P2P threshold
        apply_p2p = prompt_to_prompt and p2p_threshold > i / (len(timesteps) - 1)
        
        if apply_p2p:
            # Set a flag in the model or modify attention behavior as needed
            for module in model.modules():
                if isinstance(module, DoubleStreamBlock):
                    module.prompt_to_prompt = True
                    module.p2p_strength = 0.6
        
        # Predict the noise
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
        )
        
        if apply_p2p:
            # Reset the P2P flag to avoid affecting the next steps
            for module in model.modules():
                if isinstance(module, DoubleStreamBlock):
                    module.prompt_to_prompt = False
                    module.p2p_strength = None

        # Calculate the noise levels for Euler's method
        sigma_down, sigma_up = get_ancestral_step(t_curr, t_prev)
        dt = sigma_down - t_curr
        
        # Euler update: deterministic step
        img = img + dt * pred
        
        # Add stochastic noise if applicable
        if t_prev > 0:
            img = img + torch.randn_like(img) * sigma_up

    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Path to output dataset directory.")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to prompts .jsonl file.")
    parser.add_argument("--steps", type=int, default=15, help="Number of sampling steps.")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate per prompt.")
    parser.add_argument("--max-out-samples", type=int, default=4, help="Max number of output samples to save per prompt.")
    parser.add_argument("--clip-threshold", type=float, default=0.1, help="CLIP threshold for text-image similarity.")
    parser.add_argument("--clip-dir-threshold", type=float, default=0.1, help="CLIP threshold for directional similarity.")
    parser.add_argument("--clip-img-threshold", type=float, default=0.6, help="CLIP threshold for image-image similarity.")
    parser.add_argument("--n-partitions", type=int, default=1, help="Number of total partitions.")
    parser.add_argument("--partition", type=int, default=0, help="Partition index.")
    parser.add_argument("--min-p2p", type=float, default=0.2, help="Min prompt2prompt threshold.")
    parser.add_argument("--max-p2p", type=float, default=0.8,help="Max prompt2prompt threshold.")
    opt = parser.parse_args()

    global_seed = torch.randint(1 << 32, ()).item()
    print(f"Global seed: {global_seed}")
    torch.manual_seed(global_seed)
    offload = False
    device = "cuda"
    model, ae, t5, clip, nsfw_classifier = get_models("flux-schnell", device=device, offload=offload, is_schnell=True)
    model.cuda().eval()

    clip_similarity = ClipSimilarity().cuda()

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(opt.prompts_file) as fp:
        prompts = [json.loads(line) for line in fp]
    
    print(f"Partition index {opt.partition} ({opt.partition + 1} / {opt.n_partitions})")
    prompts = np.array_split(list(enumerate(prompts)), opt.n_partitions)[opt.partition]
    height, width = 512, 512
    print(f"Loaded {len(prompts)} prompts.")
    with torch.no_grad():
        for i, prompt in tqdm(prompts, desc="Processing Prompts"):
            prompt_dir = out_dir.joinpath(f"{i:07d}")
            prompt_dir.mkdir(exist_ok=True)

            with open(prompt_dir.joinpath("prompt.json"), "w") as fp:
                json.dump(prompt, fp)

            results = {}
            with tqdm(total=opt.n_samples, desc="Samples") as progress_bar:
                while len(results) < opt.n_samples:
                    seed = torch.randint(1 << 32, ()).item()
                    if seed in results:
                        continue
                    torch.manual_seed(seed)

                    # Generate initial noise
                    x = get_noise(2, height, width, device=device, dtype=torch.bfloat16, seed=seed)
                    if offload:
                        ae = ae.cpu()
                        torch.cuda.empty_cache()
                        t5, clip = t5.to(device), clip.to(device)
                    inp = prepare(t5, clip, x, prompt=[prompt["input"], prompt["output"]])
                    timesteps = get_schedule(opt.steps, inp["img"].shape[1])
                    if offload:
                        t5, clip = t5.cpu(), clip.cpu()
                        torch.cuda.empty_cache()
                        model = model.to(device)

                    # Denoise with FLUX
                    p2p_threshold = opt.min_p2p + torch.rand(()).item() * (opt.max_p2p - opt.min_p2p)
                    x = denoise(model, **inp, timesteps=timesteps, prompt_to_prompt=True, p2p_threshold=p2p_threshold)
                    if offload:
                        model.cpu()
                        torch.cuda.empty_cache()
                        ae.decoder.to(x.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        x_samples = ae.decode(unpack(x, height, width))
                    x_samples = torch.clamp(x_samples, -1.0, 1.0)
                    x0, x1 = x_samples[0], x_samples[1]

                    clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = clip_similarity(
                        x0[None], x1[None], [prompt["input"]], [prompt["output"]]
                    )
                    results[seed] = dict(
                        image_0=to_pil(x0),
                        image_1=to_pil(x1),
                        clip_sim_0=clip_sim_0[0].item(),
                        clip_sim_1=clip_sim_1[0].item(),
                        clip_sim_dir=clip_sim_dir[0].item(),
                        clip_sim_image=clip_sim_image[0].item(),
                    )
                    progress_bar.update()

            # CLIP filter to get best samples for each prompt.
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

    print("Dataset generation completed.")

if __name__ == "__main__":
    main()