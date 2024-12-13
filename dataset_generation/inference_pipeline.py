import sys
sys.path.append("./")
import lightning as pl
from diffusers import DiffusionPipeline
from utils.processor import extract_key_changes
from metrics.clip_similarity import ClipSimilarity
from pathlib import Path
import torch
import json
import logging
import numpy as np
from typing import Dict
import os
from tqdm import tqdm

class PromptProcessor(pl.LightningModule):
    def __init__(self, out_dir: str, steps: int, min_cfg: float, max_cfg: float, min_threshold : float, max_threshold: float, mixed_precision: bool, clip_thresholds : Dict, is_main_process: bool, rank: int):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.steps = steps
        self.min_cfg = min_cfg
        self.max_cfg = max_cfg
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.mixed_precision = mixed_precision
        self.clip_similarity = ClipSimilarity().cuda()
        self.clip_thresholds = clip_thresholds
        self.is_main_process = is_main_process
        self.rank = rank
        # Set up the Diffusion pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            "shuttleai/shuttle-3.1-aesthetic",
            torch_dtype=torch.bfloat16,
            custom_pipeline='./utils/ptp_pipeline.py'
        ).to("cuda")
        self.pipe.load_lora_weights('aihpi/flux-fashion-lora')
        self.pipe.set_progress_bar_config(disable=True)

        if mixed_precision:
            torch.set_default_dtype(torch.float16)

    def compute_clip_similarity(self, image_0, image_1, prompt_0, prompt_1):
        """
        Compute CLIP similarity metrics for a pair of images and prompts.
        """
        def pil_to_tensor(img):
            arr = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            return arr.unsqueeze(0).cuda()

        # Convert PIL images to tensors
        img_tensor_0 = pil_to_tensor(image_0)
        img_tensor_1 = pil_to_tensor(image_1)

        # Compute similarity metrics
        clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = self.clip_similarity(
            img_tensor_0, img_tensor_1, [prompt_0], [prompt_1]
        )

        return {
            "clip_sim_0": clip_sim_0[0].item(),
            "clip_sim_1": clip_sim_1[0].item(),
            "clip_sim_dir": clip_sim_dir[0].item(),
            "clip_sim_image": clip_sim_image[0].item(),
        }
    
    @staticmethod
    def save_results(results: Dict, prompt_dir: Path, opt: Dict):
        """Save generated images and metadata."""
        metadata = [
            (result["clip_sim_dir"], seed)
            for seed, result in results.items()
            # if result["clip_sim_image"] >= opt["clip_img_threshold"]
            # and result["clip_sim_dir"] >= opt["clip_dir_threshold"]
            # and result["clip_sim_0"] >= opt["clip_threshold"]
            # and result["clip_sim_1"] >= opt["clip_threshold"]
        ]
        
        metadata.sort(reverse=True)
        for _, seed in metadata[: opt["max_out_samples"]]:
            result = results[seed]
            image_0 = result.pop("image_0")
            image_1 = result.pop("image_1")
            image_0.save(prompt_dir.joinpath(f"{seed}_0.jpg"), quality=100)
            image_1.save(prompt_dir.joinpath(f"{seed}_1.jpg"), quality=100)
            with open(prompt_dir.joinpath(f"metadata.jsonl"), "a") as fp:
                fp.write(f"{json.dumps(dict(seed=seed, **result))}\n")

    def test_step(self, batch, batch_idx):
        prompt_idx, prompt, n_samples = batch
        prompt_dir = self.out_dir.joinpath(f"{prompt_idx:07d}")
        prompt_dir.mkdir(exist_ok=True, parents=True)
        prompt_info = {
                "prompt": prompt,
                "global_index": prompt_idx,
                "local_index": batch_idx,
                "gpu_rank": self.rank 
            }
        with open(prompt_dir.joinpath("prompt.json"), "w") as fp:
            json.dump(prompt_info, fp, indent=2)
        results = {}
        pbar = tqdm(total=n_samples, desc="Samples", disable=not self.is_main_process)

        for _ in range(n_samples):
            seed = torch.randint(1 << 32, ()).item()
            if seed in results: continue
            generator = torch.Generator(device="cuda").manual_seed(seed)
            cfg_scale = self.min_cfg + torch.rand(()).item() * (self.max_cfg - self.min_cfg)
            p2p_threshold = np.random.uniform(low=0.6, high=0.9)
            amplify, suppress, shared  = extract_key_changes(prompt["original_caption"], prompt["resulting_caption"])

            # Generate two images: one from the original caption and one from the resulting caption
            # by passing them as a list. Each prompt in the list produces one image.
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

            # Generate images
            images = self.pipe(
                [prompt["original_caption"], prompt["resulting_caption"]],
                num_inference_steps=self.steps,
                guidance_scale=cfg_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                num_images_per_prompt=1,
                generator=generator,
            ).images

            # Compute CLIP similarity
            clip_metrics = self.compute_clip_similarity(
                images[0], images[1], 
                prompt["original_caption"], prompt["resulting_caption"]
            )

            results[seed] = {
                **clip_metrics,
                "shared": shared,
                "amplify": amplify,
                "suppress": suppress,
                "image_0": images[0],
                "image_1": images[1],
            }
            pbar.update(1)
        # Save results using the utility method
        self.save_results(results, prompt_dir, self.clip_thresholds)

        return results