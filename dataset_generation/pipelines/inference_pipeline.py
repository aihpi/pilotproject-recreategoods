import sys
sys.path.append("./")
import lightning as pl
from diffusers import DiffusionPipeline
from utils.processor import extract_key_changes
from pathlib import Path
from metrics.clip_similarity import ClipSimilarity
from metrics.dinov2 import DINOv2Similarity
import torch
import json
import logging
import numpy as np
from typing import Dict
import os
from tqdm import tqdm


class PromptProcessor(pl.LightningModule):
    def __init__(self, config, is_main_process: bool, rank: int):
        super().__init__()
        self.config = config
        self.out_dir = Path(config.output_dir)
        self.steps = config.generation.steps
        self.min_cfg = config.generation.cfg_min
        self.max_cfg = config.generation.cfg_max
        self.min_threshold = config.generation.p2p_threshold_min
        self.max_threshold = config.generation.p2p_threshold_max
        print(f"Using similarity model: {config.similarity_model.name}")
        if config.similarity_model.name == "clip":
            self.similarity_model = ClipSimilarity(config.similarity_model.variant).cuda()
        elif config.similarity_model.name == "dino":
            self.similarity_model = DINOv2Similarity("facebook/dinov2-base").cuda()
        else:
            raise ValueError(f"Unsupported similarity model: {config.similarity_model.name}")

        self.sim_thresholds = config.sim_thresholds
        self.is_main_process = is_main_process
        self.rank = rank

        # Initialize the Diffusion pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            config.model.name,
            torch_dtype=getattr(torch, config.model.dtype),
            custom_pipeline='./utils/ptp_pipeline.py'
        ).to("cuda")
        self.pipe.load_lora_weights(config.model.lora_weights)
        self.pipe.set_progress_bar_config(disable=True)


    def compute_similarity(self, image_0, image_1, prompt_0, prompt_1):
        """
        Compute similarity metrics for a pair of images and prompts using the selected similarity model.
        """
        def pil_to_tensor(img):
            arr = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            return arr.unsqueeze(0).cuda()

        # Convert PIL images to tensors
        img_tensor_0 = pil_to_tensor(image_0)
        img_tensor_1 = pil_to_tensor(image_1)

        if isinstance(self.similarity_model, ClipSimilarity):
            # Compute similarity metrics using CLIP
            sim_0, sim_1, sim_dir, sim_image = self.similarity_model(
                img_tensor_0, img_tensor_1, [prompt_0], [prompt_1]
            )
        elif isinstance(self.similarity_model, DINOv2Similarity):
            # Compute similarity metrics using DINOv2
            sim_image, sim_dir = self.similarity_model(img_tensor_0, img_tensor_1)
            sim_0 = sim_1 = sim_dir  # DINOv2 does not use text, so replicate sim_dir
        else:
            raise ValueError("Unsupported similarity model instance.")


        return {
            "sim_0": sim_0[0].item(),
            "sim_1": sim_1[0].item(),
            "sim_dir": sim_dir[0].item(),
            "sim_image": sim_image[0].item(),
        }
    
    @staticmethod
    def save_results(results: Dict, prompt_dir: Path, opt: Dict, viescores: Dict = None):
        """Save generated images and metadata."""
        if opt["enable_filtering"] != True:
            metadata = [(result["sim_dir"], seed) for seed, result in results.items()]
            metadata.sort(reverse=True)
            for _, seed in metadata:
                result = results[seed]
                image_0 = result.pop("image_0")
                image_1 = result.pop("image_1")
                image_0.save(prompt_dir.joinpath(f"{seed}_0.jpg"), quality=100)
                image_1.save(prompt_dir.joinpath(f"{seed}_1.jpg"), quality=100)
                with open(prompt_dir.joinpath(f"metadata.jsonl"), "a") as fp:
                    fp.write(f"{json.dumps(dict(seed=seed, **result))}\n")
        else:
            metadata = [
                (result["sim_dir"], seed)
                for seed, result in results.items()
                if result["sim_image"] >= opt["img_threshold"]
                and result["sim_dir"] >= opt["dir_threshold"]
                and result["sim_0"] >= opt["clip_threshold"]
                and result["sim_1"] >= opt["clip_threshold"]
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

    # @staticmethod
    # def remove_low_score_images(results: Dict, prompt_dir: Path, viescores: Dict, threshold: float):
    #     """Remove images and metadata for seeds with overall_score below the threshold."""
    #     seeds_to_remove = [seed for seed, scores in viescores.items() if scores["overall_score"] < threshold]
    #     for seed in seeds_to_remove:
    #         image_0_path = prompt_dir.joinpath(f"{seed}_0.jpg")
    #         image_1_path = prompt_dir.joinpath(f"{seed}_1.jpg")
    #         if image_0_path.exists():
    #             image_0_path.unlink()
    #         if image_1_path.exists():
    #             image_1_path.unlink()
    #         if seed in results:
    #             del results[seed]
    #         if seed in viescores:
    #             del viescores[seed]
    #     # Update metadata.jsonl
    #     metadata_path = prompt_dir.joinpath("metadata.jsonl")
    #     if metadata_path.exists():
    #         with open(metadata_path, "r") as fp:
    #             metadata_entries = [json.loads(line) for line in fp]
    #         metadata_entries = [entry for entry in metadata_entries if entry["seed"] not in seeds_to_remove]
    #         with open(metadata_path, "w") as fp:
    #             for entry in metadata_entries:
    #                 fp.write(f"{json.dumps(entry)}\n")
    #     # Write updated viescores to viescores.json
    #     viescores_path = prompt_dir / "viescores.json"
    #     with open(viescores_path, "w") as viescores_file:
    #         json.dump(viescores, viescores_file, indent=2)

    def test_step(self, batch, batch_idx):
        def create_scale(min, max):
            return min + torch.rand(()).item() * (max - min)
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
            cfg_scale = create_scale(self.min_cfg, self.max_cfg)
            p2p_threshold = create_scale(self.min_threshold, self.max_threshold)
            amplify, suppress, shared  = extract_key_changes(prompt["original_caption"], prompt["resulting_caption"])
            amplify_factor = create_scale(1.0, self.config.attention.amplification_factor)
            suppress_factor = create_scale(self.config.attention.suppression_factor, 1.0)
            shared_factor = create_scale(1.0, self.config.attention.shared_factor)
            joint_attention_kwargs = {
                'self_replace_steps': p2p_threshold,
                'resulting_caption': prompt["resulting_caption"],
                'words_shared': shared,
                "words_amplification": amplify,
                "words_suppression": suppress,
                "shared_factor": shared_factor,
                "amplification_factor": amplify_factor,
                "suppression_factor": suppress_factor,
            }

            # Generate images
            images = self.pipe(
                [prompt["original_caption"], prompt["resulting_caption"]],
                num_inference_steps=self.steps,
                guidance_scale=cfg_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                num_images_per_prompt=1,
                generator=generator,
                height=self.config.generation.image_size.height,
                width=self.config.generation.image_size.width,
            ).images

            # Compute CLIP similarity
            sim_metrics = self.compute_similarity(
                images[0], images[1], 
                prompt["original_caption"], prompt["resulting_caption"]
            )

            results[seed] = {
                **sim_metrics,
                "shared": shared,
                "amplify": amplify,
                "suppress": suppress,
                "amplify_factor": amplify_factor,
                "suppress_factor": suppress_factor,
                "shared_factor": shared_factor,
                "p2p_threshold": p2p_threshold,
                "cfg_scale": cfg_scale,
                "self_replace_steps": p2p_threshold,
                "image_0": images[0],
                "image_1": images[1],
            }
            pbar.update(1)
        # Save results using the utility method
        self.save_results(results, prompt_dir, self.sim_thresholds)

        return results