import torch
import json
from pathlib import Path
from PIL import Image
from typing import Dict
import lightning as pl
from metrics.viescore import VIEScore



class VIEScoreEvaluator(pl.LightningModule):
    def __init__(self, config : Dict):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.overall_score_threshold = self.config.sim_thresholds.overall_score_threshold
        self.viescore = VIEScore(backbone="qwen2", task="tie")  # Initialize VIEScore

    def forward(self, batch):
        """
        Evaluate a batch of prompts and their corresponding images.
        """
        prompt_dir, prompt, images = batch
        viescores = {}
        for data in images:
            seed, image1, image2 = data
            text_prompt = prompt["edit_instruction"]
            try:
                score_list = self.viescore.evaluate([image1, image2], text_prompt)
                sementics_score, quality_score, overall_score = score_list
                viescores[seed] = {
                    "sementics_score": sementics_score,
                    "quality_score": quality_score,
                    "overall_score": overall_score,
                }
            except Exception as e:
                print(f"Error computing VIEScore for seed {seed} in folder {prompt_dir}: {e}")
                continue

        # Save VIEScores to a JSON file
        viescores_path = prompt_dir / "viescores.json"
        with open(viescores_path, "w") as viescores_file:
            json.dump(viescores, viescores_file, indent=2)
        print(f"Saved VIEScores to {viescores_path}")

        # Remove low-score images and update metadata
        self.remove_low_score_images(prompt_dir, viescores, self.overall_score_threshold)

    @staticmethod
    def remove_low_score_images(prompt_dir: Path, viescores: Dict, threshold: float):
        """Remove images and metadata for seeds with overall_score below the threshold."""
        seeds_to_remove = [seed for seed, scores in viescores.items() if scores["overall_score"] < threshold]
        removed_scores = {}
        removed_images_dir = prompt_dir / "removed_images"
        removed_images_dir.mkdir(exist_ok=True)

        for seed in seeds_to_remove:
            image_0_path = prompt_dir.joinpath(f"{seed}_0.jpg")
            image_1_path = prompt_dir.joinpath(f"{seed}_1.jpg")
            removed_image_0_path = removed_images_dir.joinpath(f"{seed}_0.jpg")
            removed_image_1_path = removed_images_dir.joinpath(f"{seed}_1.jpg")

            if image_0_path.exists():
                image_0_path.rename(removed_image_0_path)
            if image_1_path.exists():
                image_1_path.rename(removed_image_1_path)
            if seed in viescores:
                removed_scores[seed] = viescores.pop(seed)  # Move scores to removed_scores

        # Update metadata.jsonl
        metadata_path = prompt_dir.joinpath("metadata.jsonl")
        if metadata_path.exists():
            with open(metadata_path, "r") as fp:
                metadata_entries = [json.loads(line) for line in fp]
            metadata_entries = [entry for entry in metadata_entries if entry["seed"] not in seeds_to_remove]
            with open(metadata_path, "w") as fp:
                for entry in metadata_entries:
                    fp.write(f"{json.dumps(entry)}\n")

        # Write updated viescores to viescores.json
        viescores_path = prompt_dir / "viescores.json"
        with open(viescores_path, "w") as viescores_file:
            json.dump(viescores, viescores_file, indent=2)

        # Save removed scores to removed_viescores.json
        removed_viescores_path = prompt_dir / "removed_viescores.json"
        if removed_scores:
            if removed_viescores_path.exists():
                with open(removed_viescores_path, "r") as fp:
                    existing_removed_scores = json.load(fp)
                removed_scores.update(existing_removed_scores)  # Combine with existing scores
            with open(removed_viescores_path, "w") as fp:
                json.dump(removed_scores, fp, indent=2)
        image_count = len(seeds_to_remove) * 2
        print(f"Moved {len(seeds_to_remove)} seeds and {image_count} images")

    def test_step(self, batch, batch_idx):
        """Process a batch of prompts."""
        self(batch)