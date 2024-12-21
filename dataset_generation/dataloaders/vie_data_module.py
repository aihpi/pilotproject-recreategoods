import os
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict
import torch
import lightning as pl
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class VIEScoreDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.folders = [folder for folder in self.dataset_dir.iterdir() if folder.is_dir()]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        metadata_path = folder / "metadata.jsonl"
        prompt_path = folder / "prompt.json"

        if not metadata_path.exists() or not prompt_path.exists():
            raise FileNotFoundError(f"Metadata or prompt.json missing in {folder}")

        # Load metadata and prompt
        with open(metadata_path, "r") as metadata_file:
            metadata_entries = [json.loads(line) for line in metadata_file]
        with open(prompt_path, "r") as prompt_file:
            prompt_data = json.load(prompt_file)["prompt"]

        # Collect images and associated seeds
        image_pairs = []
        for entry in metadata_entries:
            seed = entry["seed"]
            image1_path = folder / f"{seed}_0.jpg"
            image2_path = folder / f"{seed}_1.jpg"
            if image1_path.exists() and image2_path.exists():
                image1 = Image.open(image1_path)
                image2 = Image.open(image2_path)
                image_pairs.append((seed, image1, image2))
            else:
                print(f"Missing images for seed {seed} in {folder}")

        return folder, prompt_data, image_pairs


class VIEScoreDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: str, world_size: int, local_rank: int, num_workers: int = 4):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = VIEScoreDataset(self.dataset_dir)

    def my_collate_fn(self, batch):
        folders, prompts, image_pairs = zip(*batch)  # Unpack batch items
        return folders[0], prompts[0], image_pairs[0]

    def test_dataloader(self):
        sampler = DistributedSampler(
            self.dataset, 
            num_replicas=self.world_size, 
            rank=self.local_rank, 
            drop_last=False
        )
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=1, 
            shuffle=False, 
            persistent_workers=True, 
            collate_fn=self.my_collate_fn, 
            sampler=sampler, 
            num_workers=self.num_workers
        )