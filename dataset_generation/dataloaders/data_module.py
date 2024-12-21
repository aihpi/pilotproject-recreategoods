import json
import torch
import lightning as pl
from typing import List, Dict
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        idx, prompt, num_samples = self.data[idx]
        return (
            idx,  # Index
            prompt,
            num_samples,  # Fixed number of samples
        )
    
class PromptDataModule(pl.LightningDataModule):
    def __init__(self, prompts_file: str, n_samples: int, world_size: int, local_rank: int, num_workers: int = 27):
        super().__init__()
        self.prompts_file = prompts_file
        self.n_samples = n_samples
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_workers = num_workers

    def setup(self, stage=None):
        prepend_phrase = "Neutral Gray Background, Wide-angle, high quality, detailed, ultrarealistic photography, "
        with open(self.prompts_file, 'r') as fp:
            self.all_prompts = json.load(fp)
        for prompt in self.all_prompts:
            if 'original_caption' in prompt and not prompt['original_caption'].startswith(prepend_phrase):
                prompt['original_caption'] = prepend_phrase + prompt['original_caption']
            if 'resulting_caption' in prompt and not prompt['resulting_caption'].startswith(prepend_phrase):
                prompt['resulting_caption'] = prepend_phrase + prompt['resulting_caption']
        self.my_prompts = [(i, prompt, self.n_samples) for i, prompt in enumerate(self.all_prompts)]
        
    def my_collate_fn(self, batch):
        indices, prompts, n_samples = zip(*batch)  # Unpack batch items
        return (indices[0], prompts[0], n_samples[0])
    
    def test_dataloader(self):
        dataset = PromptDataset(self.my_prompts)
        sampler = DistributedSampler(
            dataset, 
            num_replicas=self.world_size, 
            rank=self.local_rank, 
            drop_last=False
        )
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, persistent_workers=True, collate_fn=self.my_collate_fn, sampler=sampler, num_workers=self.num_workers)