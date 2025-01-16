from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as pl
import json
import os
import random
import torchvision.transforms.functional as F



class EditDataset(Dataset):
    def __init__(self, path, metadata_file, min_resize_res, max_resize_res, crop_res, flip_prob=0.0, to_transform=True):
        """
        Dataset for image editing tasks.

        Args:
            path (str or Path): Path to the dataset split directory (e.g., train, val, test).
            metadata_file (str or Path): Path to the metadata JSON file corresponding to the split.
            min_resize_res (int): Minimum resize resolution for images.
            max_resize_res (int): Maximum resize resolution for images.
            crop_res (int): Crop resolution for images.
            flip_prob (float): Probability of horizontal flipping during augmentation.
        """
        self.data_dir = Path(path)
        self.metadata_file = Path(metadata_file)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        # Load metadata
        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)

        self.to_transform : bool = to_transform

    
    def __len__(self):
        return len(self.metadata)
    
    def paired_transform(self, input_image, output_image):
        if random.random() < self.flip_prob:
            input_image = F.hflip(input_image)
            output_image = F.hflip(output_image)
            
        input_image = F.resize(input_image, (self.min_resize_res, self.max_resize_res))
        output_image = F.resize(output_image, (self.min_resize_res, self.max_resize_res))

        input_image = F.to_tensor(input_image)
        output_image = F.to_tensor(output_image)

        input_image = 2.0 * input_image - 1.0
        output_image = 2.0 * output_image - 1.0

        return input_image, output_image
    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict: A dictionary containing:
                - "input_image": Transformed input image tensor.
                - "output_image": Transformed output image tensor.
                - "edit_instruction": Tokenized edit instruction.
        """
        item = self.metadata[idx]

        # Load and transform input and output images
        input_image_path = self.data_dir / item["input_image"]
        output_image_path = self.data_dir / item["output_image"]

        input_image = Image.open(input_image_path).convert("RGB")
        output_image = Image.open(output_image_path).convert("RGB")
        if self.to_transform:
            input_image, output_image = self.paired_transform(input_image, output_image)
        else:
            input_image = transforms.ToTensor()(input_image)
            output_image = transforms.ToTensor()(output_image)

        edit_instruction = item["edit_instruction"]

        return {
            "input_image": input_image,
            "output_image": output_image,
            "edit_instruction": edit_instruction,
        }

class FLUXDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir, min_resize_res, max_resize_res, crop_res, flip_prob):
        """
        Data module for FLUX training, validation, and testing.

        Args:
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for dataloaders.
            data_dir (str or Path): Path to the main dataset directory.
            min_resize_res (int): Minimum resize resolution for images.
            max_resize_res (int): Maximum resize resolution for images.
            crop_res (int): Crop resolution for images.
            flip_prob (float): Probability of horizontal flipping for training.
            tokenizer: Tokenizer object for processing edit instructions.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

    def setup(self, stage=None):
        """
        Prepare datasets for the specified stage.

        Args:
            stage (str): One of {"fit", "test", None}.
        """
        if stage in (None, "fit"):
            self.train_dataset = EditDataset(
                path=self.data_dir / "train",
                metadata_file=os.path.join(self.data_dir, "train", "train_metadata.json"),
                min_resize_res=self.min_resize_res,
                max_resize_res=self.max_resize_res,
                crop_res=self.crop_res,
                flip_prob=self.flip_prob,
            )
            self.val_dataset = EditDataset(
                path=self.data_dir / "val",
                metadata_file=os.path.join(self.data_dir, "val", "val_metadata.json"),
                min_resize_res=self.min_resize_res,
                max_resize_res=self.max_resize_res,
                crop_res=self.crop_res,
                flip_prob=0.0,  
                to_transform=False,
            )
        if stage in (None, "test"):
            self.test_dataset = EditDataset(
                path=self.data_dir / "test",
                metadata_file=os.path.join(self.data_dir, "test", "test_metadata.json"),
                min_resize_res=self.min_resize_res,
                max_resize_res=self.max_resize_res,
                crop_res=self.crop_res,
                flip_prob=0.0, 
                to_transform=False,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)