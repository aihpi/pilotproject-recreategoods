import json
from pathlib import Path
from torch.utils.data import random_split
from PIL import Image
import shutil
from argparse import ArgumentParser
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class ImageEditDataset:
    def __init__(self, dataset_dir):
        """
        Custom dataset to load pairs of images and their corresponding edit instructions.

        Args:
            dataset_dir (Path): Path to the dataset directory.
        """
        self.dataset_dir = Path(dataset_dir)
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load the dataset samples by parsing `seeds.json` and corresponding `prompt.json` files.

        Returns:
            List[Dict]: List of dictionaries, each containing image paths and metadata.
        """
        # Load the seeds.json file
        with open(self.dataset_dir / "seeds.json", "r") as f:
            seeds = json.load(f)

        samples = {}
        for class_name, class_seeds in seeds:
            prompt_file = self.dataset_dir / class_name / "prompt.json"
            if not prompt_file.exists():
                continue

            # Load prompt.json for the current class
            with open(prompt_file, "r") as pf:
                prompt_data = json.load(pf)

            class_samples = []
            for seed in class_seeds:
                # Prepare paths for input (_0) and output (_1) images
                input_image_path = self.dataset_dir / class_name / f"{seed}_0.jpg"
                output_image_path = self.dataset_dir / class_name / f"{seed}_1.jpg"

                if input_image_path.exists() and output_image_path.exists():
                    class_samples.append({
                        "input_image": input_image_path,
                        "output_image": output_image_path,
                        "original_caption": prompt_data["prompt"]["original_caption"],
                        "edit_instruction": prompt_data["prompt"]["edit_instruction"],
                        "resulting_caption": prompt_data["prompt"]["resulting_caption"],
                        "class_name": class_name,
                    })
            if class_samples:
                samples[class_name] = class_samples

        return samples

    def get_classes(self):
        """
        Get all the class folders (keys) in the dataset.
        """
        return list(self.samples.keys())

    def get_samples_for_class(self, class_name):
        """
        Get all samples for a specific class.

        Args:
            class_name (str): Name of the class folder.

        Returns:
            List[Dict]: List of samples for the given class.
        """
        return self.samples.get(class_name, [])


def prepare_split_datasets_by_folder(dataset_dir, output_dir, train_ratio=0.9, val_ratio=0.05):
    """
    Prepares train, validation, and test splits based on class folders.

    Args:
        dataset_dir (str): Path to the dataset directory containing images and prompt.json.
        output_dir (str): Path to the output directory for split datasets.
        train_ratio (float): Ratio of folders to be used for training.
        val_ratio (float): Ratio of folders to be used for validation.

    Returns:
        None
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Create the dataset
    dataset = ImageEditDataset(dataset_dir)

    # Get all class folders
    all_classes = dataset.get_classes()

    # Shuffle and split the class folders
    num_classes = len(all_classes)
    train_size = int(train_ratio * num_classes)
    val_size = int(val_ratio * num_classes)

    shuffled_classes = all_classes.copy()
    import random
    random.shuffle(shuffled_classes)

    train_classes = shuffled_classes[:train_size]
    val_classes = shuffled_classes[train_size:train_size + val_size]
    test_classes = shuffled_classes[train_size + val_size:]

    splits = {
        "train": train_classes,
        "val": val_classes,
        "test": test_classes,
    }

    # Save the split datasets into folders and create metadata JSON
    save_split_datasets_to_folders(dataset, splits, output_dir)

def copy_sample(sample, class_dir):
    """
    Copies input and output images to the class directory.

    Args:
        sample (dict): Dictionary containing sample metadata.
        class_dir (Path): Target directory for the class.

    Returns:
        dict: Updated metadata with relative paths.
    """
    input_image_path = class_dir / sample["input_image"].name
    output_image_path = class_dir / sample["output_image"].name

    # Copy images
    shutil.copy(sample["input_image"], input_image_path)
    shutil.copy(sample["output_image"], output_image_path)

    # Return metadata with relative paths
    return {
        "input_image": f"{class_dir.name}/{sample['input_image'].name}",
        "output_image": f"{class_dir.name}/{sample['output_image'].name}",
        "original_caption": sample["original_caption"],
        "edit_instruction": sample["edit_instruction"],
        "resulting_caption": sample["resulting_caption"],
        "class_name": class_dir.name,
    }

def save_split_datasets_to_folders(dataset, splits, output_dir):
    """
    Saves the train, validation, and test datasets into separate folders and creates JSON metadata.

    Args:
        dataset (ImageEditDataset): The full dataset.
        splits (Dict[str, List[str]]): Dictionary containing the class splits for train, val, and test.
        output_dir (str): Directory where the splits will be saved.

    Returns:
        None
    """
    output_dir = Path(output_dir)

    for split_name, class_names in tqdm(splits.items(), desc="Processing Splits"):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        futures = []

        # Use ThreadPoolExecutor for parallel file copying
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for class_name in tqdm(class_names, desc=f"Processing {split_name} Classes", leave=False):
                class_samples = dataset.get_samples_for_class(class_name)
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                for sample in class_samples:
                    # Submit the file copying task
                    futures.append(executor.submit(copy_sample, sample, class_dir))

            # Collect results as they complete
            for future in tqdm(as_completed(futures), desc="Saving Metadata", total=len(futures), leave=False):
                metadata.append(future.result())

        # Save metadata to JSON
        with open(split_dir / f"{split_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory for split datasets.")
    args = parser.parse_args()

    prepare_split_datasets_by_folder(args.dataset_dir, args.output_dir)