import os
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import argparse

def download_and_prepare_dataset(dataset_name, output_dir, splits=None, max_samples=None):
    """
    Download a dataset from Hugging Face and prepare it for the training pipeline.
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
        output_dir (str): Directory to save the prepared dataset
        splits (list): List of splits to download (default: all available splits)
        max_samples (int): Maximum number of samples to download (default: all samples)
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load the dataset
    if splits:
        dataset = {split: load_dataset(dataset_name, split=split) for split in splits}
    else:
        dataset = load_dataset(dataset_name)
    
    # Create base directory
    base_dir = Path(output_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each split
    seeds = []
    total_samples = 0
    
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split with {len(split_data)} samples")
        
        # Group samples into classes (10 samples per class)
        for i, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # Stop if we've reached the maximum number of samples
            if max_samples is not None and total_samples >= max_samples:
                print(f"Reached maximum number of samples ({max_samples}). Stopping.")
                break
                
            # Create a class folder for each sample or group of samples
            class_name = f"{split_name}_class_{i//10}"  # Group every 10 samples into one class
            class_dir = base_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Get images
            input_image = Image.open(BytesIO(sample["input_image"])).convert("RGB")
            output_image = Image.open(BytesIO(sample["output_image"])).convert("RGB")
            
            # Save images
            seed = f"{i}"
            input_path = class_dir / f"{seed}_0.jpg"
            output_path = class_dir / f"{seed}_1.jpg"
            
            input_image.save(input_path)
            output_image.save(output_path)
            
            # Create or update class seeds
            if not any(item[0] == class_name for item in seeds):
                seeds.append((class_name, []))
            
            # Find the class in seeds and add the seed
            for idx, (name, class_seeds) in enumerate(seeds):
                if name == class_name:
                    seeds[idx][1].append(seed)
                    break
            
            # Create prompt.json for this class if it doesn't exist
            prompt_file = class_dir / "prompt.json"
            
            # Get captions and edit instruction
            original_caption = sample.get("original_caption", "")
            edit_instruction = sample.get("edit_instruction", "")
            resulting_caption = sample.get("resulting_caption", "")
            
            prompt_data = {
                "prompt": {
                    "original_caption": original_caption,
                    "edit_instruction": edit_instruction,
                    "resulting_caption": resulting_caption
                }
            }
            
            with open(prompt_file, "w") as f:
                json.dump(prompt_data, f, indent=2)
                
            total_samples += 1
            
        # Stop processing splits if we've reached the maximum number of samples
        if max_samples is not None and total_samples >= max_samples:
            break
    
    # Save seeds.json file
    with open(base_dir / "seeds.json", "w") as f:
        json.dump(seeds, f, indent=2)
    
    print(f"Dataset prepared at {base_dir}")
    print(f"Total samples downloaded: {total_samples}")
    print(f"Total classes: {len(seeds)}")
    print(f"Next steps:")
    print(f"1. Run: python train/dataset/split_dataset.py --dataset_dir {output_dir} --output_dir {output_dir}_split")
    print(f"2. Update your training config to use: {output_dir}_split")
    print(f"3. Run training: python train/main.py --config_path your_config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare a dataset from Hugging Face for training")
    parser.add_argument("--dataset", type=str, default="AI-ServicesBB/fashion-edit-dataset", 
                        help="Name of the Hugging Face dataset")
    parser.add_argument("--output_dir", type=str, default="fashion_edit_dataset",
                        help="Directory to save the prepared dataset")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Dataset splits to download (default: all available splits)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to download (default: all samples)")
    
    args = parser.parse_args()
    
    download_and_prepare_dataset(args.dataset, args.output_dir, args.splits, args.max_samples) 