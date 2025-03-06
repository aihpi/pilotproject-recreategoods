import argparse
import os
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import numpy as np

def stream_dataset_samples(dataset_name, output_dir, num_samples=100, split="val"):
    """
    Stream samples from a Hugging Face dataset without downloading the entire dataset.
    This uses the streaming mode of the datasets library.
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
        output_dir (str): Directory to save the samples
        num_samples (int): Number of samples to download
        split (str): Split to download from (e.g., "train", "val", "test")
    """
    print(f"Streaming {num_samples} samples from {dataset_name} ({split} split)")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load the dataset in streaming mode
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        print(f"Successfully loaded dataset in streaming mode")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Process samples
    seeds = []
    count = 0
    
    try:
        # Create an iterator with a progress bar
        dataset_iter = dataset.take(num_samples)
        
        for i, sample in enumerate(tqdm(dataset_iter, total=num_samples, desc="Processing samples")):
            # Create a class folder for each group of 10 samples
            class_name = f"{split}_class_{i//10}"
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            try:
                # Handle different types of image data
                # The images might be PIL Images, numpy arrays, or binary data
                
                # Input image
                if isinstance(sample["input_image"], Image.Image):
                    input_image = sample["input_image"].convert("RGB")
                elif isinstance(sample["input_image"], np.ndarray):
                    input_image = Image.fromarray(sample["input_image"]).convert("RGB")
                else:
                    # Try to interpret as binary data
                    input_image = Image.open(BytesIO(sample["input_image"]["bytes"] if isinstance(sample["input_image"], dict) else sample["input_image"])).convert("RGB")
                
                # Output image
                if isinstance(sample["output_image"], Image.Image):
                    output_image = sample["output_image"].convert("RGB")
                elif isinstance(sample["output_image"], np.ndarray):
                    output_image = Image.fromarray(sample["output_image"]).convert("RGB")
                else:
                    # Try to interpret as binary data
                    output_image = Image.open(BytesIO(sample["output_image"]["bytes"] if isinstance(sample["output_image"], dict) else sample["output_image"])).convert("RGB")
                
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
                
                # Create prompt.json for this class
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
                
                count += 1
                print(f"Successfully processed sample {i}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                # Print the sample keys to help debug
                print(f"Sample keys: {sample.keys()}")
                if "input_image" in sample:
                    print(f"Input image type: {type(sample['input_image'])}")
                if "output_image" in sample:
                    print(f"Output image type: {type(sample['output_image'])}")
                continue
            
            # Stop if we've reached the desired number of samples
            if count >= num_samples:
                break
                
    except Exception as e:
        print(f"Error streaming dataset: {str(e)}")
    
    # Save seeds.json file
    with open(output_dir / "seeds.json", "w") as f:
        json.dump(seeds, f, indent=2)
    
    print(f"Downloaded {count} samples to {output_dir}")
    print(f"Next steps:")
    print(f"1. Run: python train/dataset/split_dataset.py --dataset_dir {output_dir} --output_dir {output_dir}_split")
    print(f"2. Update your training config to use: {output_dir}_split")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream samples from a Hugging Face dataset")
    parser.add_argument("--dataset", type=str, default="AI-ServicesBB/fashion-edit-dataset", 
                        help="Name of the Hugging Face dataset")
    parser.add_argument("--output_dir", type=str, default="fashion_dataset_stream",
                        help="Directory to save the samples")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to download")
    parser.add_argument("--split", type=str, default="val",
                        help="Split to download from (e.g., 'train', 'val', 'test')")
    
    args = parser.parse_args()
    
    stream_dataset_samples(args.dataset, args.output_dir, args.num_samples, args.split) 