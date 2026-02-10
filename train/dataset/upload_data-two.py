from datasets import Dataset, DatasetDict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import base64
from PIL import Image
import io
from tqdm.auto import tqdm
from math import ceil
import pandas as pd
from huggingface_hub import HfApi
import os

def load_metadata(jsonl_file):
    metadata = []
    with open(jsonl_file, "r") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

def image_to_bytes(image_path):
    with open(image_path, "rb") as img_file:
        return img_file.read()

def process_entry(entry, dataset_dir):
    input_image_path = dataset_dir / entry["input_image"]
    output_image_path = dataset_dir / entry["output_image"]
    
    if input_image_path.exists() and output_image_path.exists():
        filtered_entry = {key: value for key, value in entry.items() if key not in {"input_image", "output_image"}}
        return {
            "input_image": image_to_bytes(input_image_path),
            "output_image": image_to_bytes(output_image_path),
            **filtered_entry,
        }
    else:
        print(f"Warning: Missing image(s) for entry {entry['input_image']} or {entry['output_image']}.")
        return None

def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def create_huggingface_dataset(records):
    if not records:
        return Dataset.from_dict({})
    data = {key: [record[key] for record in records] for key in records[0]}
    return Dataset.from_dict(data)

def upload_batches(dataset_dir, metadata_file, split, hub_repo, max_workers=4, batch_size=40000, save_dir="processed_datasets"):
    dataset_dir = Path(dataset_dir) / split
    metadata = load_metadata(metadata_file)
    num_batches = ceil(len(metadata) / batch_size)
    
    api = HfApi()
    save_path = Path(save_dir) / split
    save_path.mkdir(parents=True, exist_ok=True)  # Create save directory
    
    batch_files = []

    # Step 1: Process and Save Each Batch Locally
    for batch_idx, batch in enumerate(tqdm(chunk_list(metadata, batch_size), total=num_batches, desc=f"Processing {split}")):
        records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_entry, entry, dataset_dir): entry for entry in batch}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch {batch_idx+1}/{num_batches}"):
                result = future.result()
                if result is not None:
                    records.append(result)
        
        # Convert to Hugging Face dataset
        dataset = create_huggingface_dataset(records)
        
        # Save locally instead of pushing
        batch_filename = save_path / f"batch_{batch_idx+1}.parquet"
        dataset.to_parquet(batch_filename)  # Save as Parquet for efficiency
        batch_files.append(batch_filename)
        print(f"Batch {batch_idx+1} saved locally at {batch_filename}")

    # Step 2: Load All Batches and Push to Hub
    print(f"\n🔄 Combining all batches and pushing to Hugging Face Hub: {hub_repo}\n")

    # Load all saved datasets
    all_datasets = [Dataset.from_parquet(str(file)) for file in batch_files]
    combined_dataset = DatasetDict({split: Dataset.from_dict(all_datasets[0].to_dict())})  # Start with first dataset

    # Merge all datasets into one (if multiple batches)
    for dataset in all_datasets[1:]:
        combined_dataset[split] = combined_dataset[split].concatenate(dataset)

    # Push final dataset to Hugging Face Hub
    combined_dataset.push_to_hub(hub_repo, max_shard_size="1GB")
    print(f"\n✅ Successfully pushed full dataset to Hugging Face Hub: {hub_repo}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload dataset in batches to Hugging Face Hub.")
    parser.add_argument("--dataset_dir", required=True, help="Path to the dataset directory containing images.")
    parser.add_argument("--metadata_train", required=True, help="Path to train metadata JSONL file.")
    parser.add_argument("--hub_repo", required=True, help="Name of the Hugging Face Hub repository.")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="Number of threads for multithreading.")
    args = parser.parse_args()
    
    upload_batches(dataset_dir=args.dataset_dir, metadata_file=args.metadata_train, split="train", hub_repo=args.hub_repo, max_workers=args.max_workers)
