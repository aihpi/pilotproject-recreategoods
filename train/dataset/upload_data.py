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
import math

def load_metadata(jsonl_file):
    """
    Load metadata from a JSONL file.

    Args:
        jsonl_file (str): Path to the metadata JSONL file.

    Returns:
        List[Dict]: A list of metadata entries.
    """
    metadata = []
    with open(jsonl_file, "r") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

def image_to_bytes(image_path):
    with open(image_path, "rb") as img_file:
        return img_file.read()


def process_entry(entry, dataset_dir):
    """
    Process a single metadata entry by embedding input and output images as Base64 strings.

    Args:
        entry (dict): Metadata entry containing image paths and additional data.
        dataset_dir (Path): Path to the dataset directory containing images.

    Returns:
        dict: Processed entry with Base64-encoded images.
    """
    input_image_path = dataset_dir / entry["input_image"]
    output_image_path = dataset_dir / entry["output_image"]

    if input_image_path.exists() and output_image_path.exists():
        filtered_entry = {key: value for key, value in entry.items() if key not in {"input_image", "output_image"}}

        return {
            "input_image": image_to_bytes(input_image_path),
            "output_image": image_to_bytes(output_image_path),
            **filtered_entry,  # Include all metadata fields
        }
    else:
        print(f"Warning: Missing image(s) for entry {entry['input_image']} or {entry['output_image']}.")
        return None

def chunk_list(data, chunk_size):
    """Split a list into smaller chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]
        
def process_record(record):
    """Processes a single record (modify this function as needed)."""
    return {key: value for key, value in record.items()}

def create_huggingface_dataset(records, max_workers):
    """
    Create a Hugging Face Dataset from a list of records using `Dataset.from_dict`.

    Args:
        records (list): List of dictionaries containing dataset records.

    Returns:
        Dataset: Hugging Face Dataset object.
    """
   # Converts records to a dataset while processing in parallel and removing records to save memory.

    if not records:
        return Dataset.from_dict({})

    total_records = len(records)  # Get total number of records for progress bar

    # Use a generator to stream processed records
    def record_generator():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = set()  # Use a set to track active futures
            
            with tqdm(total=total_records, desc="Processing Records", unit="record") as pbar:
                while records or futures:
                    # Submit new tasks if slots are available
                    while len(futures) < max_workers and records:
                        futures.add(executor.submit(process_record, records.pop(0)))

                    # Process completed tasks
                    for future in as_completed(futures):
                        yield future.result()  # Yield processed record immediately
                        futures.remove(future)  # Free memory
                        pbar.update(1)  # Update progress bar
    # Convert generator to dataset without holding everything in memory
    dataset = Dataset.from_generator(record_generator)

    records.clear()  # Ensure all records are removed
    return dataset

def create_dataset_object(dataset_dir, metadata_files, max_workers=4, batch_size = 50000):
    """
    Create a Hugging Face DatasetDict object from metadata and images.

    Args:
        dataset_dir (str): Path to the dataset directory containing images.
        metadata_files (Dict[str, str]): Paths to metadata JSONL files for each split (train, val, test).
        max_workers (int): Number of threads to use for image processing.

    Returns:
        DatasetDict: Hugging Face DatasetDict object containing train, val, and test splits.
    """
    dataset_dir = Path(dataset_dir)

    splits = {}
    for split, metadata_file in metadata_files.items():
        print(f"Processing {split} split...")

        # Load metadata for the split
        metadata = load_metadata(metadata_file)

        # Prepare records for the split
        records = []
        dataset_dir_split = dataset_dir / split
        num_batches = ceil(len(metadata) / batch_size)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_idx, batch in enumerate(tqdm(chunk_list(metadata, batch_size), total=num_batches, desc="Submitting Batches")):
                futures = {executor.submit(process_entry, entry, dataset_dir_split): entry for entry in batch}

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split} entries"):
                    result = future.result()
                    if result is not None:
                        records.append(result)

        # Convert records to a Dataset object
        print("Convert Records to Dataset Object")
        splits[split] = create_huggingface_dataset(records, max_workers)

    # Combine splits into a DatasetDict
    dataset = DatasetDict(splits)
    return dataset


def get_rows_per_shard(avg_image_size_kb=673, target_size_gb=1):
    """
    Calculate the approximate number of rows that would produce a target file size in GB.
    avg_image_size_kb: Average size of an image in KB.
    target_size_gb: Desired file size in GB.
    """
    avg_bytes = avg_image_size_kb * 1024  # convert KB to bytes
    return int(target_size_gb * 1_000_000_000 // avg_bytes)

def save_dataset_shards(dataset: DatasetDict, save_dir: str, avg_image_size_kb=673, max_workers=4):
    """
    Save each split of the dataset as multiple Parquet shards in parallel.
    Each shard will contain approximately the number of rows that should produce ~1GB file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    def save_shard(split, shard_idx, num_shards):
        shard = dataset[split].shard(num_shards=num_shards, index=shard_idx)
        shard_path = os.path.join(save_dir, f"{split}_shard_{shard_idx+1}_of_{num_shards}.parquet")
        shard.to_parquet(shard_path)
        return shard_path

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for split in dataset.keys():
            total_rows = len(dataset[split])
            rows_per_shard = get_rows_per_shard(avg_image_size_kb)
            num_shards = math.ceil(total_rows / rows_per_shard)
            for shard_idx in range(num_shards):
                futures.append(executor.submit(save_shard, split, shard_idx, num_shards))
        
        # Use tqdm to track progress for all shard saves
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving shards", unit="shard"):
            shard_path = future.result()
            print(f"✅ Saved {shard_path}")
    
if __name__ == "__main__":
    import argparse
    from huggingface_hub import HfApi
    import os

    parser = argparse.ArgumentParser(description="Create a Hugging Face Dataset object from metadata and images.")
    parser.add_argument("--dataset_dir", required=True, help="Path to the dataset directory containing images.")
    parser.add_argument("--metadata_train", required=True, help="Path to train metadata JSONL file.")
    # parser.add_argument("--metadata_val", required=True, help="Path to val metadata JSONL file.")
    # parser.add_argument("--metadata_test", required=True, help="Path to test metadata JSONL file.")
    parser.add_argument("--hub_repo", required=True, help="Name of the Hugging Face Hub repository.")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="Number of threads for multithreading.")
    args = parser.parse_args()
    if args.max_workers < os.cpu_count():
        print(f"Max Workers lower than total CPU Count: {os.cpu_count()}")
    metadata_files = {
        "train": args.metadata_train,
        # "val": args.metadata_val,
        # "test": args.metadata_test,
    }

    # Create DatasetDict object
    dataset = create_dataset_object(dataset_dir=args.dataset_dir, metadata_files=metadata_files, max_workers=args.max_workers)
    print(dataset)
    save_dir = "./saved_shards"
    save_dataset_shards(dataset, save_dir, max_workers=args.max_workers)
    # # Push the dataset to the Hugging Face Hub
    # print(f"Pushing dataset to Hugging Face Hub: {args.hub_repo}")
    # dataset.push_to_hub(args.hub_repo, max_shard_size="1GB")
    # print("Dataset successfully pushed to the Hub!")

