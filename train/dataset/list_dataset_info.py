import argparse
from datasets import get_dataset_config_names, load_dataset_builder
import os
import sys
from huggingface_hub import HfApi

def get_dataset_info(dataset_name):
    """
    Get information about a Hugging Face dataset including available splits and their sizes
    without downloading the actual data.
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
    """
    print(f"\nGetting information for dataset: {dataset_name}")
    
    try:
        # Try to get configuration names (some datasets have multiple configurations)
        try:
            config_names = get_dataset_config_names(dataset_name)
            if config_names:
                print(f"\nAvailable configurations: {', '.join(config_names)}")
                print("If you want to use a specific configuration, use: dataset_name/config_name")
        except Exception as e:
            print("No specific configurations found.")
        
        # Use the dataset builder to get info without downloading the data
        builder = load_dataset_builder(dataset_name)
        
        # Get dataset info
        info = builder.info
        
        # Print dataset description
        if info.description:
            print(f"\nDescription: {info.description[:200]}...")
        
        # Print split information
        print("\nAvailable splits:")
        total_samples = 0
        
        for split_name, split_info in info.splits.items():
            num_examples = split_info.num_examples
            total_samples += num_examples
            
            # Estimate size (very rough approximation)
            # This is just a guess based on typical dataset sizes
            estimated_size_mb = num_examples * 0.1  # Assuming ~100KB per sample on average
            
            print(f"  - {split_name}: {num_examples:,} samples (estimated size: ~{estimated_size_mb:.1f} MB)")
        
        # Print feature information
        if info.features:
            print("\nFeatures:")
            for feature_name, feature_info in info.features.items():
                print(f"  - {feature_name}: {feature_info}")
        
        print(f"\nTotal samples across all splits: {total_samples:,}")
        print(f"Estimated total size: ~{total_samples * 0.1 / 1024:.2f} GB (very rough estimate)")
        
        # Try to get repository info for more accurate size information
        try:
            api = HfApi()
            repo_info = api.repo_info(repo_id=dataset_name, repo_type="dataset")
            if hasattr(repo_info, 'size_in_bytes'):
                actual_size_gb = repo_info.size_in_bytes / (1024 * 1024 * 1024)
                print(f"Actual repository size: {actual_size_gb:.2f} GB")
        except Exception as e:
            print("Could not retrieve actual repository size information.")
        
    except Exception as e:
        print(f"Error getting dataset info: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get information about a Hugging Face dataset without downloading it")
    parser.add_argument("--dataset", type=str, default="AI-ServicesBB/fashion-edit-dataset", 
                        help="Name of the Hugging Face dataset")
    
    args = parser.parse_args()
    
    get_dataset_info(args.dataset) 