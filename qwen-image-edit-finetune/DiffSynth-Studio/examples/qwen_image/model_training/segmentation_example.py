#!/usr/bin/env python3
"""
Example usage of segmentation-aware Qwen-Image-Edit training.

This script demonstrates how to:
1. Set up a segmentation-aware training pipeline
2. Configure dataset with segmentation masks
3. Train a model with combined image + segmentation loss
4. Evaluate the results

Usage:
    python segmentation_example.py
"""

import os
import sys
import torch
from pathlib import Path
import argparse
import json
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from diffsynth.trainers.unified_dataset_segmentation import UnifiedDatasetSegmentation
from diffsynth.pipelines.qwen_image import QwenImagePipeline
from diffsynth.trainers.utils import ModelLogger, launch_training_task
from diffsynth.models.qwen_image_dit import QwenImageDiT


def create_sample_segmentation_data(dataset_dir, num_samples=10):
    """Create sample dataset with segmentation masks for demonstration."""
    print(f"Creating sample segmentation data in {dataset_dir}...")
    
    # Create directories
    images_dir = dataset_dir / "images"
    edit_images_dir = dataset_dir / "edit_images" 
    segmentation_dir = dataset_dir / "segmentation" / "segmentation_output_parallel" / "predictions"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    edit_images_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata CSV
    metadata_path = dataset_dir / "metadata_edit.csv"
    
    metadata_lines = ["image,edit_image,prompt"]
    segmentation_data = {}
    
    for i in range(num_samples):
        # Create sample image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Create edit image with modification
        edit_array = img_array.copy()
        edit_array[40:80, 40:80] = [255, 0, 0]  # Add red square
        edit_img = Image.fromarray(edit_array)
        
        # Save images
        img_path = images_dir / f"image_{i:04d}.jpg"
        edit_path = edit_images_dir / f"image_{i:04d}_edit.jpg"
        
        img.save(img_path)
        edit_img.save(edit_path)
        
        # Create segmentation mask (same region as red square)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:80, 40:80] = 1
        
        # Store in metadata
        prompt = f"Add a red square in the center of image {i}"
        metadata_lines.append(f"{img_path.relative_to(dataset_dir)},{edit_path.relative_to(dataset_dir)},{prompt}")
        
        # Create segmentation JSON (RLE format simulation)
        segmentation_data[img_path.name] = {
            "mask": mask.tolist(),
            "bbox": [40, 40, 40, 40],
            "area": 1600,
            "category": "object"
        }
    
    # Write metadata
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata_lines))
    
    # Write segmentation data
    for img_name, seg_data in segmentation_data.items():
        seg_json_path = segmentation_dir / f"{img_name.split('.')[0]}_predictions.json"
        with open(seg_json_path, 'w') as f:
            json.dump(seg_data, f)
    
    print(f"Created {num_samples} samples with segmentation masks")
    return dataset_dir, metadata_path


def setup_segmentation_training(
    dataset_dir,
    output_dir,
    mask_loss_weight=1.0,
    num_epochs=2,
    learning_rate=1e-4
):
    """Set up and run segmentation-aware training."""
    
    print("Setting up segmentation-aware training...")
    
    # Create training pipeline
    pipeline = QwenImagePipeline()
    
    # Initialize models (you would load from checkpoints in practice)
    print("Initializing models...")
    # Note: In real usage, you would load pre-trained weights here
    # For demonstration, we use random initialization
    model_paths = {
        "dit": None,  # Path to DiT model
        "text_encoder": None,  # Path to text encoder  
        "vae": None,  # Path to VAE
    }
    
    # Set up segmentation configuration
    pipeline.mask_loss_weight = mask_loss_weight
    
    # Configure dataset
    segmentation_root = dataset_dir / "segmentation"
    
    dataset = UnifiedDatasetSegmentation(
        base_path=str(dataset_dir),
        metadata_path=str(dataset_dir / "metadata_edit.csv"),
        repeat=1,
        data_file_keys=["image", "edit_image"],
        mask_root_dir=str(segmentation_root),
        mask_target_size=None,  # Keep native resolution
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Configure training
    training_config = {
        "mask_loss_weight": mask_loss_weight,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": 2,  # Small batch for demo
        "gradient_accumulation_steps": 4,
        "warmup_steps": 100,
        "save_steps": 500,
        "logging_steps": 50,
        "evaluation_steps": 250,
        "output_dir": str(output_dir),
        "model_paths": model_paths,
    }
    
    return dataset, training_config


def demonstrate_training_loop(dataset, config):
    """Demonstrate a simplified training loop with segmentation."""
    
    print("Demonstrating training loop...")
    
    # Get a sample batch
    sample_batch = dataset[0]
    
    print("Sample batch contents:")
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Demonstrate VAE encoding of segmentation mask
    if "segmentation_mask" in sample_batch and sample_batch["segmentation_mask"] is not None:
        from diffsynth.models.qwen_image_vae import QwenImageVAE
        
        vae = QwenImageVAE()
        vae.eval()
        
        with torch.no_grad():
            mask = sample_batch["segmentation_mask"]
            print(f"Original mask shape: {mask.shape}")
            
            # Encode mask
            mask_latent = vae.encode_segmentation_mask(mask)
            print(f"Mask latent shape: {mask_latent.shape}")
            
            # Decode mask
            decoded_mask = vae.decode_segmentation_latent(mask_latent)
            print(f"Decoded mask shape: {decoded_mask.shape}")
            
            # Calculate reconstruction error
            mse = torch.nn.functional.mse_loss(decoded_mask, mask.float())
            print(f"Mask reconstruction MSE: {mse.item():.6f}")
    
    return True


def evaluate_segmentation_integration():
    """Run the validation tests for segmentation integration."""
    
    print("Running segmentation integration tests...")
    
    test_script = project_root / "examples" / "qwen_image" / "model_training" / "test_segmentation_integration.py"
    
    if test_script.exists():
        # Import and run tests
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_segmentation", test_script)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Run tests
        success = test_module.run_integration_tests()
        return success
    else:
        print(f"Test script not found: {test_script}")
        return False


def main():
    """Main function demonstrating segmentation-aware training."""
    
    parser = argparse.ArgumentParser(description="Segmentation-aware Qwen-Image-Edit Training Example")
    parser.add_argument("--dataset_dir", type=str, default="./data/demo_segmentation_dataset", 
                       help="Directory for demo dataset")
    parser.add_argument("--output_dir", type=str, default="./models/demo_segmentation_lora",
                       help="Directory for trained model output")
    parser.add_argument("--mask_loss_weight", type=float, default=1.0,
                       help="Weight for segmentation loss")
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--create_demo_data", action="store_true",
                       help="Create demo dataset with segmentation masks")
    parser.add_argument("--run_tests", action="store_true",
                       help="Run integration tests")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SEGMENTATION-AWARE QWEN-IMAGE-EDIT TRAINING EXAMPLE")
    print("=" * 60)
    
    # Set up directories
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    if args.create_demo_data:
        print("\n1. Creating Demo Dataset...")
        create_sample_segmentation_data(dataset_dir, num_samples=10)
    
    print("\n2. Setting Up Training Configuration...")
    dataset, training_config = setup_segmentation_training(
        dataset_dir,
        output_dir,
        mask_loss_weight=args.mask_loss_weight,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    print("\n3. Demonstrating Training Loop...")
    demonstrate_training_loop(dataset, training_config)
    
    if args.run_tests:
        print("\n4. Running Integration Tests...")
        success = evaluate_segmentation_integration()
        if success:
            print("✅ All integration tests passed!")
        else:
            print("❌ Some tests failed. Check implementation.")
    
    print("\n5. Training Configuration Summary")
    print("=" * 40)
    for key, value in training_config.items():
        print(f"{key}: {value}")
    
    print("\n6. Next Steps")
    print("=" * 20)
    print("To run actual training:")
    print(f"  cd {project_root}")
    print(f"  ./examples/qwen_image/model_training/lora/Qwen-Image-Edit-Segmentation.sh {args.mask_loss_weight}")
    print("\nTo adjust segmentation focus:")
    print(f"  - Higher weight ({args.mask_loss_weight * 2}): More spatial accuracy")
    print(f"  - Lower weight ({args.mask_loss_weight / 2}): More image quality focus")
    
    print("\n🎉 Segmentation integration example completed!")
    print(f"📁 Demo dataset: {dataset_dir}")
    print(f"📁 Model output: {output_dir}")


if __name__ == "__main__":
    main()