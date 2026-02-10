#!/usr/bin/env python3
"""
Create comprehensive test data for segmentation integration testing.

This script generates synthetic test data including:
- Synthetic images with various patterns
- Corresponding segmentation masks
- Dataset structure compatible with UnifiedDatasetSegmentation
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch


def create_test_dataset(output_dir, num_samples=20, image_size=128):
    """Create test dataset with synthetic images and segmentation masks."""
    
    output_dir = Path(output_dir)
    
    # Create directory structure
    images_dir = output_dir / "images"
    edit_images_dir = output_dir / "edit_images"
    segmentation_dir = output_dir / "segmentation" / "segmentation_output_parallel" / "predictions"
    
    for dir_path in [images_dir, edit_images_dir, segmentation_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create metadata CSV
    metadata_lines = ["image,edit_image,prompt"]
    
    # Generate different test patterns
    patterns = [
        "center_square",
        "top_left_circle", 
        "checkerboard",
        "diagonal_stripe",
        "corner_triangle"
    ]
    
    print(f"Creating {num_samples} test samples...")
    
    for i in range(num_samples):
        pattern = patterns[i % len(patterns)]
        
        # Create base image
        base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Add background texture
        base_image[:, :, 0] = np.random.randint(50, 150, (image_size, image_size))  # Red channel
        base_image[:, :, 1] = np.random.randint(50, 150, (image_size, image_size))  # Green channel  
        base_image[:, :, 2] = np.random.randint(50, 150, (image_size, image_size))  # Blue channel
        
        # Create edit image with pattern-specific modification
        edit_image = base_image.copy()
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        
        if pattern == "center_square":
            # Add red square in center
            start = image_size // 4
            end = 3 * image_size // 4
            edit_image[start:end, start:end] = [255, 0, 0]
            mask[start:end, start:end] = 1
            prompt = f"Add a red square in the center"
            
        elif pattern == "top_left_circle":
            # Add blue circle in top-left
            center_x, center_y = image_size // 4, image_size // 4
            radius = image_size // 6
            y, x = np.ogrid[:image_size, :image_size]
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            edit_image[circle_mask] = [0, 0, 255]
            mask[circle_mask] = 1
            prompt = f"Add a blue circle in the top-left corner"
            
        elif pattern == "checkerboard":
            # Add checkerboard pattern
            cell_size = image_size // 8
            for y in range(0, image_size, cell_size):
                for x in range(0, image_size, cell_size):
                    if (x + y) // cell_size % 2 == 0:
                        edit_image[y:y+cell_size, x:x+cell_size] = [255, 255, 0]  # Yellow
                        mask[y:y+cell_size, x:x+cell_size] = 1
            prompt = f"Add a yellow checkerboard pattern"
            
        elif pattern == "diagonal_stripe":
            # Add green diagonal stripe
            for y in range(image_size):
                x = int(y * 0.8 + image_size * 0.1)
                if 0 <= x < image_size:
                    stripe_width = 10
                    start_x = max(0, x - stripe_width)
                    end_x = min(image_size, x + stripe_width)
                    edit_image[y, start_x:end_x] = [0, 255, 0]
                    mask[y, start_x:end_x] = 1
            prompt = f"Add a green diagonal stripe"
            
        elif pattern == "corner_triangle":
            # Add purple triangle in bottom-right
            vertices = np.array([
                [3*image_size//4, 3*image_size//4],
                [3*image_size//4, image_size-1],
                [image_size-1, 3*image_size//4]
            ])
            y, x = np.ogrid[:image_size, :image_size]
            triangle_mask = (
                (vertices[1][1] - vertices[0][1]) * (x - vertices[0][0]) - 
                (vertices[1][0] - vertices[0][0]) * (y - vertices[0][1]) >= 0
            ) & (
                (vertices[2][1] - vertices[1][1]) * (x - vertices[1][0]) - 
                (vertices[2][0] - vertices[1][0]) * (y - vertices[1][1]) >= 0
            ) & (
                (vertices[0][1] - vertices[2][1]) * (x - vertices[2][0]) - 
                (vertices[0][0] - vertices[2][0]) * (y - vertices[2][1]) >= 0
            )
            edit_image[triangle_mask] = [128, 0, 128]  # Purple
            mask[triangle_mask] = 1
            prompt = f"Add a purple triangle in the bottom-right corner"
        
        # Save images
        img_path = images_dir / f"test_{i:04d}.jpg"
        edit_path = edit_images_dir / f"test_{i:04d}_edit.jpg"
        
        Image.fromarray(base_image).save(img_path)
        Image.fromarray(edit_image).save(edit_path)
        
        # Create segmentation JSON (simulate RLE format)
        segmentation_data = {
            "mask": mask.tolist(),
            "bbox": [int(mask.any(0).nonzero()[0][0]) if mask.any(0).any() else 0,
                    int(mask.any(1).nonzero()[0][0]) if mask.any(1).any() else 0,
                    int(mask.any(0).nonzero()[-1][0]) - int(mask.any(0).nonzero()[0][0]) + 1,
                    int(mask.any(1).nonzero()[-1][0]) - int(mask.any(1).nonzero()[0][0]) + 1],
            "area": int(mask.sum()),
            "category": "object"
        }
        
        seg_json_path = segmentation_dir / f"test_{i:04d}_predictions.json"
        with open(seg_json_path, 'w') as f:
            json.dump(segmentation_data, f)
        
        # Add to metadata
        metadata_lines.append(f"{img_path.name},{edit_path.name},{prompt}")
        
        print(f"Created sample {i+1}/{num_samples}: {pattern}")
    
    # Write metadata file
    metadata_path = output_dir / "metadata_edit.csv"
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata_lines))
    
    print(f"\nTest dataset created successfully!")
    print(f"Dataset directory: {output_dir}")
    print(f"Total samples: {num_samples}")
    print(f"Metadata file: {metadata_path}")
    
    return metadata_path


def test_data_validation(dataset_dir):
    """Validate the created test data."""
    dataset_dir = Path(dataset_dir)
    
    print("\nValidating test data...")
    
    # Check directory structure
    required_dirs = ["images", "edit_images", "segmentation/segmentation_output_parallel/predictions"]
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if not dir_path.exists():
            print(f"❌ Missing directory: {dir_name}")
            return False
        print(f"✅ Directory exists: {dir_name}")
    
    # Check files
    images_dir = dataset_dir / "images"
    edit_images_dir = dataset_dir / "edit_images"
    segmentation_dir = dataset_dir / "segmentation" / "segmentation_output_parallel" / "predictions"
    
    num_images = len(list(images_dir.glob("*.jpg")))
    num_edit_images = len(list(edit_images_dir.glob("*.jpg")))
    num_masks = len(list(segmentation_dir.glob("*.json")))
    
    if num_images == num_edit_images == num_masks:
        print(f"✅ File counts match: {num_images} samples")
    else:
        print(f"❌ File count mismatch: images={num_images}, edit={num_edit_images}, masks={num_masks}")
        return False
    
    # Validate a sample JSON
    sample_json = list(segmentation_dir.glob("*.json"))[0]
    with open(sample_json) as f:
        mask_data = json.load(f)
    
    required_keys = ["mask", "bbox", "area", "category"]
    for key in required_keys:
        if key not in mask_data:
            print(f"❌ Missing key in JSON: {key}")
            return False
    
    print(f"✅ Sample JSON validation passed")
    
    # Test mask dimensions
    mask_array = np.array(mask_data["mask"])
    if len(mask_array.shape) == 2:
        print(f"✅ Mask shape: {mask_array.shape}")
    else:
        print(f"❌ Invalid mask shape: {mask_array.shape}")
        return False
    
    print("✅ All validation checks passed!")
    return True


def main():
    """Main function to create and validate test data."""
    print("=" * 60)
    print("SEGMENTATION TEST DATA GENERATION")
    print("=" * 60)
    
    # Create test dataset
    dataset_dir = "./testing/test_dataset"
    metadata_path = create_test_dataset(dataset_dir, num_samples=20, image_size=128)
    
    # Validate the created data
    if test_data_validation(dataset_dir):
        print(f"\n🎉 Test dataset ready for segmentation integration testing!")
        print(f"Dataset path: {dataset_dir}")
        print(f"Metadata path: {metadata_path}")
    else:
        print(f"\n❌ Test data validation failed!")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)