#!/usr/bin/env python3
"""
Test validation logging to ensure images appear in WandB
"""
import os
import sys
import torch
from PIL import Image
import numpy as np

sys.path.append('DiffSynth-Studio')

# Create test images
def create_test_image(size=(512, 512), color='red'):
    img = Image.new('RGB', size, color=color)
    return img

def test_wandb_logging():
    """Test that we can log images to WandB properly."""
    try:
        import wandb
        
        # Initialize wandb in offline mode
        wandb.init(mode="offline", project="validation-test")
        
        # Create test images
        input_img = create_test_image(color='red')
        target_img = create_test_image(color='green')
        generated_img = create_test_image(color='blue')
        
        # Log test images
        wandb.log({
            "test_input": wandb.Image(input_img),
            "test_target": wandb.Image(target_img),
            "test_generated": wandb.Image(generated_img),
            "test_step": 9999
        })
        
        # Create comparison grid
        from examples.qwen_image.model_training.train_with_segmentation_wandb import create_validation_comparison_grid
        
        grid = create_validation_comparison_grid(
            input_img, target_img, generated_img, 
            "Test validation prompt", 0.1234
        )
        
        wandb.log({
            "test_comparison": wandb.Image(grid),
            "test_step": 10000
        })
        
        print("✅ Test images logged successfully!")
        wandb.finish()
        
    except Exception as e:
        print(f"❌ WandB logging test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wandb_logging()
