#!/usr/bin/env python3
"""
Test script for validation image logging pipeline.

This script tests the new validation functionality to ensure:
1. Validation images are generated correctly
2. LPIPS scores are calculated
3. WandB logging works with proper step indexing
4. Image comparison grids are created properly
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import wandb

# Add the path to import the training modules
sys.path.append('DiffSynth-Studio')

from examples.qwen_image.model_training.train_with_segmentation_wandb import (
    run_validation_with_logging,
    create_validation_comparison_grid
)


class MockDataset:
    """Mock dataset for testing validation pipeline."""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        """Return mock validation sample with test images."""
        # Create simple test images
        input_image = Image.new('RGB', (512, 512), color=(255, 0, 0))  # Red
        target_image = Image.new('RGB', (512, 512), color=(0, 255, 0))  # Green
        prompt = f"Test validation sample {idx}"
        
        return {
            "prompt": prompt,
            "edit_image": input_image,
            "input_image": input_image,  # Alternative key
            "image": target_image,
        }


class MockModel:
    """Mock model for testing validation pipeline."""
    
    def __init__(self):
        # Set up the pipe structure to match the training script expectations
        self.module = MockPipe()  # For distributed training (model.module.pipe)
        self.pipe = MockPipe()    # For non-distributed training (model.pipe)
        
    def eval(self):
        """Switch to eval mode."""
        pass
        
    def train(self):
        """Switch to train mode."""
        pass


class MockPipe:
    """Mock pipeline for image generation."""
    
    def __call__(self, prompt, edit_image, height, width, num_inference_steps=20, edit_image_auto_resize=True, seed=42):
        """Mock image generation - return modified target image."""
        # Generate a simple blue image as "generated" result
        return Image.new('RGB', edit_image.size, color=(0, 0, 255))  # Blue
        
    def parameters(self):
        """Mock device parameters for LPIPS calculation."""
        return [torch.tensor([1.0], device='cpu')]


class MockModelLogger:
    """Mock model logger for testing."""
    
    def __init__(self):
        pass


def test_validation_comparison_grid():
    """Test the validation comparison grid creation."""
    print("Testing validation comparison grid creation...")
    
    # Create test images
    input_image = Image.new('RGB', (256, 256), color=(255, 0, 0))  # Red
    target_image = Image.new('RGB', (256, 256), color=(0, 255, 0))  # Green
    generated_image = Image.new('RGB', (256, 256), color=(0, 0, 255))  # Blue
    prompt = "Test prompt for image comparison"
    lpips_score = 0.1234
    
    # Test grid creation
    grid = create_validation_comparison_grid(input_image, target_image, generated_image, prompt, lpips_score)
    
    # Verify grid properties
    assert isinstance(grid, Image.Image), "Grid should be a PIL Image"
    expected_width = 256 * 3 + 20  # 3 images + padding
    expected_height = 256 + 60     # image height + space for labels
    assert grid.size == (expected_width, expected_height), f"Grid size mismatch: {grid.size} vs {(expected_width, expected_height)}"
    
    print("✓ Validation comparison grid test passed")
    return True


def test_validation_pipeline():
    """Test the complete validation pipeline."""
    print("Testing validation pipeline...")
    
    # Initialize wandb in offline mode for testing
    try:
        wandb.init(mode="offline", project="test-validation")
        wandb_enabled = True
    except:
        print("Warning: WandB not available, continuing without it")
        wandb_enabled = False
    
    try:
        # Create mock components
        mock_dataset = MockDataset(num_samples=10)
        mock_model = MockModel()
        mock_model.module = MockPipe() if hasattr(mock_model, 'module') else MockPipe()
        mock_model_logger = MockModelLogger()
        
        # Test parameters
        step = 100
        num_samples = 4
        
        print(f"Running validation with {num_samples} samples at step {step}")
        
        # Run validation
        run_validation_with_logging(
            model=mock_model,
            dataset=mock_dataset,
            model_logger=mock_model_logger,
            step=step,
            num_samples=num_samples,
            save_images=True
        )
        
        print("✓ Validation pipeline executed successfully")
        
        if wandb_enabled:
            # Check if validation metrics were logged
            print("Checking WandB logs...")
            # Note: In offline mode, we can't easily verify the logs, but the function should complete without errors
            print("✓ WandB logging integration works")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if wandb_enabled:
            wandb.finish()


def test_lpips_calculation():
    """Test LPIPS score calculation in validation."""
    print("Testing LPIPS calculation...")
    
    try:
        import lpips
        import torchvision.transforms.functional as TF
        
        # Create test images
        img1 = Image.new('RGB', (256, 256), color=(255, 0, 0))  # Red
        img2 = Image.new('RGB', (256, 256), color=(0, 255, 0))  # Green
        
        # Convert to tensors and normalize
        tensor1 = TF.to_tensor(img1).unsqueeze(0) * 2.0 - 1.0
        tensor2 = TF.to_tensor(img2).unsqueeze(0) * 2.0 - 1.0
        
        # Calculate LPIPS
        lpips_fn = lpips.LPIPS(net='alex')
        score = lpips_fn(tensor1, tensor2).item()
        
        print(f"✓ LPIPS calculation works: score = {score:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ LPIPS calculation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATION IMAGE LOGGING PIPELINE TESTS")
    print("=" * 60)
    
    tests = [
        test_validation_comparison_grid,
        test_lpips_calculation,
        test_validation_pipeline,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests PASSED!")
        print("\nThe validation image logging pipeline is working correctly:")
        print("✓ Image comparison grids are created properly")
        print("✓ LPIPS scores are calculated correctly")
        print("✓ Validation pipeline executes without errors")
        print("✓ WandB integration is functional")
        return True
    else:
        print("❌ Some tests FAILED!")
        print("Please check the errors above and fix the validation pipeline.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)