#!/usr/bin/env python3
"""
FORCE VALIDATION: Immediate validation trigger for WandB

This script forces validation to run immediately regardless of step count.
It will:
1. Override the validation step check
2. Force-run validation with current model state
3. Log images to WandB with proper metadata
4. Ensure images are visible in WandB interface
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.append('DiffSynth-Studio')
sys.path.append('.')

def force_validation_run():
    """Force validation to run immediately."""
    print("🔥 FORCING VALIDATION RUN")
    print("=" * 40)
    
    try:
        # Import required modules
        from examples.qwen_image.model_training.train_with_segmentation_wandb import run_validation_with_logging
        import torch
        from PIL import Image
        import wandb
        import random
        
        # Initialize wandb
        wandb.init(
            mode="offline", 
            project="validation-debug",
            name="forced-validation"
        )
        
        # Create mock dataset with real validation data
        class ForceValidationDataset:
            def __init__(self):
                # Find actual validation data
                self.data = []
                data_path = Path("data/example_image_dataset")
                if data_path.exists():
                    csv_file = data_path / "metadata_edit.csv"
                    if csv_file.exists():
                        import pandas as pd
                        df = pd.read_csv(csv_file)
                        for _, row in df.head(4).iterrows():  # Take first 4 samples
                            self.data.append(row)
                        print(f"📁 Found {len(self.data)} validation samples")
                else:
                    # Create mock data if real data not found
                    for i in range(4):
                        self.data.append({
                            'image': f'mock_image_{i}.jpg',
                            'edit_image': f'mock_edit_{i}.jpg',
                            'prompt': f'Mock validation prompt {i}'
                        })
                    print("📝 Using mock validation data")
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create mock model
        class ForceModel:
            def __init__(self):
                self.module = ForcePipe()
                self.pipe = ForcePipe()
            
            def eval(self):
                pass
                
            def train(self):
                pass
        
        class ForcePipe:
            def __call__(self, prompt, edit_image, height, width, num_inference_steps=20, edit_image_auto_resize=True, seed=42):
                # Create a simple "generated" image
                color = ['red', 'blue', 'green', 'yellow'][hash(prompt) % 4]
                return Image.new('RGB', edit_image.size, color=color)
            
            def parameters(self):
                return [torch.tensor([1.0], device='cpu')]
        
        # Create components
        dataset = ForceValidationDataset()
        model = ForceModel()
        
        # Force validation run with step 1500 (current step)
        current_step = 1500
        print(f"🎯 Running validation at forced step {current_step}")
        
        run_validation_with_logging(
            model=model,
            dataset=dataset,
            model_logger=None,
            step=current_step,
            num_samples=4,
            save_images=True
        )
        
        print(f"✅ Forced validation completed at step {current_step}")
        
        # Check WandB for new images
        check_wandb_images()
        
        wandb.finish()
        
    except Exception as e:
        print(f"❌ Force validation failed: {e}")
        import traceback
        traceback.print_exc()

def check_wandb_images():
    """Check if images were added to WandB."""
    print("\n📊 CHECKING WandB FOR IMAGES")
    print("-" * 30)
    
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        print("❌ No WandB directory found")
        return
    
    runs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("offline-run-")]
    if not runs:
        print("❌ No WandB runs found")
        return
    
    for run_dir in runs:
        # Check media directory
        media_dir = run_dir / "media"
        if media_dir.exists():
            image_files = list(media_dir.glob("**/*.png")) + list(media_dir.glob("**/*.jpg")) + list(media_dir.glob("**/*.jpeg"))
            if image_files:
                print(f"✅ Found {len(image_files)} images in {run_dir.name}:")
                for img in image_files:
                    print(f"  🖼️  {img.name}")
            else:
                print(f"⚠️  No images in {run_dir.name}")
        else:
            print(f"⚠️  No media directory in {run_dir.name}")

def monitor_training_for_validation():
    """Monitor training output for validation messages."""
    print("\n👀 MONITORING TRAINING FOR VALIDATION")
    print("-" * 40)
    
    log_file = Path("logs/segment_wandb_train_1398874.err")
    if not log_file.exists():
        print("❌ Training log not found")
        return
    
    print("📖 Recent training output:")
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"❌ Error reading log: {e}")
    
    print("\n🔍 VALIDATION INDICATORS TO WATCH FOR:")
    print("- '=== Running Validation at Step'")
    print("- 'Validating sample 1/4'")
    print("- 'LPIPS ='")
    print("- 'Validation Summary'")
    
    print(f"\n📞 NEXT ACTIONS:")
    print("1. Check training output for validation messages")
    print("2. If no validation in 50 steps, restart with --validation_steps 100")
    print("3. Run: python test_wandb_logging.py")

if __name__ == "__main__":
    force_validation_run()
    monitor_training_for_validation()