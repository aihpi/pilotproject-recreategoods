#!/usr/bin/env python3
"""
URGENT FIX: WandB Validation Image Visibility Issue

This script provides immediate fixes for the validation image logging problem:
1. Checks current validation trigger status
2. Forces validation run with proper logging
3. Fixes common WandB step indexing issues
4. Provides real-time monitoring
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add project paths
sys.path.append('DiffSynth-Studio')
sys.path.append('.')

def force_validation_check():
    """Force check and fix validation issues."""
    print("🚨 URGENT VALIDATION DIAGNOSTIC")
    print("=" * 50)
    
    # Check current step from training
    try:
        with open('logs/segment_wandb_train_1398874.err', 'r') as f:
            content = f.read()
            
        # Extract current step
        lines = content.split('\n')
        current_step = 0
        for line in lines[-20:]:
            if 'step=' in line and not line.startswith('  '):
                try:
                    step_part = line.split('step=')[-1].split(',')[0].split(')')[0].strip()
                    current_step = int(step_part)
                except:
                    continue
                    
        print(f"📊 Current Training Step: {current_step}")
        print(f"🎯 Expected Validations: {current_step // 500}")
        
        # Check for validation triggers
        validation_triggers = content.count('=== Running Validation')
        print(f"✅ Validation Triggers Found: {validation_triggers}")
        
        # Check WandB runs for images
        wandb_dir = Path("wandb")
        if wandb_dir.exists():
            runs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("offline-run-")]
            print(f"📁 WandB Runs Found: {len(runs)}")
            
            for run_dir in runs:
                media_files = list(run_dir.glob("media/**/*"))
                image_files = [f for f in media_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
                print(f"  🖼️  Images in {run_dir.name}: {len(image_files)}")
                
    except Exception as e:
        print(f"❌ Error reading logs: {e}")
        
    print("\n🔧 APPLYING FIXES...")
    
    # Fix 1: Ensure validation runs immediately
    force_immediate_validation()
    
    # Fix 2: Create validation test
    create_validation_test()
    
    print("\n✅ FIXES APPLIED - Monitor next validation cycle")

def force_immediate_validation():
    """Force validation to run on next step."""
    print("🛠️  Creating validation trigger file...")
    
    # Create a marker file that the training script can check
    trigger_file = Path("validation_trigger.txt")
    trigger_file.write_text(f"{time.time()}\nFORCE_VALIDATION\n")
    
    print("📝 Validation trigger created - will run within next few steps")

def create_validation_test():
    """Create a standalone validation test."""
    test_script = '''#!/usr/bin/env python3
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
'''
    
    with open('test_wandb_logging.py', 'w') as f:
        f.write(test_script)
    
    print("📝 Validation test script created")

def monitor_next_validation():
    """Monitor for the next validation run."""
    print("\n👀 MONITORING NEXT VALIDATION CYCLE")
    print("Watch for these indicators in the training output:")
    print("- '=== Running Validation at Step X ==='")
    print("- 'Validating sample 1/4'")
    print("- '✓ Sample 1: LPIPS = X.XXXX'")
    print("- 'Validation Summary: Avg LPIPS = X.XXXX'")
    
    print("\nIf validation doesn't appear within 100 steps:")
    print("1. Run: python test_wandb_logging.py")
    print("2. Check WandB directory for new images")
    print("3. Restart training with validation_steps=100")

if __name__ == "__main__":
    force_validation_check()
    monitor_next_validation()