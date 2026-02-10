#!/usr/bin/env python3
"""
Script to check WandB runs for validation images and metrics.
This helps verify that the validation pipeline is working correctly.
"""

import os
import json
import glob
from pathlib import Path

def check_wandb_runs():
    """Check all WandB offline runs for validation content."""
    wandb_dir = Path("wandb")
    
    if not wandb_dir.exists():
        print("No WandB directory found")
        return
        
    runs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("offline-run-")]
    
    if not runs:
        print("No offline runs found")
        return
        
    print(f"Found {len(runs)} offline WandB runs:")
    
    for run_dir in runs:
        print(f"\n--- Run: {run_dir.name} ---")
        
        # Check run metadata
        run_files = glob.glob(str(run_dir / "run-*.wandb"))
        if run_files:
            print(f"  Run file: {os.path.basename(run_files[0])}")
            
        # Check logs for validation info
        log_files = glob.glob(str(run_dir / "logs" / "*.log"))
        for log_file in log_files:
            with open(log_file, 'r') as f:
                content = f.read()
                if "validation" in content.lower():
                    print(f"  Found validation info in {os.path.basename(log_file)}")
                    
        # Check for image files
        media_files = glob.glob(str(run_dir / "media") + "/**/*", recursive=True)
        image_files = [f for f in media_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            print(f"  Found {len(image_files)} image files")
            for img in image_files[:5]:  # Show first 5
                print(f"    - {os.path.basename(img)}")
        else:
            print("  No image files found")
            
        # Check for validation-related files
        all_files = glob.glob(str(run_dir / "**/*"), recursive=True)
        validation_files = [f for f in all_files if "validation" in f.lower()]
        
        if validation_files:
            print(f"  Found {len(validation_files)} validation-related files")

if __name__ == "__main__":
    check_wandb_runs()