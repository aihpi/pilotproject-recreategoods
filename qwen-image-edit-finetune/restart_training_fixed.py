#!/usr/bin/env python3
"""
Restart training with all validation fixes applied.

This script will restart the training using the fixed WandB training script
with all validation image logging issues resolved.
"""

import subprocess
import sys
import os
from pathlib import Path

def restart_training():
    """Restart training with validation fixes."""
    print("🚀 RESTARTING TRAINING WITH VALIDATION FIXES")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path("/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune")
    os.chdir(project_dir)
    
    print("📍 Working directory:", os.getcwd())
    print("🔧 Using fixed training script with validation logging")
    
    # Start training with the fixed script
    print("\n▶️  Starting training...")
    print("Command: python train_launcher_wandb.py")
    
    # Run the training launcher
    try:
        result = subprocess.run([
            sys.executable, "train_launcher_wandb.py"
        ], cwd=project_dir)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
        else:
            print(f"⚠️  Training exited with code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error running training: {e}")

if __name__ == "__main__":
    restart_training()