#!/usr/bin/env python3
import sys
sys.path.append('DiffSynth-Studio')
sys.path.append('.')

from diffsynth.pipelines.qwen_image import QwenImagePipeline
from PIL import Image
import wandb
import tempfile
import os

# Create a simple validation pipe
class SimpleValidationPipe:
    def __call__(self, prompt, edit_image, height, width, num_inference_steps=20, edit_image_auto_resize=True, seed=42):
        # Generate a simple "edited" version of the input
        # For validation, we'll create a simple colored version
        color_variants = ['blue', 'green', 'yellow', 'purple', 'orange']
        color = color_variants[hash(prompt) % len(color_variants)]
        return Image.new('RGB', edit_image.size, color=color)
    
    def parameters(self):
        return [__import__('torch').tensor([1.0], device='cpu')]

def run_validation():
    # Initialize wandb
    wandb.init(mode="offline", project="validation-fix", name="immediate-validation")
    
    # Create validation data
    input_img = Image.new('RGB', (256, 256), color='red')
    target_img = Image.new('RGB', (256, 256), color='green')
    
    # Create validation pipe
    pipe = SimpleValidationPipe()
    
    # Generate validation image
    generated = pipe(prompt="Fix validation", edit_image=input_img, height=256, width=256)
    
    # Import the comparison grid function
    sys.path.append('DiffSynth-Studio/examples/qwen_image/model_training/')
    from train_with_segmentation_wandb import create_validation_comparison_grid
    
    # Create comparison grid
    grid = create_validation_comparison_grid(
        input_img, target_img, generated, 
        "Validation fix test", 0.1234
    )
    
    # Log to WandB
    wandb.log({
        "validation/comparison_grid": wandb.Image(grid),
        "validation/input": wandb.Image(input_img),
        "validation/target": wandb.Image(target_img),
        "validation/generated": wandb.Image(generated),
        "validation/step": 1500
    })
    
    print("✅ Validation images logged successfully!")
    wandb.finish()

if __name__ == "__main__":
    run_validation()
