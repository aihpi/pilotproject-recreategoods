#!/usr/bin/env python3
import sys
sys.path.append('.')

def minimal_wandb_test():
    try:
        import wandb
        from PIL import Image
        
        # Initialize wandb
        wandb.init(mode="offline", project="minimal-test", name="debug-test")
        
        # Create a simple test image
        test_img = Image.new('RGB', (100, 100), color='blue')
        
        # Log the image with minimal setup
        wandb.log({"test_image": wandb.Image(test_img)})
        
        print("✅ Minimal WandB test passed!")
        wandb.finish()
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")

if __name__ == "__main__":
    minimal_wandb_test()
