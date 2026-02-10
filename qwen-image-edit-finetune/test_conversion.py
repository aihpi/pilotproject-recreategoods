# Quick key format inspection
from diffusers import QwenImageEditPipeline
import torch

# Load the pipeline to see component names
pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")

# Check what the transformer component is actually called
print("Pipeline components:")
for attr_name in dir(pipe):
    if not attr_name.startswith('_') and hasattr(pipe, attr_name):
        attr_value = getattr(pipe, attr_name)
        if hasattr(attr_value, '__class__') and 'torch' in str(type(attr_value)):
            print(f"   - {attr_name}: {type(attr_value).__name__}")

# Check transformer specifically
if hasattr(pipe, 'transformer'):
    print(f"\nTransformer component: {type(pipe.transformer)}")
    print(f"Transformer class name: {pipe.transformer.__class__.__name__}")