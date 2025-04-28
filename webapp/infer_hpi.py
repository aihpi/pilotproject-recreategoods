import torch
from diffusers import DiffusionPipeline
import PIL.Image
from diffusers.models import modeling_utils

# Monkey patch to fix dimension mismatch
original_load_state_dict = modeling_utils.load_state_dict

# The function signature needs to match the original
def patched_load_state_dict(checkpoint_file, **kwargs):
    state_dict = original_load_state_dict(checkpoint_file, **kwargs)
    
    # Handle specific dimension mismatch in x_embedder.weight
    if 'x_embedder.weight' in state_dict and state_dict['x_embedder.weight'].shape[1] != 64:
        # Reshape to match expected dimensions
        original_shape = state_dict['x_embedder.weight'].shape
        # Take first 64 columns or use another strategy based on your needs
        state_dict['x_embedder.weight'] = state_dict['x_embedder.weight'][:, :64]
        print("\n" + "="*80)
        print(f"Reshaped x_embedder.weight from {original_shape} to {state_dict['x_embedder.weight'].shape}")
        print("="*80)
    
    return state_dict

# Apply the patch
modeling_utils.load_state_dict = patched_load_state_dict

# Load the model with custom pipeline
pipe = DiffusionPipeline.from_pretrained(
    "aihpi/fashion-edit-model",
    custom_pipeline="aihpi/fashion-edit-model",
    torch_dtype=torch.bfloat16
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
print(f"Using device: {device}")

# Load an image
input_image = PIL.Image.open("fashion_image.jpg").convert("RGB")
input_image = input_image.resize((768, 768))

# Generate transformed image
prompt = "Convert this outfit to winter style with wool textures"

result = pipe(
    prompt=prompt,
    image=input_image,
    num_inference_steps=6,
    guidance_scale=7.5
).images[0]

result.save("winter_outfit.png")

print(f"Successfully generated and saved result to winter_outfit.png")

# Restore original function
modeling_utils.load_state_dict = original_load_state_dict