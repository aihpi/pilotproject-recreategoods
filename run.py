import torch
from diffusers import DiffusionPipeline
import PIL.Image
pipe = DiffusionPipeline.from_pretrained(
    "aihpi/fashion-edit-model",
    custom_pipeline="aihpi/fashion-edit-model",
    torch_dtype=torch.bfloat16
)
# Load an image
input_image = PIL.Image.open("fashion-image.jpg").convert("RGB")
# convert to tensor
input_image = input_image.resize((256, 256))
# Generate transformed image
prompt = "Remove the sleeves, to turn this into a sleeveless dress"
result = pipe(
    height=input_image.height,
    width=input_image.width,
    prompt=prompt,
    image=input_image,
    num_inference_steps=6,
    guidance_scale=7.5,
    strength=1
).images[0]
input_image.save("input_image_resized.png")
result.save("short_sleeve.png")
