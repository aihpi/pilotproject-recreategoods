#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio')

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch

# Load the base Qwen-Image-Edit model (no training)
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# Test with a real image from your dataset
dataset_path = "data/example_image_dataset"
source_img_path = f"{dataset_path}/images/ad4a4536c07141d9ac6e58ebe728cad3.png"
target_img_path = f"{dataset_path}/control_images/ad4a4536c07141d9ac6e58ebe728cad3.png"
prompt = "Replace the floral print with a solid color in a complementary shade, keeping the contrast piping for texture variation."

print(f"Loading source image: {source_img_path}")
source_image = Image.open(source_img_path).convert('RGB')
print(f"Source image size: {source_image.size}")

print(f"Loading target image: {target_img_path}")
target_image = Image.open(target_img_path).convert('RGB')
print(f"Target image size: {target_image.size}")

print(f"Prompt: {prompt}")

# Test the base model
print("\n=== Testing Base Qwen-Image-Edit Model ===")
generated_image = pipe(prompt, edit_image=source_image, seed=42, num_inference_steps=40,
                      height=source_image.size[1], width=source_image.size[0],
                      edit_image_auto_resize=True)

# Save results
source_image.save("test_source.png")
target_image.save("test_target.png")
generated_image.save("test_generated_base.png")

print("Saved:")
print("- test_source.png (original)")
print("- test_target.png (ground truth)")
print("- test_generated_base.png (base model output)")
print("\nCompare these images to see if the base model preserves input structure!")