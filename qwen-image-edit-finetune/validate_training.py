#!/usr/bin/env python3
"""
Validation script for trained Qwen Image Edit model.
Uses a sample from the training dataset to test the model.
"""
import torch
import pandas as pd
import os
from PIL import Image
from pathlib import Path
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

def main():
    # Load a sample from the training data
    dataset_path = "data/example_image_dataset"
    metadata_path = f"{dataset_path}/metadata_edit.csv"

    # Check if merged CSV exists, otherwise use first shard
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
    else:
        # Use first available shard
        shard_file = next(Path(dataset_path).glob("metadata_edit.shard*.csv"))
        df = pd.read_csv(shard_file)

    # Get first row as test sample
    sample = df.iloc[0]
    edit_image_path = f"{dataset_path}/{sample['edit_image']}"  # input image
    target_image_path = f"{dataset_path}/{sample['image']}"     # target image
    prompt = sample['prompt']

    print(f"Testing with sample:")
    print(f"  Input image: {edit_image_path}")
    print(f"  Target image: {target_image_path}")
    print(f"  Prompt: {prompt}")

    # Load the pipeline
    print("\nLoading pipeline...")
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

    # Load the trained LoRA
    lora_path = "models/train/Qwen-Image-Edit_lora/epoch-4.safetensors"
    print(f"Loading LoRA from: {lora_path}")
    pipe.load_lora(pipe.dit, lora_path)

    # Load and process the input image
    print("\nProcessing image...")
    edit_image = Image.open(edit_image_path).convert("RGB").resize((1024, 1024))

    # Generate the edited image
    print("Generating edited image...")
    result_image = pipe(
        prompt,
        edit_image=edit_image,
        seed=42,
        num_inference_steps=40,
        height=1024,
        width=1024
    )

    # Save results
    os.makedirs("validation_results", exist_ok=True)

    # Save input, target, and generated images
    edit_image.save("validation_results/input_image.jpg")
    target_image = Image.open(target_image_path).convert("RGB").resize((1024, 1024))
    target_image.save("validation_results/target_image.jpg")
    result_image.save("validation_results/generated_image.jpg")

    # Save prompt for reference
    with open("validation_results/prompt.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Input: {edit_image_path}\n")
        f.write(f"Target: {target_image_path}\n")

    print("\nValidation complete!")
    print("Results saved to validation_results/:")
    print("  - input_image.jpg (original)")
    print("  - target_image.jpg (ground truth)")
    print("  - generated_image.jpg (model output)")
    print("  - prompt.txt (test details)")

if __name__ == "__main__":
    main()