#!/usr/bin/env python3
"""
Shared utilities for LoRA weight conversion and management
Provides consistent LoRA handling across validation and VIEScore scripts
"""

import os
import sys
import torch
from PIL import Image, ImageOps
from pathlib import Path
from typing import Optional


def convert_diffsynth_lora_to_diffusers(input_path: str, output_path: str = None) -> str:
    """
    Convert DiffSynth-Studio LoRA weights to diffusers format

    Args:
        input_path: Path to DiffSynth-Studio LoRA checkpoint
        output_path: Optional output path for converted weights

    Returns:
        Path to converted weights file
    """
    from safetensors.torch import load_file, save_file

    print(f"Converting LoRA weights from: {input_path}")

    # Load DiffSynth-Studio weights
    diffsynth_weights = load_file(input_path)

    # Create mapping from DiffSynth-Studio keys to diffusers keys
    diffusers_weights = {}

    # Convert key mappings
    # The LoRA weights need to be prefixed with the transformer component name
    # diffusers expects: "transformer.transformer_blocks.X.attn.to_k.lora_A.weight"
    # but our keys are: "transformer_blocks.X.attn.to_k.lora_A.default.weight"

    for old_key, tensor in diffsynth_weights.items():
        new_key = old_key

        # Remove .default suffix if present
        if ".default.weight" in new_key:
            new_key = new_key.replace(".default.weight", ".weight")
        elif ".default" in new_key:
            new_key = new_key.replace(".default", "")

        # Add transformer prefix for diffusers compatibility
        if not new_key.startswith("transformer."):
            new_key = f"transformer.{new_key}"

        diffusers_weights[new_key] = tensor

    print(f"Processed {len(diffusers_weights)} weight tensors")

    # Set output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"{base_name}_diffusers.safetensors")

    # Save converted weights with proper tensor format
    # Ensure all tensors are contiguous and on CPU
    cleaned_weights = {}
    for key, tensor in diffusers_weights.items():
        if hasattr(tensor, 'contiguous'):
            cleaned_weights[key] = tensor.contiguous().cpu()
        else:
            cleaned_weights[key] = tensor

    save_file(cleaned_weights, output_path)
    print(f"✓ Converted weights saved to: {output_path}")

    return output_path


def smart_resize_and_pad_to_1024(image_path: str, target_size: int = 1024, background_color: str = "white") -> Image.Image:
    """
    Smart resize + pad to 1024x1024: maximize resolution, minimize padding

    Strategy:
    1. Scale up to use maximum possible resolution within 1024x1024
    2. Pad only the dimension that needs it

    Args:
        image_path: Path to input image
        target_size: Target size (default 1024x1024)
        background_color: Background color for padding

    Returns:
        PIL Image at target_size x target_size with optimal quality
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Get original dimensions
    original_width, original_height = image.size
    print(f"   Original size: {original_width}x{original_height}")

    # If image is already exactly the target size, return as-is
    if original_width == target_size and original_height == target_size:
        print(f"   Already {target_size}x{target_size} - no processing needed")
        return image

    # If image is already larger than target in both dimensions, scale down
    if original_width > target_size and original_height > target_size:
        scale_factor = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"   Scaled down by {scale_factor:.3f} to: {new_width}x{new_height}")
        original_width, original_height = new_width, new_height

    # Smart upscaling: scale up to use maximum resolution within 1024x1024
    if original_width < target_size or original_height < target_size:
        # Find the scale factor that maximizes size while staying within bounds
        scale_factor = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Only scale up if it actually increases the resolution
        if scale_factor > 1.0:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   Scaled up by {scale_factor:.3f} to: {new_width}x{new_height}")
            original_width, original_height = new_width, new_height

    # Pure padding to reach target size
    final_image = ImageOps.pad(image, (target_size, target_size), color=background_color)

    # Calculate actual padding applied
    padding_x = target_size - original_width
    padding_y = target_size - original_height
    print(f"   Final: {target_size}x{target_size} (padding: {padding_x}px horizontal, {padding_y}px vertical)")

    return final_image


class LocalQwenImageEdit:
    def __init__(self, model_path: str = "", device: str = "cuda", base_model_id: str = "Qwen/Qwen-Image-Edit", lora_name: str = "", resize: bool = True):
        """
        Initialize local Qwen-Image-Edit model for inference using Hugging Face

        Args:
            model_path: Path to the trained LoRA checkpoint directory
            device: Device to run inference on
            base_model_id: Base model ID from Hugging Face Hub
            lora_name: Name of the LoRA model file (for identification)
            resize: Whether to automatically resize images to 1024x1024
        """
        self.model_path = model_path
        self.device = device
        self.base_model_id = base_model_id
        self.lora_name = lora_name
        self.resize = resize
        self.pipe = None

        self._load_model()

    def _load_model(self):
        """Load the trained Qwen-Image-Edit model using diffusers pipeline"""
        try:
            from diffusers import QwenImageEditPipeline
            import torch

            print(f"Loading QwenImageEditPipeline from: {self.base_model_id}")

            # Load the pipeline
            self.pipe = QwenImageEditPipeline.from_pretrained(self.base_model_id, device_map="balanced", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
            print("✓ Pipeline loaded")

            # Configure pipeline
            self.pipe.set_progress_bar_config(disable=True)  # Disable progress bar for cleaner output

            # Load LoRA weights if available
            if self.model_path and os.path.exists(self.model_path) and self.lora_name:
                print(f"Loading LoRA weights from {self.model_path}")
                try:
                    # First, try converting DiffSynth-Studio format to diffusers format
                    full_path = os.path.join(self.model_path, self.lora_name)
                    if self.lora_name.endswith("_diffusers.safetensors"):
                        converted_path = full_path
                        print(f"Using pre-converted LoRA weights: {converted_path}")
                    else:
                        print(f"Converting LoRA from DiffSynth-Studio format: {full_path}")
                        converted_path = convert_diffsynth_lora_to_diffusers(full_path)

                    # Load the converted weights
                    converted_dir = os.path.dirname(converted_path)
                    converted_name = os.path.basename(converted_path)

                    self.pipe.load_lora_weights(converted_dir, weight_name=converted_name, adapter_name="current_lora")
                    self.pipe.enable_lora()
                    print("✓ Converted LoRA weights loaded successfully!")

                except Exception as e:
                    print(f"Warning: Could not convert and load LoRA weights: {e}")
                    print("Continuing with base model only")
            else:
                print("No LoRA name/path provided or path doesn't exist, using base model")

            print("✓ Model loaded successfully!")

            # Debug: Print pipeline components
            print("🔍 Pipeline components:")
            for attr_name in dir(self.pipe):
                if not attr_name.startswith('_') and hasattr(self.pipe, attr_name):
                    attr_value = getattr(self.pipe, attr_name)
                    if hasattr(attr_value, '__class__') and 'torch' in str(type(attr_value)):
                        print(f"   - {attr_name}: {type(attr_value).__name__}")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise e

    def edit_image(self, image_path: str, instruction: str, output_path: str,
                   num_inference_steps: int = 50, true_cfg_scale: float = 4.0, seed: int = 0) -> str:
        """
        Edit an image using the local model pipeline

        Args:
            image_path: Path to input image
            instruction: Edit instruction text
            output_path: Path to save edited image
            num_inference_steps: Number of inference steps
            true_cfg_scale: CFG scale for generation
            seed: Random seed for reproducibility

        Returns:
            Path to saved edited image
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        try:
            # Load image and optionally resize + pad to 1024x1024
            print(f"🖼️  Processing image: {image_path}")
            if self.resize:
                image = smart_resize_and_pad_to_1024(image_path, target_size=1024, background_color="white")
            else:
                image = Image.open(image_path).convert('RGB')

            # Prepare inputs for the pipeline
            inputs = {
                "image": image,
                "prompt": instruction,
                "generator": torch.manual_seed(seed),
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": " ",
                "num_inference_steps": num_inference_steps,
            }

            # Run inference
            with torch.inference_mode():
                output = self.pipe(**inputs)
                output_image = output.images[0]
                output_image.save(output_path)

            print(f"✓ Image edited and saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"Error during image editing: {e}")
            # Fallback: copy original image
            Image.open(image_path).save(output_path)
            return output_path

    def load_lora_adapter(self, lora_path: str, lora_name: str, adapter_name: str = "current_lora"):
        """Load a new LoRA adapter"""
        try:
            # Convert DiffSynth-Studio format to diffusers format
            full_path = os.path.join(lora_path, lora_name)
            print(f"Converting LoRA from DiffSynth-Studio format: {full_path}")

            # Convert the weights
            converted_path = convert_diffsynth_lora_to_diffusers(full_path)

            # Load the converted weights
            converted_dir = os.path.dirname(converted_path)
            converted_name = os.path.basename(converted_path)

            self.pipe.load_lora_weights(converted_dir, weight_name=converted_name, adapter_name=adapter_name)
            print("✓ New LoRA adapter loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load LoRA adapter: {e}")
            raise e

    def unload_lora_adapter(self, adapter_name: str = "current_lora"):
        """Unload current LoRA adapter"""
        try:
            # Disable the adapter first
            if hasattr(self.pipe, 'disable_lora'):
                self.pipe.disable_lora()
                print("   ✓ Adapters disabled")

            # Then unload the LoRA weights
            self.pipe.unload_lora_weights()
            print("   ✓ LoRA weights unloaded")

            # Clear GPU cache
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ⚠️  Unload warning: {e}")
