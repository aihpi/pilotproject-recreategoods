#!/usr/bin/env python3

import sys
import os
import json
import glob
from pathlib import Path

import torch
from PIL import Image
from lora_utils import LocalQwenImageEdit, convert_diffsynth_lora_to_diffusers, smart_resize_and_pad_to_1024

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
        # Calculate scale factor - use the larger one that fits within target
        scale_factor = min(target_size / original_width, target_size / original_height)

        # Only scale up, don't scale down unnecessarily
        if scale_factor > 1.0:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Ensure we don't exceed target_size due to rounding
            if new_width > target_size:
                new_width = target_size
            if new_height > target_size:
                new_height = target_size

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   Scaled up by {scale_factor:.3f} to: {new_width}x{new_height}")
            original_width, original_height = new_width, new_height

    # Create target canvas
    final_image = Image.new("RGB", (target_size, target_size), background_color)

    # Center the resized image
    x_offset = (target_size - original_width) // 2
    y_offset = (target_size - original_height) // 2

    # Paste onto canvas
    final_image.paste(image, (x_offset, y_offset))

    padding_x = target_size - original_width
    padding_y = target_size - original_height
    print(f"   Final: {target_size}x{target_size} (padding: {padding_x}px horizontal, {padding_y}px vertical)")

    return final_image

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

class LocalQwenImageEdit:
    def __init__(self, model_path: str, device: str = "cuda", base_model_id: str = "Qwen/Qwen-Image-Edit", lora_name: str = ""):
        """
        Initialize local Qwen-Image-Edit model for inference using Hugging Face

        Args:
            model_path: Path to the trained LoRA checkpoint
            device: Device to run inference on
            base_model_id: Base model ID from Hugging Face Hub
            lora_name: Name of the LoRA model (for identification)
        """
        self.model_path = model_path
        self.device = device
        self.base_model_id = base_model_id
        self.lora_name = lora_name
        self.model = None
        self.processor = None

        self._load_model()

    def _load_model(self):
        """Load the trained Qwen-Image-Edit model using diffusers pipeline"""
        try:
            from diffusers import QwenImageEditPipeline
            import torch

            print(f"Loading QwenImageEditPipeline from: {self.base_model_id}")

            # Load the pipeline
            self.pipe = QwenImageEditPipeline.from_pretrained(self.base_model_id)
            print("✓ Pipeline loaded")

            # Configure pipeline
            self.pipe.to(torch.bfloat16)
            self.pipe.to(self.device)
            self.pipe.set_progress_bar_config(disable=True)  # Disable progress bar for cleaner output

            # Load LoRA weights if available
            if self.model_path and os.path.exists(self.model_path) and self.lora_name:
                print(f"Loading LoRA weights from {self.model_path}")
                try:
                    # First, try converting DiffSynth-Studio format to diffusers format
                    full_path = os.path.join(self.model_path, self.lora_name)
                    print(f"Converting LoRA from DiffSynth-Studio format: {full_path}")

                    # Convert the weights
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

        Returns:
            Path to saved edited image
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        try:
            # Load and smart resize + pad image to 1024x1024
            print(f"🖼️  Processing image: {image_path}")
            image = smart_resize_and_pad_to_1024(image_path, target_size=1024, background_color="white")

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

def create_output_directories(base_dir, step_name):
    """Create separate directories for validation and test results"""
    validation_dir = Path(base_dir) / "validation_results" / step_name
    test_dir = Path(base_dir) / "test_results" / step_name

    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    return validation_dir, test_dir

def save_metadata(output_dir, step_name, checkpoint_path, prompt, image_path, settings):
    """Save metadata about the test run"""
    metadata = {
        "step": step_name,
        "checkpoint_path": checkpoint_path,
        "prompt": prompt,
        "source_image": image_path,
        "settings": settings,
        "generated_files": {
            "original": f"47_original.png",
            "edited": f"47_edited.png"
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_path

def test_lora_checkpoint(pipeline, checkpoint_path, validation_sample, test_sample, output_base_dir):
    """Test a single LoRA checkpoint and save results"""
    step_name = Path(checkpoint_path).stem  # e.g., "step-147000"
    print(f"\n{'='*60}")
    print(f"Testing LoRA checkpoint: {step_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation: Image {validation_sample['row']}, Prompt: '{validation_sample['prompt']}'")
    print(f"Test: Image {test_sample['row']}, Prompt: '{test_sample['prompt']}'")
    print(f"{'='*60}")

    # Create output directories
    validation_dir, test_dir = create_output_directories(output_base_dir, step_name)

    # Unload previous LoRA and load new one
    try:
        lora_filename = Path(checkpoint_path).name

        # Unload any existing LoRA weights and clear GPU memory
        print("🔄 Unloading previous LoRA weights...")
        try:
            # First disable any active adapters
            transformer = getattr(pipeline.pipe, 'transformer', None) or getattr(pipeline.pipe, 'unet', None)
            if transformer and hasattr(transformer, 'disable_adapters'):
                transformer.disable_adapters()
                print("   ✓ Adapters disabled")

            # Then unload the LoRA weights
            pipeline.pipe.unload_lora_weights()
            print("   ✓ LoRA weights unloaded")
        except Exception as e:
            print(f"   ⚠️  Unload warning: {e}")  # May not have any LoRA loaded for first checkpoint

        # Clear GPU cache to ensure memory is freed
        print("🧹 Clearing GPU memory cache...")
        torch.cuda.empty_cache()

        # Show current GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            print(f"   GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

        # Load new LoRA weights
        print(f"📥 Loading LoRA weights: {lora_filename}")
        model_dir = str(Path(checkpoint_path).parent)

        # Debug: Check what's in the LoRA file
        try:
            import safetensors
            full_path = str(Path(model_dir) / lora_filename)
            with safetensors.safe_open(full_path, framework="pt") as f:
                keys = list(f.keys())
                print(f"   📋 LoRA file contains {len(keys)} keys")
                print(f"   📋 Sample keys: {keys[:3]}...")  # Show first 3 keys

                # Check for different prefixes
                prefixes = set(key.split('.')[0] for key in keys if '.' in key)
                print(f"   📋 Key prefixes found: {prefixes}")
        except Exception as e:
            print(f"   ⚠️  Could not inspect LoRA file: {e}")

        # Load the LoRA weights with proper adapter name and prefix=None as suggested
        print(f"   Loading from directory: {model_dir}")
        print(f"   Weight name: {lora_filename}")

        # Use conversion-based loading approach
        full_path = str(Path(model_dir) / lora_filename)

        try:
            print("   🔄 Converting and loading LoRA weights...")
            # Convert DiffSynth-Studio format to diffusers format
            converted_path = convert_diffsynth_lora_to_diffusers(full_path)

            # Load the converted weights
            converted_dir = os.path.dirname(converted_path)
            converted_name = os.path.basename(converted_path)

            pipeline.pipe.load_lora_weights(
                converted_dir,
                weight_name=converted_name,
                adapter_name="current_lora"
            )

            # Explicitly enable the adapter
            pipeline.pipe.enable_lora()
            print("   ✓ LoRA weights converted and loaded successfully!")

        except Exception as e:
            print(f"   ❌ Conversion and loading failed: {e}")
            print(f"   ❌ Cannot proceed without LoRA weights - exiting")
            sys.exit(1)

        # Enable the adapter explicitly
        transformer = getattr(pipeline.pipe, 'transformer', None) or getattr(pipeline.pipe, 'unet', None)
        if transformer and hasattr(transformer, 'enable_adapters'):
            transformer.enable_adapters()
            print("   ✓ LoRA adapters enabled")

        # Set the adapter as active
        if transformer and hasattr(transformer, 'set_adapter'):
            transformer.set_adapter("current_lora")
            print("   ✓ Set current_lora as active adapter")

        # Verify LoRA is loaded by checking adapter state
        try:
            if transformer:
                # Check for LoRA adapters
                adapters = getattr(transformer, 'peft_config', None)
                if adapters:
                    print(f"✓ LoRA adapters detected: {list(adapters.keys())}")

                # Count LoRA layers
                lora_layers = [name for name, _ in transformer.named_modules() if 'lora' in name.lower()]
                print(f"✓ Found {len(lora_layers)} LoRA layers in transformer")

                # Check if adapter is active
                if hasattr(transformer, 'active_adapters'):
                    print(f"✓ Active adapters: {transformer.active_adapters}")
                elif hasattr(transformer, 'active_adapter'):
                    print(f"✓ Active adapter: {transformer.active_adapter}")
            else:
                print("⚠️  Could not find transformer/unet component")

        except Exception as e:
            print(f"⚠️  Could not verify LoRA loading: {e}")

        # Show GPU memory after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            print(f"✓ LoRA weights loaded! GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

    except Exception as e:
        print(f"❌ Failed to load LoRA weights: {e}")
        # Try to clear memory even if loading failed
        torch.cuda.empty_cache()
        return

    # Test settings with fixed seeds for consistent comparison across all models
    validation_seed = 0  # Fixed seed for validation across all checkpoints
    test_seed = 500     # Fixed seed for test across all checkpoints

    # Alternative: Use step-dependent seeds for variation
    # step_number = int(step_name.split('-')[1])
    # validation_seed = step_number % 1000
    # test_seed = (step_number + 500) % 1000

    # Run validation test
    print("🧪 Running validation test...")
    try:
        val_img_name = f"{validation_sample['row']}_original.png"
        val_edited_name = f"{validation_sample['row']}_edited.png"
        validation_output = validation_dir / val_edited_name

        test_settings = {
            "num_inference_steps": 40,
            "true_cfg_scale": 4.0,
            "seed": validation_seed
        }

        print(f"   Using validation seed: {validation_seed}")
        pipeline.edit_image(validation_sample['image_path'], validation_sample['prompt'], str(validation_output), **test_settings)

        # Save original image copy
        source_image = Image.open(validation_sample['image_path'])
        source_image.save(validation_dir / val_img_name)

        save_metadata(validation_dir, step_name, checkpoint_path, validation_sample['prompt'], val_img_name, test_settings)
        print(f"✓ Validation results saved to: {validation_dir}")

    except Exception as e:
        print(f"❌ Validation test failed: {e}")

    # Run test set evaluation
    print("🧪 Running test set evaluation...")
    try:
        test_img_name = f"{test_sample['row']}_original.png"
        test_edited_name = f"{test_sample['row']}_edited.png"
        test_output = test_dir / test_edited_name

        test_settings = {
            "num_inference_steps": 40,
            "true_cfg_scale": 4.0,
            "seed": test_seed
        }

        print(f"   Using test seed: {test_seed}")
        pipeline.edit_image(test_sample['image_path'], test_sample['prompt'], str(test_output), **test_settings)

        # Save original image copy
        source_image = Image.open(test_sample['image_path'])
        source_image.save(test_dir / test_img_name)

        save_metadata(test_dir, step_name, checkpoint_path, test_sample['prompt'], test_img_name, test_settings)
        print(f"✓ Test results saved to: {test_dir}")

    except Exception as e:
        print(f"❌ Test evaluation failed: {e}")

def run_viescore_evaluation(lora_validation_dir, step_name, lora_name, env_file_path=None):
    """Run VIEScore evaluation directly using the LocalQwenImageEdit class"""
    try:
        # Import the VIEScore classes directly
        sys.path.append('viescore')
        from viescore.script import VIEScoreViaOpenWebUI, OpenWebUIClient

        # Load environment variables from specified location
        if env_file_path and os.path.exists(env_file_path):
            from dotenv import load_dotenv
            load_dotenv(env_file_path)
            print(f"🔑 Loaded API key from: {env_file_path}")
        else:
            # Try default locations
            from dotenv import load_dotenv
            load_dotenv(".env")  # Current directory
            load_dotenv("viescore/.env")  # VIEScore directory
            load_dotenv("../.env")  # Parent directory

        print(f"🔍 Running VIEScore evaluation for {step_name}...")

        # Initialize OpenWebUI client for VIEScore
        try:
            client = OpenWebUIClient(base_url="https://chat.hpi-sci.de")
            evaluator = VIEScoreViaOpenWebUI(client, model_name="mistralai/Pixtral-12B-2409")
        except Exception as e:
            print(f"❌ Could not initialize VIEScore evaluator: {e}")
            print(f"💡 Make sure API_KEY is set in your .env file")
            if env_file_path:
                print(f"💡 Checked: {env_file_path}")
            return False

        # Paths for evaluation - look in test directory for image 47
        test_dir = lora_validation_dir.parent.parent / "test_results" / step_name
        original_path = str(test_dir / "47_original.png")
        edited_path = str(test_dir / "47_edited.png")
        instruction = "add wide tulle sleeves"

        # Check if files exist
        if not os.path.exists(original_path) or not os.path.exists(edited_path):
            print(f"❌ Required images not found in {test_dir}")
            print(f"   Looking for: {original_path}")
            print(f"   Looking for: {edited_path}")
            if test_dir.exists():
                print(f"   Directory contents: {list(test_dir.glob('*'))}")
            return False

        # Run VIEScore evaluation
        scores = evaluator.evaluate_edit(original_path, edited_path, instruction)

        # Save results to the validation directory
        results_file = lora_validation_dir / f"viescore_results_{step_name}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "step": step_name,
                "lora_name": lora_name,
                "scores": scores,
                "instruction": instruction,
                "original_image": "47_original.png",
                "edited_image": "47_edited.png"
            }, f, indent=2)

        # Display results
        if 'semantic_consistency' in scores:
            print(f"✓ VIEScore evaluation completed for {step_name}")
            print(f"  Semantic Consistency: {scores.get('semantic_consistency', 'N/A')}/10")
            print(f"  Perceptual Quality: {scores.get('perceptual_quality', 'N/A')}/10")
            print(f"  Overall Score: {scores.get('overall', 'N/A')}/10")
            print(f"  Results saved to: {results_file}")
            return True
        else:
            print(f"❌ VIEScore evaluation returned incomplete results for {step_name}")
            return False

    except Exception as e:
        print(f"❌ Failed to run VIEScore evaluation: {e}")
        return False

def load_dataset_samples():
    """Load validation and test samples"""
    import pandas as pd

    # For validation, use validation.csv with control images as input
    validation_csv = "data/example_image_dataset/validation.csv"
    if not os.path.exists(validation_csv):
        print(f"❌ Validation CSV not found: {validation_csv}")
        sys.exit(1)

    val_df = pd.read_csv(validation_csv)
    # Use first sample from validation set
    val_row = val_df.iloc[0]
    validation_sample = {
        'row': 'val_0',  # Custom identifier for validation
        'prompt': val_row['prompt'],
        'image_path': f"data/example_image_dataset/{val_row['edit_image']}"  # Use control image as input
    }

    # For testing, use image 47 as originally planned
    test_sample = {
        'row': 47,
        'prompt': 'add wide tulle sleeves',
        'image_path': 'viescore/extracted_images/47.png'
    }

    # Verify both images exist
    for sample, name in [(validation_sample, "validation"), (test_sample, "test")]:
        if not os.path.exists(sample['image_path']):
            print(f"❌ {name.title()} image not found: {sample['image_path']}")
            sys.exit(1)

    return validation_sample, test_sample

def main(env_file_path=None):
    print("🚀 Starting LoRA Validation Pipeline with VIEScore Integration")
    print("="*80)

    if env_file_path:
        print(f"🔑 Using API key from: {env_file_path}")

    # Load validation and test samples
    validation_sample, test_sample = load_dataset_samples()

    print(f"Validation sample: {validation_sample['row']}, Prompt: '{validation_sample['prompt']}'")
    print(f"Test sample: Image {test_sample['row']}, Prompt: '{test_sample['prompt']}'")

    # Verify images
    for sample, name in [(validation_sample, "validation"), (test_sample, "test")]:
        image = Image.open(sample['image_path']).convert('RGB')
        print(f"{name.title()} image size: {image.size}")

    # Find all LoRA checkpoints
    lora_path = "models/curriculum/Qwen-Image-Edit_day_long_lora_rank16_lr8e-05"
    print(f"\nLooking for LoRA weights in: {lora_path}")

    checkpoints = glob.glob(f"{lora_path}/*.safetensors")
    # Filter out diffusers converted files, use only original checkpoints
    checkpoints = [cp for cp in checkpoints if '_diffusers' not in Path(cp).stem]
    checkpoints = sorted(checkpoints, key=lambda x: int(Path(x).stem.split('-')[1]))

    print(f"Found {len(checkpoints)} LoRA checkpoints")

    if not checkpoints:
        print("❌ NO LoRA CHECKPOINTS FOUND!")
        print(f"Directory contents: {os.listdir(lora_path) if os.path.exists(lora_path) else 'Directory does not exist'}")
        sys.exit(1)

    # Process all checkpoints (remove [:3] limit for full validation)
    # checkpoints = checkpoints[:3]  # Uncomment this line to test with just 3 checkpoints

    # Show first few checkpoints
    print("Testing checkpoints:")
    for i, cp in enumerate(checkpoints):
        print(f"  {i+1}. {Path(cp).name}")

    # Create base output directory
    output_base_dir = "lora_validation_results"
    Path(output_base_dir).mkdir(exist_ok=True)

    print(f"\nResults will be saved to: {output_base_dir}/")
    print(f"Processing {len(checkpoints)} checkpoints...")

    # Initialize the pipeline once (without LoRA)
    print(f"\n🔧 Initializing base pipeline...")
    try:
        pipeline = LocalQwenImageEdit("", device="cuda", lora_name="")  # Empty paths = base model only
        print("✓ Base pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize base pipeline: {e}")
        sys.exit(1)

    # Option to enable/disable VIEScore evaluation
    enable_viescore = True
    successful_evaluations = []
    failed_evaluations = []

    # Test each checkpoint
    for i, checkpoint_path in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Processing checkpoint...")
        test_lora_checkpoint(pipeline, checkpoint_path, validation_sample, test_sample, output_base_dir)

        # Run VIEScore evaluation if enabled
        if enable_viescore:
            step_name = Path(checkpoint_path).stem
            lora_name = Path(checkpoint_path).name
            validation_dir = Path(output_base_dir) / "validation_results" / step_name

            if validation_dir.exists():
                success = run_viescore_evaluation(validation_dir, step_name, lora_name, env_file_path)
                if success:
                    successful_evaluations.append(step_name)
                else:
                    failed_evaluations.append(step_name)

    print(f"\n🎉 All LoRA checkpoints tested!")
    print(f"Results saved in:")
    print(f"  - {output_base_dir}/validation_results/")
    print(f"  - {output_base_dir}/test_results/")

    if enable_viescore:
        print(f"\n📊 VIEScore Evaluation Summary:")
        print(f"  ✓ Successful evaluations: {len(successful_evaluations)}")
        print(f"  ❌ Failed evaluations: {len(failed_evaluations)}")

        if successful_evaluations:
            print(f"  Successfully evaluated: {', '.join(successful_evaluations[:5])}{'...' if len(successful_evaluations) > 5 else ''}")

        if failed_evaluations:
            print(f"  Failed to evaluate: {', '.join(failed_evaluations[:5])}{'...' if len(failed_evaluations) > 5 else ''}")

    # Final cleanup
    print(f"\n🧹 Final cleanup...")
    try:
        # Disable adapters first
        transformer = getattr(pipeline.pipe, 'transformer', None) or getattr(pipeline.pipe, 'unet', None)
        if transformer and hasattr(transformer, 'disable_adapters'):
            transformer.disable_adapters()
            print("✓ Adapters disabled")

        # Then unload LoRA weights
        pipeline.pipe.unload_lora_weights()
        torch.cuda.empty_cache()
        print("✓ Final GPU memory cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

    # Final memory report
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        print(f"📊 Final GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

    print(f"\n📋 Next steps:")
    print(f"  1. Check VIEScore results in evaluation_results_*.csv files")
    print(f"  2. Compare progression across training steps")
    print(f"  3. Identify best performing checkpoints")
    print(f"  4. Analyze validation vs test performance")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Validation Pipeline with VIEScore")
    parser.add_argument("--env-file", type=str, default=None,
                       help="Path to .env file containing API_KEY for VIEScore evaluation")
    parser.add_argument("--disable-viescore", action="store_true",
                       help="Disable VIEScore evaluation (only generate images)")

    args = parser.parse_args()

    # Pass the env file path to main
    main(env_file_path=args.env_file)