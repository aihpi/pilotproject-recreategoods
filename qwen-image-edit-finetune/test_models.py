import torch
from PIL import Image
import os
import pandas as pd
import argparse
import shutil
import yaml
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import FluxImagePipeline

def load_model_config(config_path):
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_pipeline(model_config):
    """Initialize pipeline based on model configuration"""
    pipeline_type = model_config['pipeline_type']
    model_configs = [ModelConfig(**config) for config in model_config['model_configs']]

    common_args = {
        'torch_dtype': getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
        'device': model_config.get('device', 'cuda'),
        'model_configs': model_configs,
    }

    if pipeline_type == 'qwen':
        # Add Qwen-specific configs
        if 'tokenizer_config' in model_config:
            common_args['tokenizer_config'] = model_config['tokenizer_config']
        if 'processor_config' in model_config:
            common_args['processor_config'] = ModelConfig(**model_config['processor_config'])
        return QwenImagePipeline.from_pretrained(**common_args)

    elif pipeline_type == 'flux':
        return FluxImagePipeline.from_pretrained(**common_args)

    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

def test_model(lora_path, test_image_path, prompt, model_name, output_dir="test_outputs", model_config=None):
    """Test an Image Edit model with a given LoRA checkpoint or baseline"""

    # Create organized folder structure: output_dir/model_name/image_name/
    image_name = os.path.basename(test_image_path).split('.')[0]
    result_folder = os.path.join(output_dir, model_name, image_name)

    # Create specific folder for this test result
    os.makedirs(result_folder, exist_ok=True)

    # Initialize the pipeline using the provided config
    print(f"Loading pipeline...")
    if model_config is None:
        raise ValueError("model_config is required")

    pipe = initialize_pipeline(model_config)

    # Load the LoRA if provided
    if lora_path:
        print(f"Loading LoRA from {lora_path}")
        pipe.load_lora(pipe.dit, lora_path)
    else:
        print("Using baseline model (no LoRA)")

    # Load and resize the test image
    print(f"Loading test image from {test_image_path}")
    image = Image.open(test_image_path).resize((1024, 1024))

    # Save the original image in the result folder
    original_output_path = f"{result_folder}/before.jpg"
    image.save(original_output_path)
    print(f"Saved original image to: {original_output_path}")

    # Save the edit instruction as a text file
    instruction_path = f"{result_folder}/instruction.txt"
    with open(instruction_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Saved instruction to: {instruction_path}")

    print(f"Generating image for prompt: {prompt}")
    try:
        # Use different parameters based on pipeline type
        pipeline_type = model_config.get('pipeline_type', 'qwen')

        if pipeline_type == 'qwen':
            result_image = pipe(
                prompt,
                edit_image=image,
                seed=42,
                num_inference_steps=40,
                height=1024,
                width=1024
            )
        elif pipeline_type == 'flux':
            result_image = pipe(
                prompt=prompt,
                step1x_reference_image=image,
                height=1024,
                width=1024,
                cfg_scale=6,
                seed=42
            )
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

        # Save the result in the same folder
        output_path = f"{result_folder}/after.jpg"
        result_image.save(output_path)
        print(f"Saved result to: {output_path}")

        print(f"✓ Complete test result saved in: {result_folder}")
        return result_folder

    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test Image Edit models')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model configuration YAML file')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Override model directory from config')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory from config')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Override specific model checkpoint from config')
    parser.add_argument('--test_baseline', action='store_true',
                       help='Include baseline model in testing')
    parser.add_argument('--num_test_cases', type=int, default=5,
                       help='Number of test cases to run')

    args = parser.parse_args()

    # Load model configuration
    print(f"Loading model config from {args.model_config}")
    model_config = load_model_config(args.model_config)

    # Use config values, but allow command line overrides
    model_dir = args.model_dir or model_config.get('model_dir', 'models/train')
    data_dir = args.data_dir or model_config.get('data_dir', 'data/example_image_dataset')
    output_dir = args.output_dir or model_config.get('output_dir', 'test_outputs')
    default_model_name = args.model_name or model_config.get('model_name', None)

    # Get available model checkpoints
    if os.path.exists(model_dir):
        available_models = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
        available_models.sort()  # Sort by step number
        print(f"Found {len(available_models)} model checkpoints:")
        for model in available_models:
            print(f"  - {model}")
    else:
        available_models = []
        print(f"Model directory {model_dir} not found.")

    # Load metadata to get test cases
    metadata_path = os.path.join(data_dir, "metadata_edit.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        print(f"Make sure the dataset directory contains metadata_edit.csv")
        return
    df = pd.read_csv(metadata_path)

    print(f"Found {len(df)} test cases in dataset")

    # Select test cases
    test_cases = df.head(args.num_test_cases)

    # Determine which models to test
    models_to_test = []

    # Add baseline if requested
    if args.test_baseline:
        models_to_test.append((None, "baseline"))

    # Add specific model or all available models
    if default_model_name:
        if default_model_name in available_models:
            model_path = os.path.join(model_dir, default_model_name)
            model_name = default_model_name.replace('.safetensors', '')
            models_to_test.append((model_path, model_name))
        else:
            print(f"Error: Model {default_model_name} not found in {model_dir}")
            return
    elif available_models:
        # Test with the latest checkpoint by default
        model_to_test = available_models[-1]
        model_path = os.path.join(model_dir, model_to_test)
        model_name = model_to_test.replace('.safetensors', '')
        models_to_test.append((model_path, model_name))

    if not models_to_test:
        print("No models to test. Use --test_baseline or ensure model checkpoints exist.")
        return

    print(f"\nTesting {len(models_to_test)} models:")
    for i, (model_path, model_name) in enumerate(models_to_test):
        if model_path is None:
            print(f"  {i+1}. Baseline (no LoRA)")
        else:
            print(f"  {i+1}. {model_name}")

    for model_path, model_name in models_to_test:
        print(f"\n{'='*50}")
        if model_path is None:
            print(f"Testing BASELINE model (no LoRA)")
        else:
            print(f"Testing model: {model_name}")
        print(f"{'='*50}")

        for idx, row in test_cases.iterrows():
            image_path = os.path.join(data_dir, row['image'])
            prompt = row['prompt']

            if os.path.exists(image_path):
                # Check if results already exist
                image_name = os.path.basename(image_path).split('.')[0]
                result_folder = os.path.join(output_dir, model_name, image_name)
                after_image_path = os.path.join(result_folder, "after.jpg")

                if os.path.exists(after_image_path):
                    print(f"\nTest case {idx + 1}: SKIPPING (already exists)")
                    print(f"Image: {row['image']}")
                    print(f"Existing result: {result_folder}")
                    print("-" * 50)
                    continue

                print(f"\nTest case {idx + 1}:")
                print(f"Image: {row['image']}")
                print(f"Prompt: {prompt}")
                print("-" * 50)

                try:
                    result = test_model(model_path, image_path, prompt, model_name, output_dir, model_config)
                    if result:
                        print(f"✓ Success: {result}")

                        # Copy original image to result folder for easy comparison
                        original_dest = os.path.join(result, "original.jpg")
                        if not os.path.exists(original_dest):
                            shutil.copy2(image_path, original_dest)

                    else:
                        print("✗ Failed to generate image")
                except Exception as e:
                    print(f"✗ Error: {e}")
            else:
                print(f"✗ Image not found: {image_path}")

    print(f"\nTest completed! Check the '{output_dir}' directory for results.")
    print(f"Results are organized as: {output_dir}/model_name/image_name/")
    print(f"Each test folder contains: before.jpg, after.jpg, original.jpg, and instruction.txt")
    print(f"You can now compare baseline vs fine-tuned model performance.")

if __name__ == "__main__":
    main()