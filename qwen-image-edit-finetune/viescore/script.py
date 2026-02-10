"""
Evaluation Pipeline for Qwen-Image-Edit using VIEScore via OpenWebUI

This script:
1. Loads images and edit instructions from the dataset
2. Uses OpenWebUI API to run Qwen-Image-Edit for image editing
3. Evaluates results using Pixtral model via OpenWebUI for VIEScore metrics
"""

import os
import sys
import base64
import json
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional
from io import BytesIO
from dotenv import load_dotenv
import torch
from pathlib import Path

# Load environment variables from current directory first, then parent
load_dotenv(".env")  # Try current directory first
load_dotenv("../.env")  # Fallback to parent directory


class OpenWebUIClient:
    def __init__(self, base_url: str = "https://chat.hpi-sci.de", api_key: Optional[str] = None):
        """
        Initialize OpenWebUI API client

        Args:
            base_url: Base URL of OpenWebUI instance
            api_key: API key for authentication (if None, reads from .env)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY") or ""

        if not self.api_key:
            raise ValueError("API_KEY not found in .env file or provided")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        print(f"Initialized OpenWebUI client for {self.base_url}")

    def get_models(self) -> List[str]:
        """Get list of available models"""
        response = requests.get(f"{self.base_url}/api/models", headers=self.headers)
        response.raise_for_status()
        models = response.json()
        return [model["id"] for model in models.get("data", [])]

    def chat_completion(
        self, messages: List[Dict[str, any]], model: str, temperature: float = 0.7
    ) -> str:
        """
        Make a chat completion request

        Args:
            messages: List of message dictionaries
            model: Model ID to use
            temperature: Sampling temperature

        Returns:
            Response content
        """
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/chat/completions", headers=self.headers, json=payload
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]


class QwenImageEditViaOpenWebUI:
    def __init__(
        self, client: OpenWebUIClient, model_name: str = "edit_image.qwen-image-edit"
    ):
        """
        Initialize Qwen-Image-Edit via OpenWebUI

        Args:
            client: OpenWebUI client instance
            model_name: Name of the Qwen-Image-Edit model in OpenWebUI
        """
        self.client = client
        self.model_name = model_name

        # Verify model is available
        available_models = self.client.get_models()
        if model_name not in available_models:
            print(f"Warning: {model_name} not found in available models")
            print(f"Available models: {available_models}")
            # Try to find a similar model
            qwen_models = [
                m
                for m in available_models
                if "qwen" in m.lower() and "image" in m.lower()
            ]
            if qwen_models:
                self.model_name = qwen_models[0]
                print(f"Using {self.model_name} instead")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def edit_image(self, image_path: str, instruction: str, output_path: str) -> str:
        """
        Edit an image based on instruction using OpenWebUI API

        Args:
            image_path: Path to input image
            instruction: Edit instruction text
            output_path: Path to save edited image

        Returns:
            Path to saved edited image
        """
        # Encode the image
        image_b64 = self.encode_image(image_path)

        # Construct message for image editing
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Edit this image according to the following instruction: {instruction}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ]

        # Make the API call
        response = self.client.chat_completion(
            messages=messages, model=self.model_name, temperature=0.7
        )

        # Parse the response to get the edited image
        # The response format may vary based on OpenWebUI configuration
        # This assumes the model returns base64 encoded image in the response
        if "data:image" in response:
            # Extract base64 image from response
            import re

            match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", response)
            if match:
                image_data = base64.b64decode(match.group(1))
                image = Image.open(BytesIO(image_data))
                image.save(output_path)
                return output_path

        # If no image in response, save a placeholder or handle error
        print(f"Warning: No image returned for {image_path}")
        # Copy original as fallback
        Image.open(image_path).save(output_path)
        return output_path


class VIEScoreViaOpenWebUI:
    def __init__(
        self, client: OpenWebUIClient, model_name: str = "mistralai/Pixtral-12B-2409"
    ):
        """
        Initialize VIEScore evaluator using Pixtral via OpenWebUI

        Args:
            client: OpenWebUI client instance
            model_name: Name of the vision model (e.g., Pixtral) in OpenWebUI
        """
        self.client = client
        self.model_name = model_name

        # Verify model is available
        available_models = self.client.get_models()
        if model_name not in available_models:
            print(f"Warning: {model_name} not found in available models")
            print(f"Available models: {available_models}")
            # Try to find a vision model
            vision_models = [
                m
                for m in available_models
                if "pixtral" in m.lower() or "vision" in m.lower()
            ]
            if vision_models:
                self.model_name = vision_models[0]
                print(f"Using {self.model_name} instead")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def evaluate_edit(
        self, original_path: str, edited_path: str, instruction: str
    ) -> Dict[str, any]:
        """
        Evaluate image edit using VIEScore methodology via OpenWebUI

        Args:
            original_path: Path to original image
            edited_path: Path to edited image
            instruction: Edit instruction text

        Returns:
            Dictionary with evaluation scores and rationale
        """
        # Encode images
        original_b64 = self.encode_image(original_path)
        edited_b64 = self.encode_image(edited_path)

        # Construct prompt for VIEScore evaluation
        prompt = f"""You are evaluating an image editing result. 
        
Editing Instruction: "{instruction}"

Please evaluate the edited image based on:
1. Semantic Consistency (SC): How well does the edit follow the instruction? (0-10)
2. Perceptual Quality (PQ): How good is the visual quality of the edited image? (0-10)
3. Overall Score (O): Overall quality considering both SC and PQ (0-10)

Provide your evaluation in JSON format:
{{
    "semantic_consistency": <score>,
    "perceptual_quality": <score>,
    "overall": <score>,
    "rationale": "<brief explanation of scores>"
}}"""

        # Construct messages with both images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Original image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{original_b64}"},
                    },
                    {"type": "text", "text": "Edited image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{edited_b64}"},
                    },
                ],
            }
        ]

        # Make the API call
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_name,
            temperature=0.1,  # Low temperature for consistent scoring
        )

        # Parse response
        try:
            # Try to extract JSON from the response
            import re

            json_match = re.search(
                r'\{[^{}]*"semantic_consistency"[^{}]*\}', response, re.DOTALL
            )
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
        except:
            # Fallback if JSON parsing fails
            result = {
                "semantic_consistency": 0,
                "perceptual_quality": 0,
                "overall": 0,
                "rationale": "Failed to parse evaluation",
                "raw_response": response,
            }

        return result


# Import shared LoRA utilities
sys.path.append('..')
from lora_utils import LocalQwenImageEdit



class VIEScoreTrainingHook:
    """Hook to run VIEScore evaluation during training"""

    def __init__(self, evaluator: 'VIEScoreViaOpenWebUI', validation_images: List[Dict[str, any]],
                 eval_frequency: int = 1000, max_eval_samples: int = 5):
        """
        Initialize training hook for VIEScore evaluation

        Args:
            evaluator: VIEScore evaluator instance
            validation_images: List of validation image dicts with 'path' and 'instruction'
            eval_frequency: Evaluate every N steps
            max_eval_samples: Maximum samples to evaluate per checkpoint
        """
        self.evaluator = evaluator
        self.validation_images = validation_images
        self.eval_frequency = eval_frequency
        self.max_eval_samples = max_eval_samples
        self.step_count = 0
        self.scores_history = []

    def on_step_end(self, model_path: Optional[str] = None, step: Optional[int] = None):
        """Called at the end of each training step"""
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1

        if self.step_count % self.eval_frequency == 0:
            checkpoint_path = model_path if model_path is not None else f"step_{self.step_count}"
            self.evaluate_checkpoint(checkpoint_path)

    def evaluate_checkpoint(self, checkpoint_path: str):
        """Evaluate a model checkpoint using VIEScore"""
        print(f"\n🔍 Running VIEScore evaluation for checkpoint: {checkpoint_path}")

        # Sample validation images
        eval_samples = self.validation_images[:self.max_eval_samples]

        scores = []
        temp_dir = Path(f"temp_eval_{self.step_count}")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Initialize local model for this checkpoint
            if os.path.exists(checkpoint_path):
                local_editor = LocalQwenImageEdit(checkpoint_path)
            else:
                print(f"Checkpoint {checkpoint_path} not found, using base model")
                local_editor = LocalQwenImageEdit("")

            for i, sample in enumerate(eval_samples):
                original_path = sample['path']
                instruction = sample['instruction']
                edited_path = temp_dir / f"edited_{i}.png"

                try:
                    # Edit image with current model
                    local_editor.edit_image(original_path, instruction, str(edited_path))

                    # Evaluate with VIEScore
                    score = self.evaluator.evaluate_edit(original_path, str(edited_path), instruction)
                    score['step'] = self.step_count
                    score['checkpoint'] = checkpoint_path
                    scores.append(score)

                except Exception as e:
                    print(f"Error evaluating sample {i}: {e}")
                    continue

            # Calculate average scores
            if scores:
                avg_sc = sum(s.get('semantic_consistency', 0) for s in scores) / len(scores)
                avg_pq = sum(s.get('perceptual_quality', 0) for s in scores) / len(scores)
                avg_overall = sum(s.get('overall', 0) for s in scores) / len(scores)

                result = {
                    'step': self.step_count,
                    'checkpoint': checkpoint_path,
                    'avg_semantic_consistency': avg_sc,
                    'avg_perceptual_quality': avg_pq,
                    'avg_overall': avg_overall,
                    'num_samples': len(scores)
                }

                self.scores_history.append(result)

                print(f"📊 VIEScore Results (Step {self.step_count}):")
                print(f"   Semantic Consistency: {avg_sc:.2f}/10")
                print(f"   Perceptual Quality: {avg_pq:.2f}/10")
                print(f"   Overall Score: {avg_overall:.2f}/10")

                # Save results
                results_df = pd.DataFrame(self.scores_history)
                results_df.to_csv("viescore_training_history.csv", index=False)

        finally:
            # Cleanup temp files
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def get_latest_scores(self) -> Optional[Dict[str, any]]:
        """Get the latest VIEScore results"""
        return self.scores_history[-1] if self.scores_history else None


def run_openwebui_evaluation_pipeline(
    input_dir: str = "extracted_images",
    instructions_file: str = "edit_instructions.csv",
    output_dir: str = "openwebui_evaluation_results",
    use_viescore: bool = True,
    sample_size: Optional[int] = None,
    base_url: str = "https://chat.hpi-sci.de",
    qwen_model: str = "edit_image.qwen-image-edit",
    pixtral_model: str = "mistralai/Pixtral-12B-2409",
):
    """
    Run the complete evaluation pipeline using OpenWebUI

    Args:
        input_dir: Directory containing original images
        instructions_file: CSV file with row numbers and instructions
        output_dir: Directory to save results
        use_viescore: Whether to use VIEScore evaluation
        sample_size: Number of samples to process (None for all)
        base_url: OpenWebUI instance URL
        qwen_model: Name of Qwen-Image-Edit model in OpenWebUI
        pixtral_model: Name of Pixtral model for VIEScore
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    edited_dir = os.path.join(output_dir, "edited_images")
    os.makedirs(edited_dir, exist_ok=True)

    # Initialize OpenWebUI client
    client = OpenWebUIClient(base_url=base_url)

    # List available models
    print("\nAvailable models in OpenWebUI:")
    models = client.get_models()
    for model in models:
        print(f"  - {model}")

    # Load instructions
    df = pd.read_csv(instructions_file)

    # Filter out empty instructions
    df = df[df["edit_instruction"].notna()]
    df = df[df["edit_instruction"] != ""]

    if sample_size:
        df = df.head(sample_size)

    print(f"\nProcessing {len(df)} images with edit instructions...")

    # Initialize models
    editor = QwenImageEditViaOpenWebUI(client, model_name=qwen_model)

    evaluator = None
    if use_viescore:
        evaluator = VIEScoreViaOpenWebUI(client, model_name=pixtral_model)

    results = []

    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        row_num = row["row_number"]
        instruction = row["edit_instruction"]

        # Paths
        original_path = os.path.join(input_dir, f"{row_num}.png")
        edited_path = os.path.join(edited_dir, f"{row_num}_edited.png")

        if not os.path.exists(original_path):
            print(f"Warning: Image {original_path} not found, skipping...")
            continue

        try:
            # Edit image
            print(f"\nEditing image {row_num}...")
            editor.edit_image(original_path, instruction, edited_path)

            result = {
                "row_number": row_num,
                "instruction": instruction,
                "original_path": original_path,
                "edited_path": edited_path,
            }

            # Evaluate if VIEScore is enabled
            if use_viescore and evaluator:
                print(f"Evaluating edit for image {row_num}...")
                scores = evaluator.evaluate_edit(
                    original_path, edited_path, instruction
                )
                result.update(scores)

            results.append(result)

        except Exception as e:
            print(f"Error processing row {row_num}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_csv_path}")
    print(f"Edited images saved to: {edited_dir}")

    if use_viescore and evaluator and "semantic_consistency" in results_df.columns:
        # Calculate average scores
        avg_sc = results_df["semantic_consistency"].mean()
        avg_pq = results_df["perceptual_quality"].mean()
        avg_overall = results_df["overall"].mean()

        print(f"\nAverage VIEScore Results:")
        print(f"  Semantic Consistency: {avg_sc:.2f}/10")
        print(f"  Perceptual Quality: {avg_pq:.2f}/10")
        print(f"  Overall Score: {avg_overall:.2f}/10")

    return results_df


def run_local_model_evaluation_pipeline(
    model_path: str,
    input_dir: str = "extracted_images",
    instructions_file: str = "edit_instructions.csv",
    output_dir: str = "local_model_evaluation_results",
    use_viescore: bool = True,
    sample_size: Optional[int] = None,
    base_url: str = "https://chat.hpi-sci.de",
    pixtral_model: str = "mistralai/Pixtral-12B-2409",
    device: str = "cuda",
    lora_name: str = "",
    resize: bool = True
):
    """
    Run evaluation pipeline using a local trained model

    Args:
        model_path: Path to trained model checkpoint/LoRA
        input_dir: Directory containing original images
        instructions_file: CSV file with row numbers and instructions
        output_dir: Directory to save results
        use_viescore: Whether to use VIEScore evaluation
        sample_size: Number of samples to process (None for all)
        base_url: OpenWebUI instance URL for VIEScore evaluation
        pixtral_model: Name of Pixtral model for VIEScore
        device: Device to run inference on
        lora_name: Name of the LoRA model (used in output filenames)
    """
    # Create LoRA suffix for filenames
    lora_suffix = f"_{lora_name}" if lora_name else ""

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    edited_dir = os.path.join(output_dir, f"edited_images{lora_suffix}")
    os.makedirs(edited_dir, exist_ok=True)

    # Load instructions
    df = pd.read_csv(instructions_file)

    # Filter out empty instructions
    df = df[df["edit_instruction"].notna()]
    df = df[df["edit_instruction"] != ""]

    if sample_size:
        df = df.head(sample_size)

    print(f"\nProcessing {len(df)} images with local model: {model_path}")

    # Initialize local model
    try:
        editor = LocalQwenImageEdit(model_path, device=device, lora_name=lora_name, resize=resize)
        print("✓ Local model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load local model: {e}")
        return None

    # Initialize VIEScore evaluator if requested
    evaluator = None
    if use_viescore:
        try:
            client = OpenWebUIClient(base_url=base_url)
            evaluator = VIEScoreViaOpenWebUI(client, model_name=pixtral_model)
            print("✓ VIEScore evaluator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize VIEScore evaluator: {e}")
            use_viescore = False

    results = []

    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        row_num = row["row_number"]
        instruction = row["edit_instruction"]

        # Paths
        original_path = os.path.join(input_dir, f"{row_num}.png")
        edited_path = os.path.join(edited_dir, f"{row_num}_edited{lora_suffix}.png")

        if not os.path.exists(original_path):
            print(f"Warning: Image {original_path} not found, skipping...")
            continue

        try:
            # Edit image with local model
            print(f"\nEditing image {row_num} with local model...")
            editor.edit_image(original_path, instruction, edited_path, seed=0)

            result = {
                "row_number": row_num,
                "instruction": instruction,
                "original_path": original_path,
                "edited_path": edited_path,
                "model_path": model_path,
                "lora_name": lora_name
            }

            # Evaluate if VIEScore is enabled
            if use_viescore and evaluator:
                print(f"Evaluating edit for image {row_num}...")
                scores = evaluator.evaluate_edit(
                    original_path, edited_path, instruction
                )
                result.update(scores)

            results.append(result)

        except Exception as e:
            print(f"Error processing row {row_num}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    csv_filename = f"evaluation_results{lora_suffix}.csv"
    results_csv_path = os.path.join(output_dir, csv_filename)
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_csv_path}")
    print(f"Edited images saved to: {edited_dir}")

    if use_viescore and evaluator and "semantic_consistency" in results_df.columns:
        # Calculate average scores
        avg_sc = results_df["semantic_consistency"].mean()
        avg_pq = results_df["perceptual_quality"].mean()
        avg_overall = results_df["overall"].mean()

        print(f"\nAverage VIEScore Results for {model_path}:")
        print(f"  Semantic Consistency: {avg_sc:.2f}/10")
        print(f"  Perceptual Quality: {avg_pq:.2f}/10")
        print(f"  Overall Score: {avg_overall:.2f}/10")

    return results_df


def setup_viescore_training_hook(
    validation_images_dir: str = "extracted_images",
    validation_instructions_file: str = "edit_instructions.csv",
    eval_frequency: int = 1000,
    max_eval_samples: int = 5,
    base_url: str = "https://chat.hpi-sci.de",
    pixtral_model: str = "mistralai/Pixtral-12B-2409"
) -> VIEScoreTrainingHook:
    """
    Setup VIEScore evaluation hook for training

    Args:
        validation_images_dir: Directory with validation images
        validation_instructions_file: CSV with validation instructions
        eval_frequency: Evaluate every N steps
        max_eval_samples: Max samples to evaluate per checkpoint
        base_url: OpenWebUI instance URL
        pixtral_model: Pixtral model name for VIEScore

    Returns:
        VIEScoreTrainingHook instance
    """
    # Load validation data
    df = pd.read_csv(validation_instructions_file)
    df = df[df["edit_instruction"].notna()]
    df = df[df["edit_instruction"] != ""]

    validation_images = []
    for _, row in df.iterrows():
        image_path = os.path.join(validation_images_dir, f"{row['row_number']}.png")
        if os.path.exists(image_path):
            validation_images.append({
                'path': image_path,
                'instruction': row['edit_instruction'],
                'row_number': row['row_number']
            })

    # Initialize evaluator
    client = OpenWebUIClient(base_url=base_url)
    evaluator = VIEScoreViaOpenWebUI(client, model_name=pixtral_model)

    # Create hook
    hook = VIEScoreTrainingHook(
        evaluator=evaluator,
        validation_images=validation_images,
        eval_frequency=eval_frequency,
        max_eval_samples=max_eval_samples
    )

    print(f"✓ VIEScore training hook setup with {len(validation_images)} validation images")
    print(f"  Will evaluate every {eval_frequency} steps using {max_eval_samples} samples")

    return hook


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VIEScore Evaluation Pipeline")
    parser.add_argument("--mode", choices=["openwebui", "local"], default="openwebui",
                       help="Evaluation mode: openwebui or local model")
    parser.add_argument("--model_path", type=str, default="",
                       help="Path to local model checkpoint (for local mode)")
    parser.add_argument("--lora_name", type=str, default="",
                       help="Name of the LoRA model (used in output filenames)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of samples to process (None for all)")
    parser.add_argument("--use_viescore", action="store_true", default=True,
                       help="Enable VIEScore evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for local model inference")
    parser.add_argument("--no_resize", action="store_true", default=False,
                       help="Disable automatic image resizing to 1024x1024")

    args = parser.parse_args()

    print("=" * 60)
    print("VIEScore Evaluation Pipeline")
    print("=" * 60)

    results = None

    if args.mode == "openwebui":
        print("Running OpenWebUI evaluation mode...")

        # Test connection first
        try:
            client = OpenWebUIClient()
            print("✓ Successfully connected to OpenWebUI")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("Make sure your API_KEY is set in .env file")
            exit(1)

        # Run OpenWebUI evaluation
        results = run_openwebui_evaluation_pipeline(
            sample_size=args.sample_size,
            use_viescore=args.use_viescore,
            qwen_model="edit_image.qwen-image-edit",
            pixtral_model="mistralai/Pixtral-12B-2409",
        )

    elif args.mode == "local":
        print(f"Running local model evaluation mode...")

        if not args.model_path:
            print("Warning: No model path provided, using base model")

        # Run local model evaluation
        results = run_local_model_evaluation_pipeline(
            model_path=args.model_path,
            sample_size=args.sample_size,
            use_viescore=args.use_viescore,
            device=args.device,
            lora_name=args.lora_name,
            resize=not args.no_resize
        )

    print(f"\n🎉 Evaluation completed! Processed {len(results) if results is not None else 0} samples.")
