#!/usr/bin/env python3
"""
Test script for the enhanced VIEScore functionality

This script tests:
1. Loading a local Qwen-Image-Edit model (with or without LoRA)
2. Running inference on a sample image
3. VIEScore evaluation of the result
"""

import os
import sys
from pathlib import Path

# Add the viescore module to path
viescore_path = Path(__file__).parent
if str(viescore_path) not in sys.path:
    sys.path.insert(0, str(viescore_path))

from script import LocalQwenImageEdit, VIEScoreViaOpenWebUI, OpenWebUIClient


def test_local_model_loading(model_path: str = ""):
    """Test loading a local Qwen-Image-Edit model"""
    print("=" * 50)
    print("Testing Local Model Loading")
    print("=" * 50)

    try:
        editor = LocalQwenImageEdit(model_path, device="cuda")
        print("✓ Model loaded successfully!")
        return editor
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None


def test_image_editing(editor, test_image_path: str, instruction: str, output_path: str):
    """Test image editing with the local model"""
    print("\n" + "=" * 50)
    print("Testing Image Editing")
    print("=" * 50)

    if editor is None:
        print("✗ No model available for testing")
        return None

    if not os.path.exists(test_image_path):
        print(f"✗ Test image not found: {test_image_path}")
        return None

    try:
        result = editor.edit_image(test_image_path, instruction, output_path)
        print(f"✓ Image editing completed: {result}")
        return result
    except Exception as e:
        print(f"✗ Image editing failed: {e}")
        return None


def test_viescore_evaluation(original_path: str, edited_path: str, instruction: str):
    """Test VIEScore evaluation"""
    print("\n" + "=" * 50)
    print("Testing VIEScore Evaluation")
    print("=" * 50)

    if not os.path.exists(original_path) or not os.path.exists(edited_path):
        print("✗ Required images not found for VIEScore evaluation")
        return None

    try:
        # Initialize VIEScore evaluator
        client = OpenWebUIClient()
        evaluator = VIEScoreViaOpenWebUI(client)

        # Run evaluation
        scores = evaluator.evaluate_edit(original_path, edited_path, instruction)

        print("✓ VIEScore evaluation completed!")
        print(f"   Semantic Consistency: {scores.get('semantic_consistency', 'N/A')}")
        print(f"   Perceptual Quality: {scores.get('perceptual_quality', 'N/A')}")
        print(f"   Overall Score: {scores.get('overall', 'N/A')}")
        print(f"   Rationale: {scores.get('rationale', 'N/A')}")

        return scores
    except Exception as e:
        print(f"✗ VIEScore evaluation failed: {e}")
        return None


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Enhanced VIEScore Functionality")
    parser.add_argument("--model_path", type=str, default="",
                       help="Path to LoRA checkpoint (empty for base model)")
    parser.add_argument("--test_image", type=str, default="../extracted_images/1.png",
                       help="Path to test image")
    parser.add_argument("--instruction", type=str, default="Make the image brighter",
                       help="Edit instruction for testing")
    parser.add_argument("--output_path", type=str, default="test_edited.png",
                       help="Output path for edited image")
    parser.add_argument("--skip_viescore", action="store_true",
                       help="Skip VIEScore evaluation (useful if OpenWebUI not available)")

    args = parser.parse_args()

    print("🧪 Enhanced VIEScore Functionality Test")
    print(f"Model path: {args.model_path or 'Base model'}")
    print(f"Test image: {args.test_image}")
    print(f"Instruction: {args.instruction}")

    # Test 1: Load model
    editor = test_local_model_loading(args.model_path)

    # Test 2: Edit image
    edited_path = test_image_editing(editor, args.test_image, args.instruction, args.output_path)

    # Test 3: VIEScore evaluation (if enabled)
    if not args.skip_viescore and edited_path:
        scores = test_viescore_evaluation(args.test_image, edited_path, args.instruction)
    else:
        if args.skip_viescore:
            print("\n⏭️  Skipping VIEScore evaluation as requested")
        scores = None

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"✓ Model loading: {'SUCCESS' if editor else 'FAILED'}")
    print(f"✓ Image editing: {'SUCCESS' if edited_path else 'FAILED'}")
    print(f"✓ VIEScore eval: {'SUCCESS' if scores else ('SKIPPED' if args.skip_viescore else 'FAILED')}")

    if edited_path:
        print(f"\n📁 Output files:")
        print(f"   Edited image: {edited_path}")
        if os.path.exists(edited_path.replace('.png', '_debug.txt')):
            print(f"   Debug info: {edited_path.replace('.png', '_debug.txt')}")


if __name__ == "__main__":
    main()