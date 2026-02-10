#!/usr/bin/env python3
"""
Batch LoRA Weight Converter
Converts DiffSynth-Studio LoRA weights to diffusers format for all checkpoints
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from lora_utils import convert_diffsynth_lora_to_diffusers


def convert_single_checkpoint(checkpoint_path: str, force: bool = False) -> bool:
    """
    Convert a single LoRA checkpoint from DiffSynth-Studio to diffusers format

    Args:
        checkpoint_path: Path to the checkpoint file
        force: Force conversion even if output already exists

    Returns:
        True if conversion was successful or skipped, False if failed
    """
    try:
        # Check if already converted
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_dir = os.path.dirname(checkpoint_path)
        expected_output = os.path.join(output_dir, f"{base_name}_diffusers.safetensors")

        if os.path.exists(expected_output) and not force:
            print(f"⏭️  Skipping {Path(checkpoint_path).name} (already converted)")
            return True

        print(f"🔄 Converting {Path(checkpoint_path).name}")
        converted_path = convert_diffsynth_lora_to_diffusers(checkpoint_path)

        if os.path.exists(converted_path):
            size_mb = os.path.getsize(converted_path) / (1024*1024)
            print(f"✅ Converted: {Path(converted_path).name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"❌ Conversion failed: output file not created")
            return False

    except Exception as e:
        print(f"❌ Error converting {Path(checkpoint_path).name}: {e}")
        return False


def batch_convert_checkpoints(lora_dir: str, pattern: str = "*.safetensors", force: bool = False) -> tuple:
    """
    Batch convert all LoRA checkpoints in a directory

    Args:
        lora_dir: Directory containing LoRA checkpoints
        pattern: File pattern to match (default: "*.safetensors")
        force: Force conversion even if outputs already exist

    Returns:
        Tuple of (successful_conversions, failed_conversions)
    """
    print(f"🔍 Searching for LoRA checkpoints in: {lora_dir}")
    print(f"   Pattern: {pattern}")

    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(lora_dir, pattern))

    # Filter out already converted files
    original_checkpoints = [cp for cp in checkpoint_files if '_diffusers' not in Path(cp).stem]

    if not original_checkpoints:
        print("❌ No original LoRA checkpoints found!")
        return 0, 0

    print(f"📦 Found {len(original_checkpoints)} LoRA checkpoints to process")

    # Sort by step number if they follow step-XXXXX pattern
    try:
        original_checkpoints = sorted(original_checkpoints,
                                    key=lambda x: int(Path(x).stem.split('-')[1]) if '-' in Path(x).stem else 0)
    except:
        # If sorting fails, just use alphabetical order
        original_checkpoints = sorted(original_checkpoints)

    successful = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"Starting batch conversion...")
    print(f"{'='*60}")

    # Process each checkpoint with progress bar
    with tqdm(original_checkpoints, desc="Converting LoRA checkpoints", unit="file") as pbar:
        for checkpoint_path in pbar:
            pbar.set_postfix(file=Path(checkpoint_path).name)

            if convert_single_checkpoint(checkpoint_path, force=force):
                successful += 1
            else:
                failed += 1

            pbar.set_postfix(success=successful, failed=failed)

    return successful, failed


def main():
    """Main function for batch LoRA conversion"""
    parser = argparse.ArgumentParser(description="Batch convert DiffSynth-Studio LoRA weights to diffusers format")
    parser.add_argument("--lora_dir", type=str, required=True,
                       help="Directory containing LoRA checkpoint files")
    parser.add_argument("--pattern", type=str, default="*.safetensors",
                       help="File pattern to match (default: *.safetensors)")
    parser.add_argument("--force", action="store_true",
                       help="Force conversion even if output files already exist")
    parser.add_argument("--single", type=str, default=None,
                       help="Convert a single checkpoint file instead of batch")

    args = parser.parse_args()

    print("🚀 LoRA Weight Batch Converter")
    print("=" * 50)

    if args.single:
        # Convert single file
        if not os.path.exists(args.single):
            print(f"❌ Single file not found: {args.single}")
            sys.exit(1)

        print(f"Converting single file: {args.single}")
        success = convert_single_checkpoint(args.single, force=args.force)

        if success:
            print("✅ Single file conversion completed!")
            sys.exit(0)
        else:
            print("❌ Single file conversion failed!")
            sys.exit(1)
    else:
        # Batch convert directory
        if not os.path.exists(args.lora_dir):
            print(f"❌ LoRA directory not found: {args.lora_dir}")
            sys.exit(1)

        successful, failed = batch_convert_checkpoints(args.lora_dir, args.pattern, args.force)

        print(f"\n{'='*60}")
        print(f"📊 Batch Conversion Results:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "   📈 Success Rate: 0%")
        print(f"{'='*60}")

        if failed > 0:
            print("⚠️  Some conversions failed. Check the output above for details.")
            sys.exit(1)
        else:
            print("🎉 All conversions completed successfully!")
            sys.exit(0)


if __name__ == "__main__":
    main()