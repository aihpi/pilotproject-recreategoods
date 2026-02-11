#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = REPO_ROOT / "qwen-image-edit-finetune"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(QWEN_ROOT))

from lora_utils import LocalQwenImageEdit, convert_diffsynth_lora_to_diffusers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen-Image-Edit inference over a test set.")
    default_metadata = Path("model-testing/test-data/metadata_edit.csv")
    default_dataset_root = Path("model-testing/test-data")
    if Path.cwd().name == "model-testing":
        default_metadata = Path("test-data/metadata_edit.csv")
        default_dataset_root = Path("test-data")
    parser.add_argument("--metadata", type=Path, default=default_metadata)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, default=Path(""))
    parser.add_argument("--lora-name", type=str, default="")
    parser.add_argument("--base-model-id", type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", action="store_true", default=True)
    parser.add_argument("--no-resize", dest="resize", action="store_false")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    lora_name = args.lora_name
    if args.lora_path and lora_name:
        lora_path = Path(args.lora_path)
        if not lora_name.endswith("_diffusers.safetensors"):
            converted_name = f"{Path(lora_name).stem}_diffusers.safetensors"
            converted_path = lora_path / converted_name
            if not converted_path.exists():
                convert_diffsynth_lora_to_diffusers(str(lora_path / lora_name))
            lora_name = converted_name

    editor = LocalQwenImageEdit(
        model_path=str(args.lora_path) if args.lora_path else "",
        device=args.device,
        base_model_id=args.base_model_id,
        lora_name=lora_name,
        resize=args.resize,
    )

    with args.metadata.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rel_image_path = row["image"]
            instruction = row.get("prompt", "")
            input_path = (args.dataset_root / rel_image_path).resolve()
            output_path = args.outputs_dir / Path(rel_image_path).name

            if not input_path.exists():
                print(f"Missing input image: {input_path}")
                continue

            editor.edit_image(
                image_path=str(input_path),
                instruction=instruction,
                output_path=str(output_path),
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.cfg_scale,
                seed=args.seed,
            )

    print(f"Done. Outputs in {args.outputs_dir}")


if __name__ == "__main__":
    main()
