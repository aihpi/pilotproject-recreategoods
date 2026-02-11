#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    mask = Image.open(path).convert("L")
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    return (mask_np > 0.5).astype(np.float32)


def _to_numpy(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _apply_background_only(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return edited
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = edited.copy()
    out[mask_3 > 0.5] = original[mask_3 > 0.5]
    return out


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _ssim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    return float(ssim(a, b, channel_axis=-1, data_range=1.0))


def _lpips(a: np.ndarray, b: np.ndarray, device: str) -> Optional[float]:
    try:
        import torch
        import lpips
    except Exception:
        return None
    a_t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    b_t = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device)
    loss_fn = lpips.LPIPS(net="alex").to(device)
    with torch.no_grad():
        score = loss_fn(a_t, b_t).item()
    return float(score)


def _clip_similarity(img: Image.Image, text: str, device: str, clip_model: str) -> Optional[float]:
    try:
        import torch
        import open_clip
    except Exception:
        return None
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(clip_model)
    model = model.to(device)
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    text_tensor = tokenizer([text]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = (image_features * text_features).sum().item()
    return float(score)


def _change_map_iou(original: np.ndarray, edited: np.ndarray, mask: np.ndarray, diff_threshold: float) -> Optional[float]:
    if mask is None:
        return None
    diff = np.mean(np.abs(original - edited), axis=2)
    change = (diff > diff_threshold).astype(np.float32)
    mask_bin = (mask > 0.5).astype(np.float32)
    intersection = np.sum(change * mask_bin)
    union = np.sum((change + mask_bin) > 0)
    if union == 0:
        return 0.0
    return float(intersection / union)


def _resolve_output_path(outputs_dir: Path, image_path: Path) -> Path:
    return outputs_dir / image_path.name


def evaluate(
    metadata_path: Path,
    dataset_root: Path,
    outputs_dir: Path,
    mask_dir: Optional[Path],
    mask_suffix: str,
    diff_threshold: float,
    device: str,
    clip_model: Optional[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with metadata_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rel_image_path = row["image"]
            prompt = row.get("prompt", "")
            image_path = (dataset_root / rel_image_path).resolve()
            output_path = _resolve_output_path(outputs_dir, Path(rel_image_path))

            if not image_path.exists() or not output_path.exists():
                rows.append({
                    "image": rel_image_path,
                    "status": "missing_input_or_output",
                })
                continue

            original_img = _load_image(image_path)
            edited_img = _load_image(output_path)
            original_np = _to_numpy(original_img)
            edited_np = _to_numpy(edited_img)

            mask_np = None
            if mask_dir:
                stem = Path(rel_image_path).stem
                mask_name = f"{stem}{mask_suffix}"
                mask_path = mask_dir / mask_name
                mask_np = _load_mask(mask_path)

            edited_bg = _apply_background_only(original_np, edited_np, mask_np)

            result: Dict[str, object] = {
                "image": rel_image_path,
                "output": str(output_path),
            }

            result["psnr_bg"] = _psnr(original_np, edited_bg)
            ssim_val = _ssim(original_np, edited_bg)
            result["ssim_bg"] = ssim_val if ssim_val is not None else "skip"

            lpips_val = _lpips(original_np, edited_bg, device)
            result["lpips_bg"] = lpips_val if lpips_val is not None else "skip"

            iou_val = _change_map_iou(original_np, edited_np, mask_np, diff_threshold)
            result["change_iou"] = iou_val if iou_val is not None else "skip"

            if clip_model:
                clip_val = _clip_similarity(edited_img, prompt, device, clip_model)
                result["clip_similarity"] = clip_val if clip_val is not None else "skip"
            else:
                result["clip_similarity"] = "skip"

            rows.append(result)
    return rows


def _write_csv(out_csv: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate edit quality and preservation metrics.")
    parser.add_argument("--metadata", type=Path, default=Path("model-testing/test-data/metadata_edit.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("model-testing/test-data"))
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--mask-suffix", type=str, default="_mask.png")
    parser.add_argument("--diff-threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--out-csv", type=Path, default=Path("model-testing/eval_results.csv"))
    args = parser.parse_args()

    rows = evaluate(
        metadata_path=args.metadata,
        dataset_root=args.dataset_root,
        outputs_dir=args.outputs_dir,
        mask_dir=args.mask_dir,
        mask_suffix=args.mask_suffix,
        diff_threshold=args.diff_threshold,
        device=args.device,
        clip_model=args.clip_model,
    )
    _write_csv(args.out_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
