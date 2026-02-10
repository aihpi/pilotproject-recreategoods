'''Updated utility functions for loading segmentation masks from multiple formats.

This extends the original segmentation_utils.py to handle both:
1. COCO-style RLE format: {"predictions": [{"segmentation": {"size": [h,w], "counts": "..."}}]}
2. Direct binary mask format: {"mask": [[...]]}

Both formats are converted to torch.Tensor of shape (1, H, W) with float32 values in [0, 1].
'''

import json
import os
from typing import Tuple, Optional

import torch
import numpy as np

# Optional import – the repository may not have pycocotools installed.
try:
    from pycocotools import mask as maskUtils  # type: ignore
except Exception:  # pragma: no cover
    maskUtils = None


def _decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE mask to a NumPy ``uint8`` array."""
    if maskUtils is not None:
        return maskUtils.decode(rle)

    # ---- Simple pure‑Python decoder (fallback) ----
    height, width = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = [int(x) for x in counts.split()]
    
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if c == 0:
            continue
        if val == 1:
            flat[idx : idx + c] = 1
        idx += c
        val = 1 - val  # toggle between 0 and 1
    return flat.reshape((height, width))


def load_segmentation_mask(
    json_path: str,
    *,
    mask_index: int = 0,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Load a binary segmentation mask from JSON file (supports multiple formats).

    Args:
        json_path: Path to JSON file
        mask_index: Which prediction to use (for RLE format)
        target_size: Desired (height, width) of output mask
        
    Returns:
        torch.Tensor: Shape (1, H, W) with float32 values in [0, 1]
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Segmentation JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle direct binary mask format
    if "mask" in data:
        mask_np = np.array(data["mask"], dtype=np.uint8)
    # Handle COCO RLE format
    elif "predictions" in data:
        predictions = data["predictions"]
        if not predictions:
            raise ValueError(f"No predictions found in {json_path}")
        if mask_index >= len(predictions):
            raise IndexError(f"mask_index {mask_index} out of range (found {len(predictions)} predictions)")
        
        seg = predictions[mask_index].get("segmentation")
        if seg is None:
            raise KeyError("'segmentation' field missing in selected prediction")
        
        mask_np = _decode_rle(seg)
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}. Expected 'mask' or 'predictions' key.")

    # Convert to torch tensor, add channel dimension, and cast to float.
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

    if target_size is not None:
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),  # (1, 1, H, W)
            size=target_size,
            mode="nearest",
        ).squeeze(0)
    return mask_tensor