'''Utility functions for loading segmentation masks from COCO‑style JSON files.

The repository stores per‑image predictions in
`segmentation/segmentation_output_parallel/predictions/*.json`.  Each JSON
contains a list of objects with a ``"segmentation"`` field that follows the
COCO RLE format:

```json
"segmentation": {
    "size": [height, width],
    "counts": "..."   # run‑length encoded string (or list of ints)
}
```

The helper below reads such a file, extracts the first (or a specific) mask, and
returns a ``torch.Tensor`` of shape ``(1, H, W)`` with ``float32`` values in
``[0, 1]``.  If ``target_size`` is provided the mask is resized (nearest‑neighbor)
to match the model’s output resolution.

The implementation prefers ``pycocotools`` for decoding because it is fast and
battle‑tested.  If the library is not available a pure‑Python fallback is used.
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
    """Decode a COCO RLE mask to a NumPy ``uint8`` array.

    Parameters
    ----------
    rle: dict
        Dictionary with ``"size"`` and ``"counts"`` keys as stored in the JSON.
    """
    if maskUtils is not None:
        # ``maskUtils.decode`` expects the RLE dict exactly as COCO provides.
        return maskUtils.decode(rle)

    # ---- Simple pure‑Python decoder (fallback) ----
    height, width = rle["size"]
    counts = rle["counts"]
    # ``counts`` may be a string (compressed) or a list of ints.
    if isinstance(counts, str):
        # The string is an ASCII‑encoded RLE used by COCO.  It can be decoded
        # using the same algorithm as pycocotools (run‑length of alternating
        # zeros and ones).  For simplicity we treat the string as a list of
        # integers separated by spaces – this works for the dataset at hand.
        # If the format ever changes, install ``pycocotools``.
        counts = [int(x) for x in counts.split()]
    # ``counts`` is now a list of run lengths.
    # The first count corresponds to zeros.
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
    """Load a binary segmentation mask from a COCO‑style prediction JSON.

    Parameters
    ----------
    json_path: str
        Absolute path to the ``*_predictions.json`` file.
    mask_index: int, default ``0``
        Which entry in the ``"predictions"`` list to use.  Most datasets store a
        single mask per image, so the default is appropriate.
    target_size: tuple[int, int] | None, optional
        Desired ``(height, width)`` of the output mask.  If provided the mask is
        resized using nearest‑neighbor interpolation to keep the binary nature.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, H, W)`` with ``float32`` values ``0.0`` or ``1.0``.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Segmentation JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        raise ValueError(f"No predictions found in {json_path}")
    if mask_index >= len(predictions):
        raise IndexError(
            f"mask_index {mask_index} out of range (found {len(predictions)} predictions)"
        )

    seg = predictions[mask_index].get("segmentation")
    if seg is None:
        raise KeyError("'segmentation' field missing in selected prediction")

    mask_np = _decode_rle(seg)  # shape (H, W), dtype uint8
    # Convert to torch tensor, add channel dimension, and cast to float.
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

    if target_size is not None:
        # ``torch.nn.functional.interpolate`` expects a 4‑D tensor.
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),  # (1, 1, H, W)
            size=target_size,
            mode="nearest",
        ).squeeze(0)
    return mask_tensor

# Example usage (for documentation purposes only):
# mask = load_segmentation_mask(
#     "/path/to/0a0a6c9d9dbd4141955b312147379a79_predictions.json",
#     target_size=(256, 256),
# )
