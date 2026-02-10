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
from typing import Tuple, Optional, Dict, Any, List

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


def filter_predictions_by_similarity(
    predictions_data: Dict[str, Any],
    similarity_threshold: float = 0.5,
    enable_dynamic_filtering: bool = True
) -> Dict[str, Any]:
    """Filter predictions based on similarity scores using dynamic threshold logic.
    
    Parameters
    ----------
    predictions_data : Dict[str, Any]
        The full predictions data containing 'predictions' and 'similarity_analysis'
    similarity_threshold : float, default 0.5
        The threshold for high-confidence filtering
    enable_dynamic_filtering : bool, default True
        Whether to use dynamic threshold logic or simple threshold filtering
    
    Returns
    -------
    Dict[str, Any]
        Filtered predictions data with only relevant predictions retained
    """
    if not enable_dynamic_filtering:
        # Simple threshold filtering - keep all predictions above threshold
        return predictions_data
    
    if 'similarity_analysis' not in predictions_data:
        # No similarity analysis - return all predictions
        return predictions_data
    
    similarity_data = predictions_data['similarity_analysis']
    category_similarities = similarity_data.get('category_similarities', {})
    
    if not category_similarities:
        # No similarity scores - return all predictions
        return predictions_data
    
    max_similarity = max(category_similarities.values())
    
    # Apply dynamic threshold logic
    if max_similarity >= similarity_threshold:
        # High confidence: keep all categories above threshold
        filtered_categories = {
            k: v for k, v in category_similarities.items()
            if v >= similarity_threshold
        }
    else:
        # Lower confidence: keep only the top 1 most relevant category
        sorted_items = sorted(category_similarities.items(), key=lambda x: x[1], reverse=True)
        filtered_categories = {sorted_items[0][0]: sorted_items[0][1]}
    
    # Filter predictions to only include relevant categories
    filtered_predictions = []
    for prediction in predictions_data.get('predictions', []):
        category_name = prediction.get('category_name', '')
        if category_name in filtered_categories:
            filtered_predictions.append(prediction)
    
    # Create filtered data structure
    filtered_data = predictions_data.copy()
    filtered_data['predictions'] = filtered_predictions
    
    # Update similarity analysis to reflect filtered categories
    if 'similarity_analysis' in filtered_data:
        filtered_data['similarity_analysis'] = similarity_data.copy()
        filtered_data['similarity_analysis']['category_similarities'] = filtered_categories
        filtered_data['similarity_analysis']['filtered_categories'] = list(category_similarities.keys())
        filtered_data['similarity_analysis']['kept_categories'] = list(filtered_categories.keys())
        filtered_data['similarity_analysis']['filtering_applied'] = True
        filtered_data['similarity_analysis']['max_similarity'] = max_similarity
        filtered_data['similarity_analysis']['threshold_used'] = similarity_threshold
    
    return filtered_data


def load_segmentation_mask_with_filtering(
    json_path: str,
    *,
    mask_index: int = 0,
    target_size: Optional[Tuple[int, int]] = None,
    similarity_threshold: float = 0.5,
    enable_dynamic_filtering: bool = True,
) -> torch.Tensor:
    """Load a binary segmentation mask with optional similarity-based filtering.
    
    This function extends load_segmentation_mask to include prediction filtering
    based on similarity scores before mask creation.
    
    Parameters
    ----------
    json_path : str
        Absolute path to the ``*_predictions.json`` file.
    mask_index : int, default ``0``
        Which entry in the ``"predictions"`` list to use. Most datasets store a
        single mask per image, so the default is appropriate.
    target_size : tuple[int, int] | None, optional
        Desired ``(height, width)`` of the output mask. If provided the mask is
        resized using nearest-neighbor interpolation to keep the binary nature.
    similarity_threshold : float, default 0.5
        The threshold for high-confidence filtering when enable_dynamic_filtering=True
    enable_dynamic_filtering : bool, default True
        Whether to apply dynamic similarity-based filtering before mask creation
    
    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, H, W)`` with ``float32`` values ``0.0`` or ``1.0``.
        If filtering results in no predictions, returns a zero mask.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Segmentation JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Apply similarity-based filtering if enabled
    if enable_dynamic_filtering:
        data = filter_predictions_by_similarity(data, similarity_threshold, enable_dynamic_filtering)

    predictions = data.get("predictions", [])
    if not predictions:
        # No predictions after filtering - return zero mask
        if target_size is None:
            # Default size if no target specified
            target_size = (1024, 1024)
        return torch.zeros((1, target_size[0], target_size[1]), dtype=torch.float32)
    
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
