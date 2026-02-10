import os
import math
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import lightning as pl
from lightning.utilities.rank_zero import rank_zero_only

from diffusers import QwenImageEditPipeline, AutoencoderKLQwenImage


# -----------------------------
# Utilities
# -----------------------------
def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int, None]:
    """Return (width, height, None), both multiples of 32, same math as your script."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height, None

def _to_pixel_values(img: Image.Image, height: int, width: int) -> torch.Tensor:
    """
    Resize with pipeline.image_processor-like semantics, normalize to [-1,1],
    return shape [3,H,W] (no batch, no frame).
    """
    # PIL resize to exact size
    img = img.resize((width, height), Image.BILINEAR)
    arr = (np.array(img).astype(np.float32) / 127.5) - 1.0  # [-1,1]
    # Ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
    return t

# -----------------------------
# Dataset
# -----------------------------

class EditDataset(Dataset):
    def __init__(path_to_parquet: str, width: int, height : int):
        pass