'''UnifiedDatasetSegmentation

A thin wrapper around :class:`UnifiedDataset` that additionally loads a
segmentation mask for each sample.  The mask is stored in a COCO‑style JSON
file under ``segmentation/segmentation_output_parallel/predictions``.  The
utility function :func:`load_segmentation_mask` (see ``segmentation_utils.py``)
decodes the RLE mask and returns a ``torch.Tensor`` of shape ``(1, H, W)``.

The class mirrors the original ``UnifiedDataset`` API – it can be used
interchangeably wherever a ``torch.utils.data.Dataset`` is expected.
'''

import os
from typing import Tuple, Optional

import torch

# Import the original dataset implementation
from .unified_dataset import UnifiedDataset

# Import the mask‑loading helper we added earlier
from diffsynth.trainers.segmentation_utils import load_segmentation_mask_with_filtering


class UnifiedDatasetSegmentation(UnifiedDataset):
    """Dataset that returns the usual fields plus a ``segmentation_mask``.

    Parameters
    ----------
    mask_root_dir: str
        Directory that contains the ``segmentation_output_parallel/predictions``
        sub‑folder with the JSON files.  The JSON file name is derived from the
        original image path by keeping the stem and appending
        ``"_predictions.json"``.
    mask_target_size: Tuple[int, int] | None, optional
        Desired size of the mask tensor (height, width).  If ``None`` the mask is
        returned at its native resolution.
    similarity_threshold: float, default 0.5
        Similarity threshold for dynamic filtering. Used when enable_dynamic_filtering=True.
    enable_dynamic_filtering: bool, default True
        Whether to apply dynamic similarity-based filtering before mask creation.
    **kwargs: Any
        All other arguments are forwarded to the base ``UnifiedDataset``
        constructor.
    """

    def __init__(
        self,
        mask_root_dir: str,
        mask_target_size: Optional[Tuple[int, int]] = None,
        similarity_threshold: float = 0.5,
        enable_dynamic_filtering: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mask_root_dir = mask_root_dir
        self.mask_target_size = mask_target_size
        self.similarity_threshold = similarity_threshold
        self.enable_dynamic_filtering = enable_dynamic_filtering

    def _mask_path_from_image(self, image_path: str) -> str:
        """Derive the JSON mask path from an image file path.

        The repository stores masks in:
        ``<mask_root_dir>/segmentation_output_parallel/predictions/<stem>_predictions.json``
        where ``<stem>`` is the filename without extension.
        """
        stem = os.path.splitext(os.path.basename(image_path))[0]
        json_name = f"{stem}_predictions.json"
        return os.path.join(
            self.mask_root_dir,
            "segmentation_output_parallel",
            "predictions",
            json_name,
        )

    def __getitem__(self, idx):
        # Get the original sample dict from the parent class
        sample = super().__getitem__(idx)

        # Determine the image path – the original dataset stores the raw path under the
        # key defined in ``self.data_file_keys`` (usually ``"image"``).  We fall back
        # to ``sample.get("image")`` if the key is missing.
        image_path = sample.get("image")
        if isinstance(image_path, str):
            mask_json_path = self._mask_path_from_image(image_path)
            try:
                mask_tensor = load_segmentation_mask_with_filtering(
                    mask_json_path,
                    target_size=self.mask_target_size,
                    similarity_threshold=self.similarity_threshold,
                    enable_dynamic_filtering=self.enable_dynamic_filtering,
                )
                sample["segmentation_mask"] = mask_tensor
            except Exception as exc:
                # If the mask cannot be loaded we raise a clear error – this helps
                # debugging data pipelines early.
                raise FileNotFoundError(
                    f"Failed to load segmentation mask for '{image_path}': {exc}"
                ) from exc
        else:
            # No image path – we simply skip mask loading.
            sample["segmentation_mask"] = None

        return sample
