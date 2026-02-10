'''Tests for the segmentation utilities.'''

import os
import torch

from diffsynth.trainers.segmentation_utils import load_segmentation_mask


def test_load_segmentation_mask():
    """Load a mask from a known prediction JSON and verify its properties."""
    # Construct absolute path to a sample prediction JSON in the repository.
    json_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../segmentation/segmentation_output_parallel/predictions/0a0a6c9d9dbd4141955b312147379a79_predictions.json",
        )
    )

    # Load the mask using the helper.
    mask = load_segmentation_mask(json_path)

    # Basic sanity checks.
    assert isinstance(mask, torch.Tensor), "Mask should be a torch.Tensor"
    assert mask.dtype == torch.float32, "Mask tensor should be float32"
    # Expected shape: (1, H, W)
    assert mask.ndim == 3 and mask.shape[0] == 1, "Mask should have shape (1, H, W)"
