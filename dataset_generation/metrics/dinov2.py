from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoFeatureExtractor, Dinov2Model


class DINOv2Similarity(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        """
        Initialize the DINOv2 similarity module.

        Args:
            model_name (str): The name of the DINOv2 model to load.
        """
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.eval().requires_grad_(False)

        # Image normalization parameters for DINOv2
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)))  # Standard mean
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)))  # Standard std

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image using the DINOv2 model.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W) in range [0, 1].

        Returns:
            torch.Tensor: Image embeddings.
        """
        # Normalize the input image
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")

        # Extract image embeddings
        with torch.no_grad():
            outputs = self.model(pixel_values=image)
        image_features = outputs.last_hidden_state.mean(dim=1)  # Pool over spatial dimensions
        image_features = image_features / image_features.norm(dim=1, keepdim=True)  # Normalize

        return image_features

    def forward(
        self, image_0: torch.Tensor, image_1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine similarities between image pairs.

        Args:
            image_0 (torch.Tensor): Tensor of shape (B, C, H, W) for the first set of images.
            image_1 (torch.Tensor): Tensor of shape (B, C, H, W) for the second set of images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Cosine similarity for images and their difference.
        """
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)

        # Compute cosine similarities
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, image_features_1 - image_features_0)

        return sim_image, sim_direction