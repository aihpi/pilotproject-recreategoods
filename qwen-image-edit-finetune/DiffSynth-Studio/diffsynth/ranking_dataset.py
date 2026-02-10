# diffsynth/ranking_dataset.py
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Any
from diffsynth.trainers.unified_dataset import RouteByType, ToAbsolutePath, LoadImage, ImageCropAndResize

class RankingDataset(Dataset[Dict[str, Any]]):
    """
    Dataset that reads from ranking CSV with curriculum tier information.

    Expected CSV columns: image, edit_image, prompt, tier, difficulty, mag,
                         mean_patch_delta, max_patch_delta, lpips, lab_area

    Compatible with DiffSynth QwenImage training pipeline.
    """

    def __init__(
        self,
        ranking_csv_path,
        base_path="",
        max_pixels=1024*1024,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        repeat=1,
    ):
        """
        Args:
            ranking_csv_path: Path to the ranking CSV file
            base_path: Base directory for resolving relative image paths
            max_pixels: Maximum pixels for dynamic resolution
            height: Fixed height (None for dynamic)
            width: Fixed width (None for dynamic)
            height_division_factor: Height must be divisible by this
            width_division_factor: Width must be divisible by this
            repeat: Number of times to repeat dataset per epoch
        """
        self.base_path = base_path
        self.repeat = repeat

        # Load ranking data
        self.df = pd.read_csv(ranking_csv_path)
        required_cols = {"image", "edit_image", "prompt", "tier", "difficulty"}
        if not required_cols.issubset(set(self.df.columns)):
            raise ValueError(f"CSV missing required columns. Expected: {required_cols}, Got: {set(self.df.columns)}")

        self.df = self.df.reset_index(drop=True)
        print(f"Loaded ranking dataset with {len(self.df)} samples")
        print(f"Tier distribution: {self.df['tier'].value_counts().sort_index().to_dict()}")

        # Set up image processing pipeline (same as UnifiedDataset)
        self.image_operator = RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(
                height, width, max_pixels, height_division_factor, width_division_factor
            )),
        ])

        # Dynamic resolution settings
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            raise ValueError("Either both height and width should be specified, or both should be None.")

    def __len__(self):
        return len(self.df) * self.repeat

    def __getitem__(self, idx):
        """
        Return sample in format expected by QwenImageTrainingModule.

        Returns:
            dict with keys: image, prompt, tier, difficulty, and optional metrics
        """
        # Handle repeat logic
        actual_idx = idx % len(self.df)
        row = self.df.iloc[actual_idx]

        # Load and process input image (the image to be edited)
        image_path = row['image']
        try:
            image = self.image_operator(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Load and process target image (the edited result)
        edit_image_path = row['edit_image']
        try:
            edit_image = self.image_operator(edit_image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load edit image {edit_image_path}: {e}")

        # Build sample dict compatible with QwenImageTrainingModule
        sample = {
            # Core training data
            "image": image,  # Input image to be edited
            "edit_image": edit_image,  # Target edited image
            "prompt": str(row['prompt']),  # Edit instruction

            # Curriculum information
            "tier": int(row['tier']),
            "difficulty": float(row['difficulty']),

            # Optional metrics for analysis/logging
            "mag": float(row.get('mag', 0.0)),
            "mean_patch_delta": float(row.get('mean_patch_delta', 0.0)),
            "max_patch_delta": float(row.get('max_patch_delta', 0.0)),
            "lpips": float(row.get('lpips', 0.0)),
            "lab_area": float(row.get('lab_area', 0.0)),

            # Metadata
            "sample_idx": actual_idx,
            "image_path": image_path,
            "edit_image_path": edit_image_path,
        }

        return sample

    def get_tier_stats(self):
        """Return statistics about tier distribution."""
        return {
            "tier_counts": self.df['tier'].value_counts().sort_index().to_dict(),
            "difficulty_stats": self.df.groupby('tier')['difficulty'].agg(['mean', 'std', 'min', 'max']).to_dict(),
            "total_samples": len(self.df),
        }

    def get_samples_by_tier(self, tier):
        """Return all sample indices for a specific tier."""
        return self.df[self.df['tier'] == tier].index.tolist()

# Convenience function for compatibility with existing DiffSynth patterns
def create_ranking_dataset(
    ranking_csv_path,
    base_path="",
    max_pixels=1024*1024,
    height=None,
    width=None,
    repeat=1,
):
    """
    Create a RankingDataset with standard parameters.

    This function provides a simple interface similar to UnifiedDataset creation.
    """
    return RankingDataset(
        ranking_csv_path=ranking_csv_path,
        base_path=base_path,
        max_pixels=max_pixels,
        height=height,
        width=width,
        repeat=repeat,
    )


class ValidationDataset(Dataset[Dict[str, Any]]):
    """
    Simple validation dataset for LPIPS evaluation during curriculum training.

    Expected CSV columns: image, edit_image, prompt
    No tier/difficulty information needed for validation.
    """

    def __init__(
        self,
        validation_csv_path,
        base_path="",
        max_pixels=1024*1024,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
    ):
        """
        Args:
            validation_csv_path: Path to the validation CSV file
            base_path: Base directory for resolving relative image paths
            max_pixels: Maximum pixels for dynamic resolution
            height: Fixed height (None for dynamic)
            width: Fixed width (None for dynamic)
            height_division_factor: Height must be divisible by this
            width_division_factor: Width must be divisible by this
        """
        self.base_path = base_path

        # Load validation data
        self.df = pd.read_csv(validation_csv_path)
        required_cols = {"image", "edit_image", "prompt"}
        if not required_cols.issubset(set(self.df.columns)):
            raise ValueError(f"Validation CSV missing required columns. Expected: {required_cols}, Got: {set(self.df.columns)}")

        self.df = self.df.reset_index(drop=True)
        print(f"Loaded validation dataset with {len(self.df)} samples")

        # Set up image processing pipeline (same as RankingDataset)
        self.image_operator = RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(
                height, width, max_pixels, height_division_factor, width_division_factor
            )),
        ])

        # Dynamic resolution settings
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            raise ValueError("Either both height and width should be specified, or both should be None.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return validation sample in format expected by LPIPS evaluator.

        Returns:
            dict with keys: image, edit_image, prompt, and metadata
        """
        row = self.df.iloc[idx]

        # Load and process input image (the image to be edited)
        image_path = row['image']
        try:
            image = self.image_operator(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Load and process target image (the edited result)
        edit_image_path = row['edit_image']
        try:
            edit_image = self.image_operator(edit_image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load edit image {edit_image_path}: {e}")

        # Build sample dict for validation
        sample = {
            # Core validation data
            "image": image,  # Input image to be edited
            "edit_image": edit_image,  # Target edited image
            "prompt": str(row['prompt']),  # Edit instruction

            # Metadata
            "sample_idx": idx,
            "image_path": image_path,
            "edit_image_path": edit_image_path,
        }

        return sample


# Convenience function for creating validation dataset
def create_validation_dataset(
    validation_csv_path,
    base_path="",
    max_pixels=1024*1024,
    height=None,
    width=None,
):
    """
    Create a ValidationDataset with standard parameters.
    """
    return ValidationDataset(
        validation_csv_path=validation_csv_path,
        base_path=base_path,
        max_pixels=max_pixels,
        height=height,
        width=width,
    )