#!/usr/bin/env python3
"""
Analyze differences between original and edited images in training dataset
"""

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from scipy import ndimage

def compute_image_diff(img1_path, img2_path, method='lab_delta'):
    """
    Compute difference between two images
    Methods: 'simple', 'lab_delta', 'perceptual'
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return None

    # Resize to same dimensions if needed
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

    if method == 'simple':
        # Simple RGB difference
        diff = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    elif method == 'lab_delta':
        # LAB color space difference (more perceptual)
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

        # Delta E calculation (simplified)
        diff_lab = lab1.astype(np.float32) - lab2.astype(np.float32)
        diff_gray = np.sqrt(np.sum(diff_lab**2, axis=2))
        diff_gray = (diff_gray / diff_gray.max() * 255).astype(np.uint8)

    return diff_gray

def create_mask_from_diff(diff_gray, threshold=30, morphology_kernel=5):
    """
    Create a binary mask from difference image
    """
    # Threshold to create binary mask
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill holes
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255

    return mask

def process_training_diffs(df, base_dir, output_dir="training_diff_masks", max_samples=10):
    """
    Process training dataset to create difference masks
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"Processing {min(max_samples, len(df))} samples...")

    for i in range(min(max_samples, len(df))):
        row = df.iloc[i]

        # Construct absolute paths
        original_path = os.path.join(base_dir, row['image'])
        edited_path = os.path.join(base_dir, row['edit_image'])

        if not (os.path.exists(original_path) and os.path.exists(edited_path)):
            print(f"Missing files for sample {i}: {original_path} or {edited_path}")
            continue

        try:
            # Compute differences
            diff_lab = compute_image_diff(original_path, edited_path, 'lab_delta')

            if diff_lab is None:
                print(f"Could not process images for sample {i}")
                continue

            # Create masks with different thresholds
            mask_strict = create_mask_from_diff(diff_lab, threshold=50)  # High threshold
            mask_medium = create_mask_from_diff(diff_lab, threshold=30)  # Medium threshold
            mask_loose = create_mask_from_diff(diff_lab, threshold=15)   # Low threshold

            # Save masks
            base_name = f"sample_{i:04d}"

            diff_path = os.path.join(output_dir, f"{base_name}_diff.png")
            mask_strict_path = os.path.join(output_dir, f"{base_name}_mask_strict.png")
            mask_medium_path = os.path.join(output_dir, f"{base_name}_mask_medium.png")
            mask_loose_path = os.path.join(output_dir, f"{base_name}_mask_loose.png")

            cv2.imwrite(diff_path, diff_lab)
            cv2.imwrite(mask_strict_path, mask_strict)
            cv2.imwrite(mask_medium_path, mask_medium)
            cv2.imwrite(mask_loose_path, mask_loose)

            # Calculate statistics
            coverage_strict = np.mean(mask_strict > 0) * 100
            coverage_medium = np.mean(mask_medium > 0) * 100
            coverage_loose = np.mean(mask_loose > 0) * 100

            results.append({
                'sample_idx': i,
                'original_path': original_path,
                'edited_path': edited_path,
                'prompt': str(row['prompt']),
                'diff_path': diff_path,
                'mask_strict_path': mask_strict_path,
                'mask_medium_path': mask_medium_path,
                'mask_loose_path': mask_loose_path,
                'coverage_strict': coverage_strict,
                'coverage_medium': coverage_medium,
                'coverage_loose': coverage_loose,
                'diff_mean': float(np.mean(diff_lab)),
                'diff_std': float(np.std(diff_lab)),
                'diff_max': float(np.max(diff_lab))
            })

            print(f"Sample {i:3d}: Coverage - Strict: {coverage_strict:5.1f}%, Medium: {coverage_medium:5.1f}%, Loose: {coverage_loose:5.1f}%")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'diff_analysis_results.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\nProcessed {len(results)} samples successfully")
    print(f"Results saved to {results_path}")

    # Print summary statistics
    if len(results) > 0:
        print("\n" + "="*60)
        print("MASK COVERAGE STATISTICS:")
        for threshold in ['strict', 'medium', 'loose']:
            col = f'coverage_{threshold}'
            print(f"\n{threshold.upper()} threshold:")
            print(f"  Mean coverage: {results_df[col].mean():.1f}%")
            print(f"  Std coverage: {results_df[col].std():.1f}%")
            print(f"  Min coverage: {results_df[col].min():.1f}%")
            print(f"  Max coverage: {results_df[col].max():.1f}%")

        print(f"\nDIFFERENCE IMAGE STATISTICS:")
        print(f"  Mean diff intensity: {results_df['diff_mean'].mean():.1f}")
        print(f"  Mean diff variation: {results_df['diff_std'].mean():.1f}")
        print(f"  Mean diff max: {results_df['diff_max'].mean():.1f}")

    return results_df

if __name__ == "__main__":
    # Load the training dataset
    train_csv = "/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/train.csv"
    base_dir = "/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/"
    output_dir = "training_diff_analysis"

    print(f"Loading dataset from: {train_csv}")

    if not os.path.exists(train_csv):
        print(f"ERROR: CSV file not found at {train_csv}")
        exit(1)

    df_train = pd.read_csv(train_csv)
    print(f"Training dataset: {len(df_train)} samples")
    print(f"Columns: {df_train.columns.tolist()}")

    # Process first 20 samples
    results = process_training_diffs(df_train, base_dir, output_dir, max_samples=20)

    print(f"\nAnalysis complete! Check the '{output_dir}' directory for generated masks.")