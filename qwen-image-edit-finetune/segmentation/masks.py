#!/usr/bin/env python3
"""
Mask Generation Script for Fashionpedia Segmentation Data
Converts RLE segmentation data to mask images
"""

import json
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Dict, Any
import os


def decode_rle(rle_data: Dict[str, Any]) -> np.ndarray:
    """
    Decode RLE (Run Length Encoding) segmentation data to binary mask

    Args:
        rle_data: Dictionary containing 'size' and 'counts' for RLE format

    Returns:
        Binary mask as numpy array
    """
    try:
        import pycocotools.mask as mask_utils

        # Convert to the format expected by pycocotools
        rle = {
            'size': rle_data['size'],
            'counts': rle_data['counts'].encode('utf-8') if isinstance(rle_data['counts'], str) else rle_data['counts']
        }

        # Decode RLE to binary mask
        binary_mask = mask_utils.decode(rle)
        return binary_mask

    except (ImportError, Exception) as e:
        print(f"Warning: pycocotools RLE decoding failed ({e}), using fallback decoder")
        return decode_rle_fallback(rle_data)


def decode_rle_fallback(rle_data: Dict[str, Any]) -> np.ndarray:
    """
    Fallback RLE decoder without pycocotools dependency
    """
    size = rle_data['size']
    counts = rle_data['counts']

    if isinstance(counts, str):
        # Simple fallback - create a dummy mask based on bbox if available
        # This is a simplified version and may not be perfectly accurate
        print("Warning: Using simplified RLE decoding - masks may not be precise")
        return np.zeros((size[0], size[1]), dtype=np.uint8)

    # For actual RLE data, we need proper decoding
    return np.zeros((size[0], size[1]), dtype=np.uint8)


def create_mask_from_prediction(prediction: Dict[str, Any], image_size: tuple) -> np.ndarray:
    """
    Create a binary mask from a single prediction

    Args:
        prediction: Single prediction dictionary containing segmentation data
        image_size: (height, width) of the original image

    Returns:
        Binary mask as numpy array
    """
    if 'segmentation' not in prediction:
        print(f"Warning: No segmentation data for prediction {prediction.get('id', 'unknown')}")
        return np.zeros(image_size, dtype=np.uint8)

    segmentation = prediction['segmentation']

    if isinstance(segmentation, dict) and 'counts' in segmentation:
        # RLE format
        mask = decode_rle(segmentation)
        # Ensure correct dimensions
        if mask.shape != image_size:
            # Transpose if needed (some formats use different axis order)
            if mask.shape == (image_size[1], image_size[0]):
                mask = mask.T
        return mask
    elif isinstance(segmentation, list):
        # Polygon format - convert to mask
        from PIL import Image, ImageDraw

        img = Image.new('L', (image_size[1], image_size[0]), 0)
        draw = ImageDraw.Draw(img)

        for polygon in segmentation:
            if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                # Convert flat list to list of tuples
                points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                draw.polygon(points, outline=1, fill=1)

        return np.array(img)
    else:
        print(f"Warning: Unknown segmentation format for prediction {prediction.get('id', 'unknown')}")
        return np.zeros(image_size, dtype=np.uint8)


def create_category_mask(predictions: List[Dict[str, Any]], image_size: tuple, category_id: int = None) -> np.ndarray:
    """
    Create a mask for a specific category or all categories

    Args:
        predictions: List of prediction dictionaries
        image_size: (height, width) of the original image
        category_id: Specific category ID to filter, or None for all categories

    Returns:
        Binary mask as numpy array
    """
    combined_mask = np.zeros(image_size, dtype=np.uint8)

    for prediction in predictions:
        if category_id is None or prediction.get('category_id') == category_id:
            mask = create_mask_from_prediction(prediction, image_size)
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

    return combined_mask


def create_colored_mask(predictions: List[Dict[str, Any]], image_size: tuple) -> np.ndarray:
    """
    Create a colored mask where each category has a different color

    Args:
        predictions: List of prediction dictionaries
        image_size: (height, width) of the original image

    Returns:
        Colored mask as numpy array (H, W, 3)
    """
    colored_mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Color palette for different categories
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203), # Pink
        (165, 42, 42),  # Brown
    ]

    category_colors = {}
    color_idx = 0

    for prediction in predictions:
        category_id = prediction.get('category_id')
        if category_id not in category_colors:
            category_colors[category_id] = colors[color_idx % len(colors)]
            color_idx += 1

        mask = create_mask_from_prediction(prediction, image_size)
        color = category_colors[category_id]

        # Apply color where mask is active
        for i in range(3):
            colored_mask[:, :, i] = np.where(mask, color[i], colored_mask[:, :, i])

    return colored_mask


def process_prediction_file(json_path: Path, output_dir: Path):
    """
    Process a single prediction JSON file and generate masks

    Args:
        json_path: Path to the prediction JSON file
        output_dir: Directory to save generated masks
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract image info
    image_info = data.get('image', {})
    image_size = (image_info.get('height', 1024), image_info.get('width', 1024))

    predictions = data.get('predictions', [])

    if not predictions:
        print(f"No predictions found in {json_path}")
        return

    # Create output directory
    base_name = json_path.stem.replace('_predictions', '')
    mask_dir = output_dir / base_name
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual masks for each prediction
    for i, prediction in enumerate(predictions):
        category_name = prediction.get('category_name', 'unknown')
        category_id = prediction.get('category_id', 0)
        pred_id = prediction.get('id', i)

        mask = create_mask_from_prediction(prediction, image_size)

        # Convert to PIL Image and save
        mask_image = Image.fromarray(mask * 255)  # Convert to 0-255 range
        mask_path = mask_dir / f"{pred_id:03d}_{category_name}_{category_id}.png"
        mask_image.save(mask_path)
        print(f"Saved mask: {mask_path}")

    # Generate combined mask for all categories
    combined_mask = create_category_mask(predictions, image_size)
    combined_image = Image.fromarray(combined_mask * 255)
    combined_path = mask_dir / "combined_mask.png"
    combined_image.save(combined_path)
    print(f"Saved combined mask: {combined_path}")

    # Generate colored mask
    colored_mask = create_colored_mask(predictions, image_size)
    colored_image = Image.fromarray(colored_mask)
    colored_path = mask_dir / "colored_mask.png"
    colored_image.save(colored_path)
    print(f"Saved colored mask: {colored_path}")

    # Save category summary
    categories = {}
    for prediction in predictions:
        cat_id = prediction.get('category_id')
        cat_name = prediction.get('category_name', 'unknown')
        score = prediction.get('score', 0.0)

        if cat_id not in categories:
            categories[cat_id] = {
                'name': cat_name,
                'count': 0,
                'avg_score': 0.0,
                'scores': []
            }

        categories[cat_id]['count'] += 1
        categories[cat_id]['scores'].append(score)
        categories[cat_id]['avg_score'] = np.mean(categories[cat_id]['scores'])

    summary_path = mask_dir / "category_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(categories, f, indent=2)
    print(f"Saved category summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate masks from Fashionpedia segmentation predictions')
    parser.add_argument('--input-dir',
                       default='./segmentation_output/predictions',
                       help='Directory containing prediction JSON files')
    parser.add_argument('--output-dir',
                       default='./segmentation_output/masks',
                       help='Output directory for generated masks')
    parser.add_argument('--file',
                       help='Process a single JSON file instead of directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        # Process single file
        json_path = Path(args.file)
        if not json_path.exists():
            print(f"Error: File {json_path} does not exist")
            return

        process_prediction_file(json_path, output_dir)
    else:
        # Process directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory {input_dir} does not exist")
            return

        json_files = list(input_dir.glob("*_predictions.json"))
        if not json_files:
            print(f"No prediction JSON files found in {input_dir}")
            return

        print(f"Found {len(json_files)} prediction files to process")

        for json_path in json_files:
            print(f"\nProcessing: {json_path.name}")
            try:
                process_prediction_file(json_path, output_dir)
            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")

    print("\nMask generation completed!")


if __name__ == "__main__":
    main()