#!/usr/bin/env python3
"""
Worker script for processing a specific range of images
"""

import sys
import argparse
from pathlib import Path
import os

# Add the current directory to Python path
sys.path.append('/sc/home/felix.boelter/recreategoods/segmentation')

from pipeline import FashionpediaSegmentationPipeline


def get_image_chunk(input_dir: Path, start_idx: int, end_idx: int):
    """Get a specific chunk of images"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    # Sort for consistent ordering across jobs
    all_images = sorted(all_images)

    # Get the chunk for this job
    chunk = all_images[start_idx:end_idx]
    print(f"Job processing {len(chunk)} images (indices {start_idx} to {end_idx-1})")

    return chunk


def main():
    parser = argparse.ArgumentParser(description='Process a chunk of images')
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--start-idx', type=int, required=True)
    parser.add_argument('--end-idx', type=int, required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--metadata-csv', required=True)

    args = parser.parse_args()

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('API_KEY')

    if not api_key or api_key == 'sk-your-api-key-here':
        print("Error: Please set your API_KEY in the .env file")
        sys.exit(1)

    # Initialize pipeline
    pipeline = FashionpediaSegmentationPipeline(api_key, args.metadata_csv)

    # Get images for this job
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_chunk = get_image_chunk(input_dir, args.start_idx, args.end_idx)

    if not image_chunk:
        print(f"No images found for job {args.job_id}")
        return

    print(f"Job {args.job_id}: Starting processing of {len(image_chunk)} images")

    # Process each image in this chunk
    for i, image_path in enumerate(image_chunk, 1):
        print(f"Job {args.job_id}: Processing {i}/{len(image_chunk)}: {image_path.name}")

        try:
            response = pipeline.segment_image(image_path)
            if response:
                pipeline.save_results(response, image_path.stem, output_dir, image_path)
            print(f"Job {args.job_id}: Completed {i}/{len(image_chunk)}")
        except Exception as e:
            print(f"Job {args.job_id}: Error processing {image_path.name}: {e}")
            continue

    print(f"Job {args.job_id}: All images processed successfully!")


if __name__ == "__main__":
    main()
