#!/usr/bin/env python3
"""
Convert metadata_edit.csv to the format expected by VIEScore script
"""

import pandas as pd
import os
from pathlib import Path

def convert_metadata_to_viescore_format():
    """Convert metadata_edit.csv to edit_instructions.csv format"""

    # Read the metadata file
    df = pd.read_csv('metadata_edit.csv')

    # Create new format with row_number and edit_instruction
    converted_data = []

    for idx, row in df.iterrows():
        # Extract the image number from the path (e.g., "images/image_000.png" -> "0")
        image_path = row['image']
        # Extract the number from image_XXX.png
        if 'image_' in image_path:
            # Extract number after "image_" and before ".png"
            image_num_str = image_path.split('image_')[1].split('.')[0]
            # Convert to int to remove leading zeros, then add buffer of 2
            image_num = str(int(image_num_str) + 2)
        else:
            # Fallback: use the index + 2
            image_num = str(idx + 2)

        # Check if corresponding image exists in extracted_images
        image_file = f"extracted_images/{image_num}.png"
        if os.path.exists(image_file):
            converted_data.append({
                'row_number': image_num,
                'edit_instruction': row['prompt']
            })
            print(f"✓ Mapped {image_path} -> {image_num}.png")
        else:
            print(f"⚠ Image not found: {image_file} (from {image_path})")

    # Create DataFrame and save
    converted_df = pd.DataFrame(converted_data)
    converted_df.to_csv('edit_instructions.csv', index=False)

    print(f"\n✅ Conversion complete!")
    print(f"Created edit_instructions.csv with {len(converted_df)} entries")
    print(f"Available images: {len([f for f in os.listdir('extracted_images') if f.endswith('.png')])}")

    return converted_df

if __name__ == "__main__":
    print("Converting metadata_edit.csv to VIEScore format...")
    df = convert_metadata_to_viescore_format()
    print("\nFirst 5 entries:")
    print(df.head())