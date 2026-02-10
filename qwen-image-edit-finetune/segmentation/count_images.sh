#!/bin/bash

INPUT_DIR="/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/control_images/"
IMAGES_PER_JOB=4000

# Count total images
TOTAL_IMAGES=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)

# Calculate number of jobs needed
JOBS_NEEDED=$(( (TOTAL_IMAGES + IMAGES_PER_JOB - 1) / IMAGES_PER_JOB ))
MAX_ARRAY_INDEX=$(( JOBS_NEEDED - 1 ))

echo "Total images found: $TOTAL_IMAGES"
echo "Images per job: $IMAGES_PER_JOB"
echo "Jobs needed: $JOBS_NEEDED"
echo "Array range needed: 0-$MAX_ARRAY_INDEX"
echo ""
echo "Edit submit_segmentation.sh and change line 11 to:"
echo "#SBATCH --array=0-$MAX_ARRAY_INDEX"