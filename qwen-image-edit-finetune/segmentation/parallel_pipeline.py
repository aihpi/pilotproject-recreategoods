#!/usr/bin/env python3
"""
Parallel Pipeline for Large-Scale Segmentation Processing
Splits 200K images across multiple SLURM jobs for API processing
"""

import argparse
import os
from pathlib import Path
import math
from typing import List


def get_image_list(input_dir: Path) -> List[str]:
    """Get list of all images to process"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f.name for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    return sorted(image_files)


def create_job_script(job_id: int,
                     start_idx: int,
                     end_idx: int,
                     total_images: int,
                     input_dir: str,
                     output_dir: str,
                     metadata_csv: str) -> str:
    """Create SLURM job script for a chunk of images"""

    script_content = f"""#!/bin/bash
#SBATCH --job-name=segment_{job_id:03d}
#SBATCH --output=logs/segment_{job_id:03d}_%j.out
#SBATCH --error=logs/segment_{job_id:03d}_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=aisc
#SBATCH --constraint=ARCH:X86
#SBATCH --account=aisc

# Job info
echo "Job ID: {job_id}"
echo "Processing images {start_idx} to {end_idx} of {total_images}"
echo "Start time: $(date)"

# Load environment
source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio/curriculum_env/bin/activate

# Change to working directory
cd /sc/home/felix.boelter/recreategoods/segmentation

# Force CPU usage for PyTorch (EmbeddingGemma will use CPU)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4

# Run the pipeline for this chunk
python parallel_worker.py \\
    --job-id {job_id} \\
    --start-idx {start_idx} \\
    --end-idx {end_idx} \\
    --input-dir "{input_dir}" \\
    --output-dir "{output_dir}" \\
    --metadata-csv "{metadata_csv}"

echo "End time: $(date)"
echo "Job {job_id} completed successfully"
"""
    return script_content


def create_worker_script():
    """Create the worker script that processes a specific range of images"""

    worker_content = '''#!/usr/bin/env python3
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
'''

    return worker_content


def main():
    parser = argparse.ArgumentParser(description='Setup parallel segmentation processing')
    parser.add_argument('--input-dir',
                       default='/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/control_images/',
                       help='Input directory containing images')
    parser.add_argument('--output-dir',
                       default='./segmentation_output_parallel',
                       help='Output directory for results')
    parser.add_argument('--metadata-csv',
                       default='/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/metadata_edit.csv',
                       help='Path to metadata CSV file')
    parser.add_argument('--num-jobs', type=int, default=-1,
                       help='Number of parallel SLURM jobs (-1 for auto-detect)')
    parser.add_argument('--images-per-job', type=int, default=4000,
                       help='Number of images per job')

    args = parser.parse_args()

    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('slurm_scripts').mkdir(exist_ok=True)

    # Create worker script
    worker_script = create_worker_script()
    with open('parallel_worker.py', 'w') as f:
        f.write(worker_script)
    os.chmod('parallel_worker.py', 0o755)

    # Get total number of images
    input_dir = Path(args.input_dir)
    all_images = get_image_list(input_dir)
    total_images = len(all_images)

    print(f"Total images to process: {total_images}")
    print(f"Images per job: {args.images_per_job}")

    # Calculate number of jobs needed to cover all images
    num_jobs_needed = math.ceil(total_images / args.images_per_job)

    # Use either the specified number of jobs or however many we need (whichever is appropriate)
    if args.num_jobs == -1:  # Special flag for "process all"
        actual_num_jobs = num_jobs_needed
        print(f"Auto-detected: Creating {actual_num_jobs} SLURM jobs to process all {total_images} images")
        print(f"Images per job: {args.images_per_job} (last job may have fewer)")
    else:
        actual_num_jobs = min(args.num_jobs, num_jobs_needed)
        print(f"Creating {actual_num_jobs} SLURM jobs (requested: {args.num_jobs}, needed: {num_jobs_needed})")

        if actual_num_jobs < num_jobs_needed:
            images_to_process = actual_num_jobs * args.images_per_job
            print(f"WARNING: Only processing {images_to_process} out of {total_images} images")

    # Create job scripts
    job_scripts = []
    for job_id in range(actual_num_jobs):
        start_idx = job_id * args.images_per_job
        end_idx = min(start_idx + args.images_per_job, total_images)

        if start_idx >= total_images:
            break

        script_content = create_job_script(
            job_id, start_idx, end_idx, total_images,
            args.input_dir, args.output_dir, args.metadata_csv
        )

        script_path = f"slurm_scripts/job_{job_id:03d}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        job_scripts.append(script_path)
        print(f"Created job {job_id}: images {start_idx} to {end_idx-1}")

    # Create submission script
    submit_script = f"""#!/bin/bash
# Submit all parallel segmentation jobs

echo "Submitting {len(job_scripts)} SLURM jobs..."

"""

    for script in job_scripts:
        submit_script += f"sbatch {script}\n"

    submit_script += f"""
echo "All jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "Check logs in: logs/"
echo "Results will be in: {args.output_dir}"

# Wait for all jobs to complete (optional)
# echo "Waiting for all jobs to complete..."
# while squeue -u $USER | grep -q segment_; do
#     echo "$(date): $(squeue -u $USER | grep segment_ | wc -l) jobs still running..."
#     sleep 60
# done
# echo "All segmentation jobs completed!"
"""

    with open('submit_all_jobs.sh', 'w') as f:
        f.write(submit_script)
    os.chmod('submit_all_jobs.sh', 0o755)

    print(f"\nSetup complete!")
    print(f"Created {len(job_scripts)} job scripts")
    print(f"Run: ./submit_all_jobs.sh to start processing")
    print(f"Monitor: squeue -u $USER")
    print(f"Check progress: ls {args.output_dir}/predictions/ | wc -l")


if __name__ == "__main__":
    main()