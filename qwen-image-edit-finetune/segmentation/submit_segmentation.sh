#!/bin/bash
#SBATCH --job-name=segmentation_array
#SBATCH --output=logs/segment_%A_%a.out
#SBATCH --error=logs/segment_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=aisc
#SBATCH --constraint=ARCH:X86
#SBATCH --account=aisc
#SBATCH --array=0-49

# Configuration
INPUT_DIR="/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/control_images/"
OUTPUT_DIR="./segmentation_output_parallel"
METADATA_CSV="/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/metadata_edit.csv"
IMAGES_PER_JOB=4000
PY_LOG_DIR="./segmentation_logs"
mkdir -p "$PY_LOG_DIR"
# Job info
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing batch $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Load environment
source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio/curriculum_env/bin/activate

# Change to working directory
cd /sc/home/felix.boelter/recreategoods/segmentation

# Force CPU usage for PyTorch (EmbeddingGemma will use CPU)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4

# Calculate start and end indices for this job
START_IDX=$((SLURM_ARRAY_TASK_ID * IMAGES_PER_JOB))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * IMAGES_PER_JOB))

echo "Processing images $START_IDX to $((END_IDX - 1))"

# Create array worker script on the fly
cat > array_worker_${SLURM_ARRAY_TASK_ID}.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append('/sc/home/felix.boelter/recreategoods/segmentation')
from pipeline import FashionpediaSegmentationPipeline

def setup_logger(log_file: Path, task_id: int) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"seg_task_{task_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate messages if root has handlers

    fmt = logging.Formatter(
        fmt="%(asctime)s [task=%(task_id)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class TaskFilter(logging.Filter):
        def filter(self, record):
            record.task_id = task_id
            return True

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    fh.addFilter(TaskFilter())
    logger.addHandler(fh)

    # Stream to stdout as well (so it lands in SLURM .out)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    sh.addFilter(TaskFilter())
    logger.addHandler(sh)

    return logger

def get_image_chunk(input_dir: Path, start_idx: int, end_idx: int, output_dir: Path):
    """Get a specific chunk of images, pre-filtered to skip already processed ones"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    all_images = sorted(all_images)  # consistent order

    # Get the chunk for this task
    chunk = all_images[start_idx:end_idx]

    # Pre-filter: only keep images that haven't been processed yet
    predictions_dir = output_dir / "predictions"
    unprocessed = []
    for img in chunk:
        pred_json = predictions_dir / f"{img.stem}_predictions.json"
        if not pred_json.exists():
            unprocessed.append(img)

    return unprocessed

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int, required=True)
    parser.add_argument('--end-idx', type=int, required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--metadata-csv', required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--log-file', required=True)
    args = parser.parse_args()

    logger = setup_logger(Path(args.log_file), args.task_id)

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # dotenv optional; continue without failing
        pass

    api_key = os.getenv('API_KEY')
    if not api_key or api_key == 'sk-your-api-key-here':
        logger.error("API key missing. Please set API_KEY in the environment or .env.")
        sys.exit(1)

    # Initialize pipeline
    pipeline = FashionpediaSegmentationPipeline(api_key, args.metadata_csv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-filtered to only unprocessed images
    image_chunk = get_image_chunk(input_dir, args.start_idx, args.end_idx, output_dir)
    total = len(image_chunk)
    if not image_chunk:
        logger.info("No unprocessed images found for indices %d..%d - all done!", args.start_idx, args.end_idx - 1)
        return

    logger.info("Processing %d unprocessed images (indices %d..%d)", total, args.start_idx, args.end_idx - 1)

    # Process each image (all are unprocessed)
    for i, image_path in enumerate(image_chunk, 1):
        logger.info("Processing %d/%d: %s", i, total, image_path.name)
        try:
            response = pipeline.segment_image(image_path)
            if response:
                pipeline.save_results(response, image_path.stem, output_dir, image_path)
            logger.info("Completed %d/%d", i, total)
        except Exception as e:
            logger.exception("Error processing %s: %s", image_path.name, e)
            continue

    logger.info("All images processed successfully.")

if __name__ == "__main__":
    main()
EOF

# Run the worker script for this array task‚
python array_worker_${SLURM_ARRAY_TASK_ID}.py \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --metadata-csv "$METADATA_CSV" \
    --log-file "$PY_LOG_DIR/task_${SLURM_ARRAY_TASK_ID}.log" \
    --task-id $SLURM_ARRAY_TASK_ID

# Clean up the temporary worker script
rm -f array_worker_${SLURM_ARRAY_TASK_ID}.py

echo "End time: $(date)"
echo "Task $SLURM_ARRAY_TASK_ID completed successfully"