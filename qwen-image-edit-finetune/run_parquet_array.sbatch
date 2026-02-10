#!/bin/bash
#SBATCH --job-name=par2dss-array
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --time=08:00:00
#SBATCH --array=3-11                 # incomplete shards only
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --constraint=ARCH:X86
#SBATCH --exclude=gx11               # exclude slow node
#SBATCH --output=logs/arr_%A_%a.out
#SBATCH --error=logs/arr_%A_%a.err

# venv (adjust if different)
source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/.venv/bin/activate

# threads: Arrow fans out, BLAS stays single-threaded
export ARROW_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p logs data/example_image_dataset

NUM_SHARDS=16  # keep original total shard count
SHARD_ID=${SLURM_ARRAY_TASK_ID}

python dataloading.py \
  --data_root /sc/projects/sci-aisc/recreategoods/data \
  --split train \
  --out_root data/example_image_dataset \
  --workers ${SLURM_CPUS_PER_TASK} \
  --arrow_threads ${SLURM_CPUS_PER_TASK} \
  --batch_size 1024 \
  --flush_every 5000 \
  --shard ${SHARD_ID} \
  --num_shards ${NUM_SHARDS}