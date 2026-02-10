#!/bin/bash
#SBATCH --job-name=par2dss
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/par2dss_%j.out
#SBATCH --error=logs/par2dss_%j.err

# Your venv (adjust if different)
source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/.venv/bin/activate

# Threading: let Arrow use the cores; avoid BLAS oversubscription
export ARROW_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p logs data/example_image_dataset

python dataloading.py \
  --data_root /sc/projects/sci-aisc/recreategoods/data \
  --split train \
  --out_root data/example_image_dataset \
  --workers ${SLURM_CPUS_PER_TASK} \
  --arrow_threads ${SLURM_CPUS_PER_TASK} \
  --batch_size 512 \
  --flush_every 2000