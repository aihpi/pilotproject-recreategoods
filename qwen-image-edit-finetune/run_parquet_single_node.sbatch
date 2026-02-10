#!/bin/bash
#SBATCH --job-name=parquet-1node
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks=16                 # num shard processes to run on this node
#SBATCH --cpus-per-task=12          # CPUs per shard → 16*12 = 192 total
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --exclusive
#SBATCH --output=logs/1node_%j.out
#SBATCH --error=logs/1node_%j.err
#SBATCH --exclude=gx11
# If a node is slow, exclude it:
# #SBATCH --exclude=gx11

source /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/.venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DATA_ROOT=/sc/projects/sci-aisc/recreategoods/data
SPLIT=train
OUT_ROOT=data/example_image_dataset

mkdir -p logs "$OUT_ROOT"

NUM_SHARDS=${SLURM_NTASKS}
WPP=${SLURM_CPUS_PER_TASK}     # workers per process

# Each srun task gets a unique SLURM_PROCID [0..NUM_SHARDS-1]
srun --exclusive -N1 -n ${NUM_SHARDS} -c ${WPP} bash -lc '
  export ARROW_NUM_THREADS='"${WPP}"'
  python dataloading.py \
    --data_root '"${DATA_ROOT}"' \
    --split '"${SPLIT}"' \
    --out_root '"${OUT_ROOT}"' \
    --workers '"${WPP}"' \
    --arrow_threads '"${WPP}"' \
    --batch_size 1024 \
    --flush_every 5000 \
    --shard ${SLURM_PROCID} \
    --num_shards '"${NUM_SHARDS}"'
'