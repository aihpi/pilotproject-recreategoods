#!/bin/bash -eux 

# Load the SLURM module
module load slurm

# ==============================
# SLURM Job Configuration 
# ==============================

#SBATCH --job-name=fashion_edit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j/debug_output.log
#SBATCH --error=logs/%j/debug_error.log
#SBATCH --time=00:30:00
#SBATCH --partition=shortq
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --container-writable

# Create logs directory
mkdir -p logs/${SLURM_JOB_ID}
mkdir -p nccl_logs

# Paths configuration
export SHARED_STORAGE_ROOT=/home/thomas.chille
export CONTAINER_WORKSPACE_MOUNT=$(pwd)
export CONTAINER_NAME=pyxis_torch_2412

# NCCL Configuration for Distributed Training (Don't Change)
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE="/workspace/nccl_logs/${SLURM_JOB_ID}_%h_%p.txt"
export SLURM_DEBUG=verbose
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Distributed Training Settings (Don't Change)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
export NNODES=${SLURM_NNODES:-1}
export NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))
export MULTIGPU_FLAG="--multi_gpu"

# Hugging Face Cache Directory
export HF_HOME=${SHARED_STORAGE_ROOT}/.huggingface

# Adjust MULTIGPU_FLAG for Single Node
if [[ "$NNODES" -eq "1" ]]; then
    export MULTIGPU_FLAG=""
fi

# ==============================
# Job Information
# ==============================

echo "===== Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Queue: shortq"
echo "Partition: shortq"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NNODES: $NNODES"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "MULTIGPU_FLAG: $MULTIGPU_FLAG"
echo "HF_HOME: $HF_HOME"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "==========================="

# ==============================
# System Information
# ==============================

srun hostname
srun echo "CPUs on Node: $SLURM_CPUS_ON_NODE"
srun echo "Node ID: $SLURM_NODEID"

# Run the inference script with proper container mounts
srun -l \
    --container-name="$CONTAINER_NAME" \
    --container-mounts="$(pwd):/workspace,/dev/infiniband:/dev/infiniband" \
    --container-writable \
    --container-workdir=/workspace \
    --container-mount-home \
    --export=ALL \
    --nodes=$NNODES \
    --ntasks=$NNODES \
    --ntasks-per-node=1 \
    --verbose \
    bash -c "
        # Install required packages inside the container
        pip install diffusers transformers pillow safetensors huggingface_hub

        # Copy custom pipeline to a place Python can find
        mkdir -p ~/.local/lib/python3.12/site-packages/
        cp /workspace/webapp/custom_pipeline.py ~/.local/lib/python3.12/site-packages/
        
        # Run the inference script
        cd /workspace
        python webapp/infer_hpi.py
    "

echo "Job completed at $(date)" 