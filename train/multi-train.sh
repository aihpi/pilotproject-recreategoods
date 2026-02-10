#!/bin/bash -eux 
# ==============================
# Shebang Line Explanation
# ==============================
# !/bin/bash -eux
# -e: Exit immediately if a command exits with a non-zero status.
#     This ensures that the script stops execution upon encountering an error,
#     preventing subsequent commands from running in a faulty state.
# -u: Treat unset variables as an error and exit immediately.
#     This helps in catching typos or misconfigurations where variables might
#     not have been set, avoiding unexpected behaviors.
# -x: Print each command and its arguments as they are executed.
#     This is useful for debugging purposes, providing a trace of the script's execution.

# ==============================
# SLURM Job Configuration
# ==============================

#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=Ip2p-Training
#SBATCH --output=logs/%j/debug_output.log
#SBATCH --error=logs/%j/debug_error.log
#SBATCH --time=128:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --container-writable

# ==============================
# Environment Variables
# ==============================

# NCCL Configuration for Distributed Training
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE="/workspace/nccl_logs/${SLURM_JOB_ID}_%h_%p.txt"

# General Configuration
export SLURM_DEBUG=verbose
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Paths
export SHARED_STORAGE_ROOT=/home/felix.boelter/
export CONTAINER_WORKSPACE_MOUNT=${SHARED_STORAGE_ROOT}/recreategoods/train
export CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
export CONTAINER_NAME=torch2412

# Distributed Training Settings
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
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
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NNODES: $NNODES"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "MULTIGPU_FLAG: $MULTIGPU_FLAG"
echo "HF_HOME: $HF_HOME"
echo "==========================="

# ==============================
# System Information
# ==============================

srun hostname
srun echo "CPUs on Node: $SLURM_CPUS_ON_NODE"
srun echo "Node ID: $SLURM_NODEID"

# ==============================
# Run Training
# ==============================

srun -l \
    --container-name "$CONTAINER_NAME" \
    --container-mounts "$CONTAINER_WORKSPACE_MOUNT:/workspace,/dev/infiniband:/dev/infiniband" \
    --container-writable \
    --container-workdir /workspace \
    --container-mount-home \
    --export=ALL \
    --nodes=$NNODES \
    --ntasks=$NNODES \
    --ntasks-per-node=1 \
    --verbose \
     ./launcher.sh
# ==============================
# Job Completion Handling
# ==============================

if [[ $? -eq 0 ]]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi

# Cancel the SLURM job to free resources
scancel "$SLURM_JOB_ID"
