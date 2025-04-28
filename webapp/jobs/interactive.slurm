#!/bin/bash -eux 

# Load the SLURM module
module load slurm

# ==============================
# SLURM Job Configuration 
# ==============================

#SBATCH --job-name=interactive_debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j/debug_output.log
#SBATCH --error=logs/%j/debug_error.log
#SBATCH --time=48:00:00
#SBATCH --partition=defq
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
echo "Partition: defq"
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

hostname
echo "CPUs on Node: $SLURM_CPUS_ON_NODE"
echo "Node ID: $SLURM_NODEID"

# ==============================
# Run container with interactive shell
# ==============================

echo "Starting interactive session in container..."
echo "This job will appear as 'RUNNING' in squeue output"
echo "but will wait for you to connect to it."
echo ""
echo "When resources become available, connect with:"
echo "srun --jobid=${SLURM_JOB_ID} --pty bash"
echo ""
echo "IMPORTANT: This job will keep running until:"
echo "1. You connect and exit the shell, or"
echo "2. It hits the time limit (48 hours), or"
echo "3. You cancel it with: scancel ${SLURM_JOB_ID}"
echo ""

# Keep the job running until someone connects
sleep infinity