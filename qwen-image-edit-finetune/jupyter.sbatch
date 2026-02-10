#!/bin/bash
#SBATCH --job-name=jupyter-clipseg
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/jupyter_%j.out
#SBATCH --error=logs/jupyter_%j.err

# Create logs directory
mkdir -p logs

# Activate environment
source DiffSynth-Studio/curriculum_env/bin/activate

# Get node info
NODE=$(hostname)
echo "Jupyter running on node: $NODE"
echo "Job ID: $SLURM_JOB_ID"

# Print connection info
echo "To connect:"
echo "1. SSH tunnel: ssh -L 8888:$NODE:8888 felix.boelter@your-login-node"
echo "2. Open browser: http://localhost:8888"

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root