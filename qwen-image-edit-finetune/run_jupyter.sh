#!/bin/bash
# Script to run Jupyter on SLURM GPU

# Request GPU resources
srun --partition=aisc --gres=gpu:1 --mem=32G --time=4:00:00 --pty bash -c "
    cd /sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune
    source DiffSynth-Studio/curriculum_env/bin/activate

    # Get the hostname for SSH tunneling
    echo 'Starting Jupyter on node:' \$(hostname)
    echo 'You can access it via SSH tunnel:'
    echo 'ssh -L 8888:'\$(hostname)':8888 felix.boelter@login-node'

    # Start Jupyter
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
"