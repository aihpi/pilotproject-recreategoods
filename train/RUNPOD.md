# Running on RunPod

This guide explains how to run the training code on RunPod with a smaller GPU.

## Initial Setup

When you first start your RunPod instance, run the setup script:

```bash
cd /workspace
git clone https://github.com/aihpi/recreategoods.git -b macos-runpod-train
cd recreategoods
bash train/runpod_setup.sh
```

This will:
- Install Miniconda and add it to your .bashrc
- Create the conda environment
- Set up disk space monitoring

## Using Conda

After running the setup script, you need to activate conda:

```bash
# Either start a new terminal session, or:
source ~/.bashrc

# Then activate the conda environment
conda activate train-model
```

## Starting Training

Once conda is activated, you can start training:

```bash
cd /workspace/recreategoods
bash train/runpod_start.sh
```

The script will:
- Check if you're in the correct conda environment
- Automatically adjust training parameters based on your GPU
- Start training with the appropriate configuration

## Troubleshooting

If you encounter issues with conda, try the following:

1. Make sure conda is properly initialized:
```bash
source ~/.bashrc
```

2. Check if the conda environment exists:
```bash
conda env list
```

3. If the environment doesn't exist, create it:
```bash
cd /workspace/recreategoods
conda env create -f train/environment.yaml -n train-model
```

## GPU Memory Optimization

The training script automatically detects your GPU memory and adjusts the batch size and optimization strategy accordingly:

- For GPUs with >16GB memory: Uses batch size 4 with DeepSpeed Stage 2
- For GPUs with 8-16GB memory: Uses batch size 2 with DeepSpeed Stage 3
- For GPUs with <8GB memory: Uses batch size 1 with DeepSpeed Stage 3 and reduced image resolution

## Disk Space Management

The setup script installs a disk space monitor that automatically cleans up old checkpoints when disk usage exceeds 90%. 