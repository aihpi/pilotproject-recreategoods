# Qwen Image Curriculum Training Configuration

This directory contains YAML configuration files for curriculum training with the Qwen Image model.

## Usage

### Using a Configuration File

Instead of passing many command-line arguments, you can use a YAML configuration file:

```bash
# Use config file only
python DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py --config configs/example_curriculum.yaml

# Use config file with CLI overrides
python DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py --config configs/example_curriculum.yaml \
    --learning_rate 0.0002 \
    --curriculum_batch_size 8

# For SLURM/sbatch usage
sbatch --export=CONFIG_FILE=configs/my_config.yaml job_script.sh
```

### Configuration Priority

The configuration system follows this priority order:
1. **CLI arguments** (highest priority)
2. **YAML config file**
3. **Default values** (lowest priority)

This means you can set most parameters in the YAML file and override specific ones via command line as needed.

## Configuration File Structure

The YAML config is organized into logical sections:

- **`dataset`**: Dataset paths, resolution, and loading parameters
- **`model`**: Model paths and trainable components
- **`lora`**: LoRA-specific configuration
- **`training`**: Training hyperparameters and options
- **`output`**: Output paths and checkpoint settings
- **`curriculum`**: Curriculum learning schedule and parameters
- **`early_stopping`**: Early stopping configuration
- **`validation`**: Validation dataset and evaluation settings

## Example Files

- **`curriculum_config.yaml`**: Basic template with all available options
- **`example_curriculum.yaml`**: Complete example with realistic values and comments

## Required Parameters

The following parameters must be set either in the config file or via CLI:

- `dataset.base_path` (or `--dataset_base_path`)
- `curriculum.ranking_csv` (or `--ranking_csv`) - for curriculum training

## SLURM Integration

For SLURM job submission, you can reference configs from your job script:

```bash
#!/bin/bash
#SBATCH --job-name=qwen_curriculum
#SBATCH --gpus=2
#SBATCH --time=24:00:00

# Use config file with SLURM
python DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py \
    --config configs/my_training_config.yaml \
    --output_path /scratch/outputs/job_${SLURM_JOB_ID}
```

## Benefits of Using Config Files

1. **Reproducibility**: Save exact training configurations
2. **Clarity**: Better organization than long command lines
3. **Version Control**: Track configuration changes over time
4. **Flexibility**: Override specific values without changing the base config
5. **Documentation**: Comments in YAML explain each parameter
6. **SLURM-friendly**: Easier to manage complex job parameters

## Migration from CLI

To convert existing CLI commands to config files:

1. Copy `curriculum_config.yaml` as a starting template
2. Fill in your specific values for each section
3. Remove CLI arguments that are now in the config
4. Keep CLI arguments only for values you want to override frequently

Example conversion:
```bash
# Before (CLI only)
python DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py \
    --dataset_base_path /data/my_dataset \
    --ranking_csv /data/ranking.csv \
    --learning_rate 0.0001 \
    --lora_rank 32 \
    --curriculum_batch_size 16 \
    --validation_csv_path /data/validation.csv

# After (config + minimal CLI)
python DiffSynth-Studio/examples/qwen_image/model_training/train_curriculum.py \
    --config configs/my_config.yaml
```

This approach is much cleaner and easier to manage for complex training runs with SLURM.