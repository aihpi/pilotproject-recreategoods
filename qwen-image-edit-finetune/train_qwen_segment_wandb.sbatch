#!/bin/bash
#SBATCH --job-name=qwen-segment-config-wandb
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h100:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/segment_wandb_config_train_%j.out
#SBATCH --error=logs/segment_wandb_config_train_%j.err

# =============================================================================
# OMEGACONF CONFIGURATION SYSTEM INTEGRATION
# =============================================================================
# This script now uses OmegaConf-based configuration management for:
# - Hierarchical parameter organization with nested structures
# - Automatic CLI argument binding and type validation
# - Config file validation and inheritance support
# - Flexible parameter overrides and merging capabilities
# - Comprehensive logging and monitoring integration
#
# CONFIG FILE: qwen_image_edit_comprehensive_config.yaml
# CLI OVERRIDES: Any parameter can be overridden via dot notation:
#   --model.learning_rate 1e-4 --wandb.project "my-experiment" --training.num_epochs 5
# =============================================================================

# --- env tweaks for throughput/consistency ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export NCCL_DEBUG=warn

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    pip install --break-system-packages uv
fi

# Set up virtual environment with UV
VENV_DIR="/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with UV..."
    uv venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
echo "Virtual environment activated"

# Install all required dependencies from requirements.txt
uv pip install -r DiffSynth-Studio/requirements.txt

# Make sure DiffSynth is on PYTHONPATH
export PYTHONPATH=/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/DiffSynth-Studio:$PYTHONPATH

# =============================================================================
# CONFIGURATION AND TRAINING PARAMETERS
# =============================================================================
# CONFIG_FILE: Path to OmegaConf YAML configuration
CONFIG_FILE="qwen_image_edit_comprehensive_config.yaml"

# OPTIONAL CLI OVERRIDES: Uncomment and modify as needed for quick experiments
# Common override examples (uncomment to use):
# CLI_OVERRIDES="--model.learning_rate 1e-4 --training.num_epochs 5"
# CLI_OVERRIDES="--wandb.project 'quick-test-experiment' --validation.steps 250"
# CLI_OVERRIDES="--segmentation.mask_loss_weight 0.5 --model.lora_rank 32"
# CLI_OVERRIDES="--training.weight_decay 0.01 --wandb.name 'lora-debug-run'"

# Default CLI overrides (can be customized per experiment)
CLI_OVERRIDES="--validation.samples 4"

# Ensure logs/output dirs exist
mkdir -p ./models/train logs

# =============================================================================
# OMEGACONF CONFIG-BASED TRAINING LAUNCH
# =============================================================================
echo "🚀 Starting OmegaConf-based WandB training with configuration system..."
echo "📋 Configuration file: $CONFIG_FILE"
echo "🔧 CLI overrides: ${CLI_OVERRIDES:-'None (using config defaults)'}"
echo "📊 Logs will be saved with 'config' prefix to distinguish from other runs"

# Launch training with OmegaConf configuration system
if [ -n "$CLI_OVERRIDES" ]; then
    python train_launcher_wandb.py --config "$CONFIG_FILE" $CLI_OVERRIDES
else
    python train_launcher_wandb.py --config "$CONFIG_FILE"
fi

echo "✅ OmegaConf-based training completed"
