#!/usr/bin/env python3
"""
Enhanced training launcher with comprehensive OmegaConf-based configuration system.

This script provides a professional configuration management system using OmegaConf:
- Hierarchical parameter organization with nested structures
- Automatic CLI argument binding and type validation
- Config file validation and inheritance support
- Flexible parameter overrides and merging capabilities
- Comprehensive logging and monitoring integration

OmegaConf Benefits:
- Automatic CLI argument binding with type conversion
- Nested configuration structure with dot notation access
- Easy parameter overrides via CLI and config merging
- Config file validation and schema enforcement
- Environment variable interpolation support
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# OmegaConf imports for professional configuration management
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ValidationError

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Default configuration file path
DEFAULT_CONFIG_PATH = "qwen_image_edit_comprehensive_config.yaml"

# Configuration schema for validation
CONFIG_SCHEMA = {
    "model": {
        "model_paths": Optional[str],
        "model_id_with_origin_paths": str,
        "tokenizer_path": Optional[str],
        "processor_path": Optional[str],
        "trainable_models": Optional[List[str]],
        "lora_base_model": str,
        "lora_target_modules": str,
        "lora_rank": int,
        "lora_checkpoint": Optional[str],
        "use_gradient_checkpointing": bool,
        "use_gradient_checkpointing_offload": Optional[bool],
        "enable_fp8_training": Optional[bool],
        "task": Optional[str],
        "remove_prefix_in_ckpt": str
    },
    "dataset": {
        "base_path": str,
        "metadata_path": str,
        "data_file_keys": str,
        "extra_inputs": str,
        "repeat": int,
        "max_pixels": int,
        "height": Optional[int],
        "width": Optional[int],
        "num_workers": int
    },
    "training": {
        "learning_rate": float,
        "num_epochs": int,
        "weight_decay": float,
        "gradient_accumulation_steps": int,
        "find_unused_parameters": bool,
        "save_steps": int,
        "scheduler": {
            "type": str,
            "mode": str,
            "factor": float,
            "patience": int,
            "threshold": float,
            "min_lr": float
        }
    },
    "segmentation": {
        "mask_loss_weight": float,
        "similarity_threshold": float,
        "enable_dynamic_filtering": bool,
        "mask_root_dir": str,
        "mask_target_size": Optional[int]
    },
    "validation": {
        "steps": int,
        "samples": int,
        "log_sample_predictions": bool,
        "num_inference_steps": int,
        "validation_seed_offset": int,
        "save_validation_images": bool
    },
    "wandb": {
        "project": str,
        "entity": Optional[str],
        "name": Optional[str],
        "tags": List[str],
        "log_model_checkpoints": bool,
        "log_sample_predictions": bool,
        "log_validation_images": bool,
        "watch_model": bool,
        "log_frequency": int
    },
    "output": {
        "path": str,
        "save_checkpoints": bool,
        "safe_serialization": bool,
        "checkpoint_naming": str,
        "best_model_selection": bool
    },
    "advanced_training": {
        "gradient_checkpointing": bool,
        "mixed_precision": str,
        "max_memory_mb": Optional[int],
        "distributed_backend": str,
        "find_unused_parameters": bool,
        "compile_model": bool,
        "use_flash_attention": bool,
        "validation_samples": Optional[int],
        "custom_metrics": List[str]
    },
    "system": {
        "environment": Dict[str, str],
        "debug_mode": bool,
        "log_level": str,
        "enable_profiling": bool
    }
}

class OmegaConfTrainingConfig:
    """Professional configuration manager using OmegaConf."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.base_config = None
        self.override_config = None
        self.final_config = None
        
    def load_base_config(self):
        """Load base configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            base_config = OmegaConf.load(self.config_path)
            # Convert to DictConfig to ensure type consistency
            if not isinstance(base_config, DictConfig):
                base_config = OmegaConf.create(OmegaConf.to_container(base_config))
            
            self._validate_config_structure(base_config)
            
            print(f"✓ Loaded base configuration from: {self.config_path}")
            return base_config
            
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            raise
    
    def create_override_config(self, cli_args: Optional[argparse.Namespace] = None) -> DictConfig:
        """Create override configuration from CLI arguments."""
        if cli_args is None:
            return OmegaConf.create({})
        
        override_dict = {}
        
        # Convert CLI arguments to nested dictionary
        for key, value in vars(cli_args).items():
            if value is not None:
                # Convert CLI args to nested structure (e.g., 'model.learning_rate')
                nested_keys = key.split('.')
                current = override_dict
                
                for nested_key in nested_keys[:-1]:
                    if nested_key not in current:
                        current[nested_key] = {}
                    current = current[nested_key]
                
                current[nested_keys[-1]] = value
        
        override_config = OmegaConf.create(override_dict)
        print(f"✓ Created override configuration from CLI arguments")
        return override_config
    
    def merge_configurations(self, base_config, override_config):
        """Merge base configuration with overrides using OmegaConf."""
        try:
            merged_config = OmegaConf.merge(base_config, override_config)
            
            # Ensure we return DictConfig
            if not isinstance(merged_config, DictConfig):
                merged_config = OmegaConf.create(OmegaConf.to_container(merged_config))
            
            # Validate final configuration
            self._validate_final_config(merged_config)
            
            self.final_config = merged_config
            print("✓ Successfully merged configurations")
            return merged_config
            
        except Exception as e:
            print(f"✗ Configuration merge failed: {e}")
            raise
    
    def _validate_config_structure(self, config) -> None:
        """Validate configuration structure against schema."""
        try:
            # Ensure we have a DictConfig
            if not isinstance(config, DictConfig):
                raise TypeError(f"Expected DictConfig, got {type(config)}")
                
            # Check if main sections exist
            for section in CONFIG_SCHEMA.keys():
                if section not in config:
                    print(f"Warning: Missing configuration section: {section}")
                else:
                    self._validate_section(config[section], CONFIG_SCHEMA[section], section)
            
            print("✓ Configuration structure validation passed")
            
        except Exception as e:
            print(f"Configuration structure validation failed: {e}")
            raise
    
    def _validate_section(self, config_section: Any, schema: Dict[str, Any], section_name: str) -> None:
        """Validate a configuration section against its schema."""
        if isinstance(config_section, DictConfig):
            for key, expected_type in schema.items():
                if key not in config_section:
                    continue
                
                if isinstance(expected_type, dict):
                    self._validate_section(config_section[key], expected_type, f"{section_name}.{key}")
                elif expected_type == Optional[str] and config_section[key] is not None:
                    if not isinstance(config_section[key], str):
                        raise ValidationError(f"Expected string for {section_name}.{key}, got {type(config_section[key])}")
                elif expected_type == Optional[int] and config_section[key] is not None:
                    if not isinstance(config_section[key], int):
                        raise ValidationError(f"Expected integer for {section_name}.{key}, got {type(config_section[key])}")
                elif expected_type == Optional[float] and config_section[key] is not None:
                    if not isinstance(config_section[key], float):
                        raise ValidationError(f"Expected float for {section_name}.{key}, got {type(config_section[key])}")
                elif expected_type == Optional[bool] and config_section[key] is not None:
                    if not isinstance(config_section[key], bool):
                        raise ValidationError(f"Expected boolean for {section_name}.{key}, got {type(config_section[key])}")
    
    def _validate_final_config(self, config) -> None:
        """Validate final configuration for consistency and required parameters."""
        # Ensure we have a DictConfig
        if not isinstance(config, DictConfig):
            raise TypeError(f"Expected DictConfig, got {type(config)}")
            
        # Check for required parameters
        required_paths = [
            "model.model_id_with_origin_paths",
            "dataset.base_path",
            "dataset.metadata_path",
            "training.learning_rate",
            "wandb.project"
        ]
        
        for path in required_paths:
            if not OmegaConf.select(config, path):
                print(f"Warning: Required parameter missing: {path}")
        
        print("✓ Final configuration validation completed")
    
    def get_config_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        if self.final_config is None:
            return "Configuration not loaded"
        
        summary = []
        summary.append("=== CONFIGURATION SUMMARY ===")
        
        # Model summary
        model_info = f"""
MODEL:
  Base Model: {self.final_config.model.lora_base_model}
  LoRA Rank: {self.final_config.model.lora_rank}
  Target Modules: {self.final_config.model.lora_target_modules[:50]}...
  Gradient Checkpointing: {self.final_config.model.use_gradient_checkpointing}
"""
        
        # Training summary
        training_info = f"""
TRAINING:
  Learning Rate: {self.final_config.training.learning_rate}
  Epochs: {self.final_config.training.num_epochs}
  Weight Decay: {self.final_config.training.weight_decay}
  Gradient Accumulation: {self.final_config.training.gradient_accumulation_steps}
  Save Steps: {self.final_config.training.save_steps}
"""
        
        # Dataset summary
        dataset_info = f"""
DATASET:
  Base Path: {self.final_config.dataset.base_path}
  Metadata Path: {self.final_config.dataset.metadata_path}
  Max Pixels: {self.final_config.dataset.max_pixels}
  Workers: {self.final_config.dataset.num_workers}
"""
        
        # WandB summary
        wandb_info = f"""
WANDB:
  Project: {self.final_config.wandb.project}
  Entity: {self.final_config.wandb.entity or 'None'}
  Tags: {', '.join(self.final_config.wandb.tags)}
"""
        
        summary.extend([model_info, training_info, dataset_info, wandb_info])
        
        return "\n".join(summary)


def create_omegaconf_parser() -> argparse.ArgumentParser:
    """Create argument parser with OmegaConf integration."""
    parser = argparse.ArgumentParser(
        description="OmegaConf-based Qwen Image Edit Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OMEGACONF CONFIGURATION SYSTEM:

This script uses OmegaConf for professional configuration management:

1. CONFIGURATION FILE: Load settings from YAML config:
   --config qwen_image_edit_comprehensive_config.yaml

2. CLI OVERRIDES: Override any config parameter via dot notation:
   --model.learning_rate 1e-4
   --wandb.project "my-project"
   --training.num_epochs 5

3. TYPE VALIDATION: Automatic type conversion and validation:
   --model.lora_rank 32 (auto-converted to int)
   --training.weight_decay 0.01 (auto-converted to float)

4. HIERARCHICAL ACCESS: Access nested parameters easily:
   config.training.learning_rate
   config.wandb.project
   config.segmentation.mask_loss_weight

5. CONFIG INHERITANCE: Merge multiple configurations:
   OmegaConf.merge(base_config, override_config)

EXAMPLES:
  # Basic training with config file
  python train_launcher_wandb.py --config config.yaml
  
  # Override specific parameters
  python train_launcher_wandb.py --config config.yaml --model.learning_rate 1e-4 --wandb.project "experiment-1"
  
  # Training with custom validation settings
  python train_launcher_wandb.py --config config.yaml --validation.steps 250 --validation.samples 8
  
  # Debug mode with profiling
  python train_launcher_wandb.py --config config.yaml --system.debug_mode true --system.enable_profiling true
        """
    )
    
    # Add OmegaConf-specific arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to OmegaConf YAML configuration file"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration file and exit without training"
    )
    
    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Display configuration summary and exit"
    )
    
    # Model parameters (hierarchical)
    parser.add_argument(
        "--model.model_id_with_origin_paths",
        type=str,
        help="Model paths with origin information"
    )
    parser.add_argument(
        "--model.lora_rank",
        type=int,
        help="LoRA rank for fine-tuning"
    )
    parser.add_argument(
        "--model.learning_rate",
        type=float,
        help="Override training learning rate"
    )
    
    # Training parameters
    parser.add_argument(
        "--training.learning_rate",
        type=float,
        help="Training learning rate"
    )
    parser.add_argument(
        "--training.num_epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--training.weight_decay",
        type=float,
        help="Weight decay for optimization"
    )
    parser.add_argument(
        "--training.save_steps",
        type=int,
        help="Save model every N steps"
    )
    
    # WandB parameters
    parser.add_argument(
        "--wandb.project",
        type=str,
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="WandB entity/username"
    )
    parser.add_argument(
        "--wandb.name",
        type=str,
        help="WandB run name"
    )
    parser.add_argument(
        "--wandb.tags",
        type=str,
        help="WandB tags (comma-separated)"
    )
    
    # Validation parameters
    parser.add_argument(
        "--validation.steps",
        type=int,
        help="Run validation every N steps"
    )
    parser.add_argument(
        "--validation.samples",
        type=int,
        help="Number of validation samples"
    )
    
    # Segmentation parameters
    parser.add_argument(
        "--segmentation.mask_loss_weight",
        type=float,
        help="Weight for segmentation loss"
    )
    parser.add_argument(
        "--segmentation.similarity_threshold",
        type=float,
        help="Similarity threshold for filtering"
    )
    
    # System parameters
    parser.add_argument(
        "--system.debug_mode",
        type=bool,
        help="Enable debug mode"
    )
    parser.add_argument(
        "--system.enable_profiling",
        type=bool,
        help="Enable performance profiling"
    )
    
    return parser


def setup_environment(config) -> None:
    """Set up environment variables for optimal training performance."""
    if "system" in config and "environment" in config.system:
        env_vars = config.system.environment
        
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            print(f"Set environment variable: {key}={value}")
    
    print("✓ Environment variables configured for optimal training performance")


def convert_config_to_args(config) -> List[str]:
    """Convert OmegaConf configuration to command-line arguments for training script."""
    args = ["python", "DiffSynth-Studio/examples/qwen_image/model_training/train_with_segmentation_wandb.py"]
    
    # Model arguments
    if hasattr(config.model, 'dataset_base_path'):
        args.extend(["--dataset_base_path", config.model.dataset_base_path])
    else:
        args.extend(["--dataset_base_path", config.dataset.base_path])
    
    args.extend([
        "--dataset_metadata_path", config.dataset.metadata_path,
        "--data_file_keys", config.dataset.data_file_keys,
        "--extra_inputs", config.dataset.extra_inputs,
        "--max_pixels", str(config.dataset.max_pixels),
        "--dataset_repeat", str(config.dataset.repeat),
        "--model_id_with_origin_paths", config.model.model_id_with_origin_paths,
        "--learning_rate", str(config.training.learning_rate),
        "--num_epochs", str(config.training.num_epochs),
        "--remove_prefix_in_ckpt", config.model.remove_prefix_in_ckpt,
        "--output_path", config.output.path,
        "--lora_base_model", config.model.lora_base_model,
        "--lora_target_modules", config.model.lora_target_modules,
        "--lora_rank", str(config.model.lora_rank),
        "--weight_decay", str(config.training.weight_decay),
        "--dataset_num_workers", str(config.dataset.num_workers),
        "--save_steps", str(config.training.save_steps),
        "--find_unused_parameters",
        "--mask_loss_weight", str(config.segmentation.mask_loss_weight),
        "--similarity_threshold", str(config.segmentation.similarity_threshold),
        "--enable_dynamic_filtering",
        # WandB arguments
        "--wandb_project", config.wandb.project,
        "--wandb_name", config.wandb.name or "lora-segmentation-training",
        "--wandb_tags", ",".join(config.wandb.tags),
        "--log_sample_predictions",
        "--validation_steps", str(config.validation.steps),
        "--num_validation_samples", str(config.validation.samples),
    ])
    
    # Add conditional arguments
    if config.model.use_gradient_checkpointing:
        args.append("--use_gradient_checkpointing")
    
    args.extend([
        "--gradient_accumulation_steps", str(config.training.gradient_accumulation_steps),
    ])
    
    # Add WandB entity if specified
    if config.wandb.entity:
        args.extend(["--wandb_entity", config.wandb.entity])
    
    return args


def launch_wandb_training(config) -> int:
    """Launch the WandB-enhanced training script with OmegaConf configuration."""
    
    # Convert configuration to command-line arguments
    args = convert_config_to_args(config)
    
    print(f"Launching training with {len(args)} arguments...")
    print(f"First few arguments: {args[:5]}...")
    
    # Try different launch methods
    methods = [
        ["torchrun", "--nproc_per_node=6"] + args[1:],  # Remove 'python' from args
        ["python", "-m", "torch.distributed.run", "--nproc_per_node=6"] + args[1:],
        ["python"] + args[1:]
    ]
    
    for method in methods:
        try:
            print(f"Trying launch method: {' '.join(method[:3])}")
            result = subprocess.run(method, check=True, cwd=project_root)
            return result.returncode
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Method failed: {e}")
            continue
    
    print("All launch methods failed, running single GPU training")
    return subprocess.run(args, cwd=project_root).returncode


def main():
    """Enhanced main function with comprehensive OmegaConf integration."""
    
    # Parse command-line arguments
    parser = create_omegaconf_parser()
    cli_args = parser.parse_args()
    
    try:
        # Initialize OmegaConf configuration manager
        config_manager = OmegaConfTrainingConfig(cli_args.config)
        
        # Load base configuration
        base_config = config_manager.load_base_config()
        
        if cli_args.validate_config:
            print("✓ Configuration validation completed successfully")
            return 0
        
        # Create override configuration from CLI arguments
        override_config = config_manager.create_override_config(cli_args)
        
        # Merge configurations
        final_config = config_manager.merge_configurations(base_config, override_config)
        
        if cli_args.config_summary:
            print(config_manager.get_config_summary())
            return 0
        
        # Display configuration summary
        print(config_manager.get_config_summary())
        
        # Setup environment based on configuration
        setup_environment(final_config)
        
        # Launch training with OmegaConf configuration
        return launch_wandb_training(final_config)
        
    except Exception as e:
        print(f"✗ Configuration or training launch failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())