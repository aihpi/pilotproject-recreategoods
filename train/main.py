#!/usr/bin/env python
# Set environment variables BEFORE any imports
import os
import sys

# Set cache directory environment variables BEFORE importing any HF modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Set protobuf implementation to python as a workaround for protobuf version issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Set custom cache directory
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_CACHE_HOME"] = "/workspace/hf_cache"  # Might be used in some versions
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/datasets"

# Create cache directories immediately
os.makedirs("/workspace/hf_cache", exist_ok=True)
os.makedirs("/workspace/hf_cache/transformers", exist_ok=True)
os.makedirs("/workspace/hf_cache/datasets", exist_ok=True)

# Now import the rest of the modules
from pytorch_lightning import Trainer
from dataloader.data_module import FLUXDataModule
from pipelines.train_pipeline_optim import InstructPix2PixModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
import argparse
from omegaconf import OmegaConf
import shutil
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from pytorch_lightning.strategies import DeepSpeedStrategy
import psutil

def load_config(config_path: str):
    """
    Load the YAML configuration file and return a dictionary.
    If the file cannot be loaded, an exception is raised.
    """
    try:
        return OmegaConf.load(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the config file: {e}")


# Add a memory monitoring callback
class MemoryMonitorCallback(Callback):
    def __init__(self, rank=0):
        super().__init__()
        self.rank = rank
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0 and trainer.global_rank == self.rank:  # Log every 10 batches
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Batch {batch_idx}: GPU memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
            
    def on_validation_start(self, trainer, pl_module):
        if trainer.global_rank == self.rank:
            print("\nGPU memory at validation start:")
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Check available disk space
    disk_space = psutil.disk_usage('/')
    print(f"Disk space: {disk_space.free / 1e9:.2f} GB free of {disk_space.total / 1e9:.2f} GB")
    
    # Also check /tmp disk space
    tmp_space = psutil.disk_usage('/tmp')
    print(f"/tmp disk space: {tmp_space.free / 1e9:.2f} GB free of {tmp_space.total / 1e9:.2f} GB")
    
    # Always use workspace cache directory
    workspace_cache = "/workspace/hf_cache"
    os.makedirs(workspace_cache, exist_ok=True)
    os.makedirs(os.path.join(workspace_cache, "transformers"), exist_ok=True)
    os.makedirs(os.path.join(workspace_cache, "datasets"), exist_ok=True)
    
    print(f"Using {workspace_cache} for cache directory")
    
    # Update environment variables
    os.environ["HF_HOME"] = workspace_cache
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(workspace_cache, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(workspace_cache, "datasets")
    
    # Aggressively clean up disk space
    try:
        # Clear cache directories
        cache_dirs = [
            # Don't clear our workspace cache directory
            # "/workspace/hf_cache",
            # "/workspace/hf_cache/transformers",
            # "/workspace/hf_cache/datasets",
            # Clear old cache directories
            "/tmp/huggingface", 
            "/tmp/huggingface/transformers", 
            "/tmp/huggingface/datasets",
            "/root/.cache/huggingface"
        ]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                print(f"Clearing cache directory: {cache_dir}")
                try:
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                except Exception as e:
                    print(f"Warning: Could not clear cache directory {cache_dir}: {e}")
        
        # Clear other temp directories
        temp_dirs = ["/tmp", "/var/tmp"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                print(f"Cleaning up {temp_dir}...")
                try:
                    # Only remove files older than 1 day
                    cmd = f"find {temp_dir} -type f -mtime +1 -delete"
                    os.system(cmd)
                except Exception as e:
                    print(f"Warning: Could not clean {temp_dir}: {e}")
        
        # Clear pip cache
        try:
            os.system("pip cache purge")
            print("Cleared pip cache")
        except Exception as e:
            print(f"Warning: Could not clear pip cache: {e}")
            
        # Clear apt cache
        try:
            os.system("apt-get clean")
            print("Cleared apt cache")
        except Exception as e:
            print(f"Warning: Could not clear apt cache: {e}")
            
        # Remove unnecessary large files
        large_dirs = [
            "/var/lib/apt/lists",
            "/var/cache/apt/archives"
        ]
        for large_dir in large_dirs:
            if os.path.exists(large_dir):
                print(f"Cleaning up {large_dir}...")
                try:
                    os.system(f"rm -rf {large_dir}/*")
                except Exception as e:
                    print(f"Warning: Could not clean {large_dir}: {e}")
        
        # Check available disk space
        disk_space = psutil.disk_usage('/')
        print(f"Disk space: {disk_space.free / 1e9:.2f} GB free of {disk_space.total / 1e9:.2f} GB")
        
        # Also check /tmp disk space
        tmp_space = psutil.disk_usage('/tmp')
        print(f"/tmp disk space: {tmp_space.free / 1e9:.2f} GB free of {tmp_space.total / 1e9:.2f} GB")
    except Exception as e:
        print(f"Warning: Could not get disk space information: {e}")
    
    config = load_config(args.config_path)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    # Print system information for RunPod
    if local_rank == 0:
        try:
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Available system memory: {psutil.virtual_memory().available / 1e9:.2f} GB")
            print(f"Disk space: {psutil.disk_usage('/').free / 1e9:.2f} GB free of {psutil.disk_usage('/').total / 1e9:.2f} GB")
            
            # Add GPU memory monitoring function
            def print_gpu_memory():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() - torch.cuda.memory_reserved()) / 1e9
                print(f"GPU memory free: {free_memory:.2f} GB")
            
            # Print initial GPU memory state
            print("Initial GPU memory state:")
            print_gpu_memory()
        except Exception as e:
            print(f"Warning: Could not get system information: {e}")

    torch.cuda.empty_cache()
    data_config = config["data"]
    data_module = FLUXDataModule(
        batch_size=data_config["batch_size"],
        val_batch_size=data_config["val_batch_size"],
        num_workers=data_config["num_workers"],
        data_dir=data_config["data_dir"],
        model_name=config["model"]["name"],
        image_size=data_config["resize_res"]
    )

    # Initialize your FLUX model 
    model = InstructPix2PixModel(
        args=config["model"],
    )
    
    print(f"Using optimized pipeline: {model.__class__.__module__}")
    
    # Print GPU memory after model initialization
    if local_rank == 0:
        try:
            print("\nGPU memory after model initialization:")
            print_gpu_memory()
        except Exception as e:
            print(f"Warning: Could not get memory information: {e}")
    
    # Initialize WandB Logger
    wandb_logger = WandbLogger(
        project="FLUX-Training",
        name=config["training"]["wandb_run_name"],
        log_model=True, 
    )
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(os.getcwd(), "train", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,        
        filename="{epoch}-{step}",     
        save_top_k=3,                 
        monitor="val_lpips",          
        mode="min",
        save_last=True,              
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_lpips",
        patience=5,            
        mode="min",            
        verbose=True,
        min_delta=0.001,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    training_config = config["training"]
    
    # Configure strategy based on config
    if "strategy" in training_config and training_config["strategy"] == "deepspeed_stage_2":
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
            offload_parameters=False,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        )
    elif "strategy" in training_config and training_config["strategy"] == "deepspeed_stage_3":
        strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        )
    elif "strategy" in training_config and training_config["strategy"] == "deepspeed_stage_3_offload":
        strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=2e8,
            reduce_bucket_size=2e8,
        )
    elif "strategy" in training_config and training_config["strategy"] == "fsdp":
        strategy = FSDPStrategy(
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            activation_checkpointing=True,
        )
    else:
        strategy = "auto"
    
    trainer = Trainer(
        max_epochs=config["model"]["max_epochs"],
        accelerator="gpu",
        devices=training_config["gpus"],
        num_nodes=training_config["num_nodes"],
        strategy=strategy,
        precision=training_config["precision"],
        logger=wandb_logger,
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch", 1),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        log_every_n_steps=1, 
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, MemoryMonitorCallback(local_rank)],
    )
    
    # Print GPU memory before training starts
    if local_rank == 0:
        try:
            print("\nGPU memory before training starts:")
            print_gpu_memory()
        except Exception as e:
            print(f"Warning: Could not get memory information: {e}")

    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()