#!/usr/bin/env python
# Set environment variables BEFORE any imports
import os
import sys
import logging

# Configure logging to prevent errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Silence overly verbose loggers
for logger_name in [
    'transformers', 
    'diffusers', 
    'accelerate', 
    'PIL', 
    'httpx', 
    'huggingface_hub',
    'torch.distributed.distributed_c10d'
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Monkey patch tqdm to log all progress bars
import tqdm as tqdm_module
original_tqdm = tqdm_module.tqdm

logger = logging.getLogger("tqdm_tracker")

def patched_tqdm(*args, **kwargs):
    # Get the description if available
    desc = kwargs.get('desc', '')
    if not desc and args and isinstance(args[0], (list, range)):
        desc = f"Iterating over {len(args[0])} items"
    
    logger.info(f"Creating tqdm progress bar: {desc}")
    
    # Call the original tqdm
    return original_tqdm(*args, **kwargs)

# Replace the original tqdm with our patched version
tqdm_module.tqdm = patched_tqdm

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

def patch_huggingface_hub():
    """
    Monkey patch huggingface_hub to log file fetching operations and reduce memory usage.
    """
    logger = logging.getLogger("HFPatcher")
    
    try:
        # Try to patch huggingface_hub's file fetching
        import huggingface_hub
        
        # Store original methods
        original_hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
        original_cached_download = getattr(huggingface_hub.file_download, "cached_download", None)
        
        if original_hf_hub_download:
            def patched_hf_hub_download(*args, **kwargs):
                logger.info(f"Hugging Face Hub download called with args: {args}, kwargs: {kwargs}")
                
                # Force garbage collection before download
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Call original function
                result = original_hf_hub_download(*args, **kwargs)
                
                # Force garbage collection after download
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return result
            
            # Replace the original function
            huggingface_hub.hf_hub_download = patched_hf_hub_download
            logger.info("Patched huggingface_hub.hf_hub_download")
        
        if original_cached_download:
            def patched_cached_download(*args, **kwargs):
                logger.info(f"Hugging Face cached_download called with args: {args}, kwargs: {kwargs}")
                
                # Force garbage collection before download
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Call original function
                result = original_cached_download(*args, **kwargs)
                
                # Force garbage collection after download
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return result
            
            # Replace the original function
            huggingface_hub.file_download.cached_download = patched_cached_download
            logger.info("Patched huggingface_hub.file_download.cached_download")
        
        logger.info("Successfully patched huggingface_hub")
    except Exception as e:
        logger.warning(f"Could not patch huggingface_hub: {e}")

# Now call this function before any imports that might use huggingface_hub
patch_huggingface_hub()

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
import resource
import gc
import threading
import time

# Set a higher limit for open files
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Current file descriptor limits: soft={soft}, hard={hard}")
    # Set soft limit to hard limit or a reasonable value
    new_soft = min(hard, 65536)  # Use hard limit or 65536, whichever is smaller
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    print(f"Updated file descriptor limits: soft={new_soft}, hard={hard}")
except Exception as e:
    print(f"Warning: Could not set file descriptor limits: {e}")

def monitor_resources():
    """
    Monitor system resources and perform cleanup when necessary.
    This function runs in a separate thread.
    """
    logger = logging.getLogger("ResourceMonitor")
    
    while True:
        try:
            # Try to free memory before checking usage
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check memory usage with error handling
            try:
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    logger.warning(f"High memory usage detected ({memory.percent}%). Forcing garbage collection.")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error checking memory: {e}")
                # Try to recover by forcing garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Check file descriptor usage with error handling
            try:
                # Count open file descriptors
                proc = psutil.Process()
                open_files = proc.open_files()
                # Use net_connections() instead of connections() to avoid deprecation warning
                open_connections = proc.net_connections()
                total_fds = len(open_files) + len(open_connections)
                
                # Get current limits
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                
                # If we're using more than 70% of our soft limit, clean up
                if total_fds > soft * 0.7:
                    logger.warning(f"High file descriptor usage detected ({total_fds}/{soft}). Cleaning up.")
                    # Close unnecessary file descriptors
                    for fd in range(100, soft):
                        try:
                            os.close(fd)
                        except:
                            pass
                    gc.collect()
            except Exception as e:
                logger.error(f"Error checking file descriptors: {e}")
                # Try to recover by closing some file descriptors anyway
                try:
                    for fd in range(100, 1000):
                        try:
                            os.close(fd)
                        except:
                            pass
                except:
                    pass
                
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
            # Sleep for a short time to avoid tight loop in case of persistent errors
            time.sleep(5)
        
        # Sleep for 30 seconds before next check
        time.sleep(30)

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


def limit_memory_usage():
    """
    Apply various techniques to limit memory usage.
    """
    logger = logging.getLogger("MemoryLimiter")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Try to limit memory usage by closing file descriptors
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Close file descriptors above 100 (to avoid closing important ones)
        for fd in range(100, soft):
            try:
                os.close(fd)
            except:
                pass
    except Exception as e:
        logger.warning(f"Could not close file descriptors: {e}")
    
    # Try to limit memory usage by setting PyTorch memory allocator settings
    if torch.cuda.is_available():
        try:
            # Set max split size to limit memory fragmentation
            torch.cuda.set_per_process_memory_fraction(0.6)  # Use at most 60% of GPU memory
            
            # Try to enable memory efficient attention if available
            try:
                from diffusers.utils import is_xformers_available
                if is_xformers_available():
                    logger.info("Enabling xformers memory efficient attention")
                    os.environ["DIFFUSERS_XFORMERS_ATTENTION"] = "1"
            except:
                pass
        except Exception as e:
            logger.warning(f"Could not set CUDA memory fraction: {e}")
    
    # Try to limit CPU memory usage
    try:
        import psutil
        # Set process memory limit
        process = psutil.Process()
        # Limit RSS to 70% of total memory
        total_memory = psutil.virtual_memory().total
        process.rlimit(psutil.RLIMIT_RSS, (int(total_memory * 0.7), total_memory))
        
        # Try to reduce memory pressure
        if hasattr(process, "nice"):
            # Set higher nice value to reduce CPU priority
            try:
                process.nice(10)
            except:
                pass
    except Exception as e:
        logger.warning(f"Could not set process memory limit: {e}")
    
    # Try to reduce memory pressure by disabling JIT compilation
    try:
        torch.jit.disable()
        logger.info("Disabled PyTorch JIT compilation")
    except:
        pass
    
    # Try to reduce memory pressure by setting smaller thread pool
    try:
        torch.set_num_threads(1)
        logger.info("Set PyTorch thread pool to 1")
    except:
        pass
    
    # Run garbage collection multiple times
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("Applied memory usage limits")


def optimize_model_initialization(model_args):
    """
    Optimize model initialization to reduce memory usage.
    """
    logger = logging.getLogger("ModelOptimizer")
    
    # Make a copy of the model args to avoid modifying the original
    import copy
    optimized_args = copy.deepcopy(model_args)
    
    # Try to optimize model initialization settings
    try:
        # Use float32 instead of bfloat16 for initialization to reduce memory issues
        if "dtype" in optimized_args and optimized_args["dtype"] == "bfloat16":
            logger.info("Changing dtype from bfloat16 to float32 for initialization")
            optimized_args["dtype"] = "float32"
        
        # Reduce batch size if it's too large
        if "batch_size" in optimized_args and optimized_args["batch_size"] > 1:
            logger.info(f"Reducing batch size from {optimized_args['batch_size']} to 1 for initialization")
            optimized_args["batch_size"] = 1
        
        # Enable gradient checkpointing if available
        optimized_args["use_gradient_checkpointing"] = True
        logger.info("Enabled gradient checkpointing")
        
        # Disable attention slicing if available
        optimized_args["enable_attention_slicing"] = True
        logger.info("Enabled attention slicing")
        
        logger.info("Optimized model initialization settings")
    except Exception as e:
        logger.warning(f"Could not optimize model initialization settings: {e}")
    
    return optimized_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Apply memory limits
    limit_memory_usage()
    
    # Fix for PyTorch distributed logging errors
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        # Silence logging for non-master processes
        logging.getLogger().setLevel(logging.ERROR)
    
    # Start resource monitoring in a background thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    print("Started resource monitoring thread")
    
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
    
    # Fix for distributed initialization logging errors
    if world_size > 1:
        try:
            # Initialize process group with NCCL backend
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                print(f"Initialized process group with rank {local_rank}/{world_size}")
        except Exception as e:
            print(f"Warning: Could not initialize process group: {e}")
    
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

    # Add debug logging
    logger.info("Data module initialized, about to call setup")
    
    # Force setup to run now so we can debug it
    data_module.setup(stage="fit")
    
    logger.info("Data module setup completed, about to initialize model")
    
    # Very aggressive memory cleanup after setup
    logger.info("Performing aggressive memory cleanup after setup")
    
    # Clear all caches
    gc.collect()
    torch.cuda.empty_cache()
    
    # Try to reduce memory usage by closing file descriptors
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        for fd in range(100, soft):
            try:
                os.close(fd)
            except:
                pass
    except Exception as e:
        logger.warning(f"Could not close file descriptors: {e}")
    
    # Try to reduce memory pressure
    try:
        # Drop caches if possible (Linux only)
        if os.path.exists("/proc/sys/vm/drop_caches"):
            os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
            logger.info("Dropped system caches")
    except Exception as e:
        logger.warning(f"Could not drop system caches: {e}")
    
    # Run garbage collection multiple times
    for _ in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Limit memory usage before model initialization
    limit_memory_usage()
    
    # Optimize model initialization settings
    optimized_model_args = optimize_model_initialization(config["model"])
    
    # Initialize your FLUX model with optimized settings
    model = InstructPix2PixModel(
        args=optimized_model_args,
    )
    
    print(f"Using optimized pipeline: {model.__class__.__module__}")
    logger.info("Model initialized, about to set up trainer")
    
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
    
    logger.info("WandB logger initialized")
    
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
    
    logger.info("Callbacks initialized")
    
    training_config = config["training"]
    
    # Set up the trainer
    logger.info("Setting up trainer")
    
    # Get training configuration with defaults for missing values
    # Check both training and model sections for max_epochs
    max_epochs = training_config.get("max_epochs", config["model"].get("max_epochs", 100))  # Default to 100 epochs
    devices = training_config.get("gpus", training_config.get("devices", 1))  # Default to 1 device
    precision = training_config.get("precision", 16)  # Default to 16-bit precision
    gradient_clip_val = training_config.get("gradient_clip_val", 1.0)  # Default to 1.0
    accumulate_grad_batches = training_config.get("accumulate_grad_batches", 1)  # Default to 1
    strategy_name = training_config.get("strategy", "ddp")  # Default to ddp
    
    # Map strategy names to actual strategies
    if strategy_name == "deepspeed" or strategy_name == "deepspeed_stage_2":
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
            offload_parameters=False,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        )
    elif strategy_name == "deepspeed_stage_3" or strategy_name == "deepspeed_stage_3_offload":
        strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        )
    elif strategy_name == "fsdp":
        strategy = FSDPStrategy(
            auto_wrap_policy=None,
            activation_checkpointing=None,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        )
    else:
        strategy = "ddp"
    
    # Map precision strings to actual precision values
    if precision == "bf16-mixed" or precision == "bf16_mixed":
        precision = "bf16-mixed"
    elif precision == "16-mixed" or precision == "16_mixed":
        precision = "16-mixed"
    
    logger.info(f"Using strategy: {strategy}")
    logger.info(f"Training configuration: max_epochs={max_epochs}, devices={devices}, precision={precision}")
    
    # Limit memory usage before creating trainer
    limit_memory_usage()
    
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, MemoryMonitorCallback()],
        strategy=strategy,
        precision=precision,
        accelerator="gpu",
        devices=devices,
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    
    logger.info("Trainer set up, about to start training")
    
    # Start training
    try:
        logger.info("Starting training")
        trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()