#!/usr/bin/env python
# Set environment variables BEFORE any imports
import os
import sys
import logging
import gc
import resource
import torch

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

# Optimize model initialization settings
optimized_model_args = optimize_model_initialization(config["model"])

# Initialize your FLUX model with optimized settings
model = InstructPix2PixModel(
    args=optimized_model_args,
)

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