#!/usr/bin/env python3
"""
Disk management utilities for training on limited disk space.
This script provides functions to monitor and manage disk space during training.
"""

import os
import shutil
import psutil
import glob
import time
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("disk_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiskManager")

def get_disk_usage(path="/"):
    """Get disk usage as a percentage."""
    return psutil.disk_usage(path).percent

def get_directory_size(path):
    """Get the size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

def clean_checkpoints(checkpoint_dir, keep_last_n=3):
    """Clean old checkpoints, keeping only the latest n."""
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return
    
    # Get all checkpoint directories
    checkpoint_paths = sorted(
        glob.glob(os.path.join(checkpoint_dir, "epoch=*")),
        key=os.path.getmtime
    )
    
    # Keep the latest n checkpoints
    if len(checkpoint_paths) > keep_last_n:
        for path in checkpoint_paths[:-keep_last_n]:
            logger.info(f"Removing old checkpoint: {path}")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                logger.error(f"Error removing checkpoint {path}: {e}")

def clean_wandb_files(keep_last_n=3):
    """Clean old wandb files, keeping only the latest n runs."""
    wandb_dir = os.path.join(os.getcwd(), "wandb")
    if not os.path.exists(wandb_dir):
        return
    
    # Get all wandb run directories
    run_dirs = sorted(
        [d for d in glob.glob(os.path.join(wandb_dir, "run-*")) if os.path.isdir(d)],
        key=os.path.getmtime
    )
    
    # Keep the latest n runs
    if len(run_dirs) > keep_last_n:
        for path in run_dirs[:-keep_last_n]:
            logger.info(f"Removing old wandb run: {path}")
            try:
                shutil.rmtree(path)
            except Exception as e:
                logger.error(f"Error removing wandb run {path}: {e}")

def monitor_disk_space(threshold=90, check_interval=300, checkpoint_dir=None):
    """
    Monitor disk space and clean up when usage exceeds threshold.
    
    Args:
        threshold: Disk usage percentage threshold to trigger cleanup
        check_interval: Time in seconds between checks
        checkpoint_dir: Directory containing checkpoints to clean
    """
    logger.info(f"Starting disk space monitoring. Threshold: {threshold}%, Interval: {check_interval}s")
    
    while True:
        disk_usage = get_disk_usage()
        logger.info(f"Current disk usage: {disk_usage:.1f}%")
        
        if disk_usage > threshold:
            logger.warning(f"Disk usage ({disk_usage:.1f}%) exceeds threshold ({threshold}%). Cleaning up...")
            
            if checkpoint_dir and os.path.exists(checkpoint_dir):
                clean_checkpoints(checkpoint_dir)
            
            clean_wandb_files()
            
            # Check if we're still above threshold
            new_disk_usage = get_disk_usage()
            if new_disk_usage > threshold:
                logger.warning(f"Disk usage still high ({new_disk_usage:.1f}%) after cleanup.")
        
        time.sleep(check_interval)

def main():
    parser = argparse.ArgumentParser(description="Disk space management for training")
    parser.add_argument("--threshold", type=int, default=90, help="Disk usage percentage threshold to trigger cleanup")
    parser.add_argument("--interval", type=int, default=300, help="Time in seconds between checks")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing checkpoints to clean")
    args = parser.parse_args()
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.getcwd(), "train", "checkpoints")
    
    try:
        monitor_disk_space(args.threshold, args.interval, args.checkpoint_dir)
    except KeyboardInterrupt:
        logger.info("Disk monitoring stopped by user.")
    except Exception as e:
        logger.error(f"Error in disk monitoring: {e}")

if __name__ == "__main__":
    main() 