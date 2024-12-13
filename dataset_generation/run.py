import os
import sys
sys.path.append("./")
from utils.config_loader import load_config
import argparse
import lightning as pl
from lightning.pytorch.strategies import FSDPStrategy
from data_module import PromptDataModule
from inference_pipeline import PromptProcessor
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    # Set CUDA_VISIBLE_DEVICES based on config
    if "gpu" in config.keys() and "devices" in config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.devices
        print(f"Using GPUs: {config.gpu.devices}")
    else:
        print("No GPU configuration found. Using default CUDA setup.")

    # Set distributed settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision('high')
    if config.mixed_precision:
        torch.set_default_dtype(torch.float16)
    # Initialize Data Module
    datamodule = PromptDataModule(
        prompts_file=config.prompts_file,
        n_samples=config.generation.n_samples,
        world_size=world_size,
        local_rank=local_rank
    )

    # Initialize Processor
    model = PromptProcessor(
        config,
        is_main_process=local_rank == 0,
        rank=local_rank
    )

    # Configure Trainer for Inference
    trainer = pl.Trainer(
        devices='auto',
        accelerator="gpu",
        strategy=FSDPStrategy(),
        max_epochs=1,
        log_every_n_steps=10,
        precision="16-mixed" if config.mixed_precision else 32,
    )

    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()