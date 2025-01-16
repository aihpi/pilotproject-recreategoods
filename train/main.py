from lightning import Trainer
from dataloader.data_module import FLUXDataModule
from pipelines.train_pipeline import InstructPix2PixModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
import argparse
from omegaconf import OmegaConf
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from lightning.pytorch.strategies import DeepSpeedStrategy
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision('medium')

    data_config = config["data"]
    data_module = FLUXDataModule(
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        data_dir=data_config["data_dir"],
        min_resize_res=data_config["min_resize_res"],
        max_resize_res=data_config["max_resize_res"],
        crop_res=data_config["crop_res"],
        flip_prob=data_config["flip_prob"],
    )

    # Initialize your FLUX model 
    model = InstructPix2PixModel(
        args=config["model"],
    )
    # Initialize WandB Logger
    wandb_logger = WandbLogger(
        project="FLUX-Training",
        name="flux-model-run",
        log_model=True, 
    )

    # Set up the trainer
    mixed_precision_config = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
    training_config = config["training"]
    trainer = Trainer(
        max_epochs=config["model"]["max_epochs"],
        devices="auto",
        accelerator="gpu",
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        ),
        precision=training_config["precision"],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        val_check_interval=32,
        log_every_n_steps=2, 
    )

    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(datamodule=data_module)

if __name__ == "__main__":
    main()