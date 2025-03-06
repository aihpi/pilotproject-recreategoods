from pytorch_lightning import Trainer
from dataloader.data_module import FLUXDataModule
from pipelines.train_pipeline_optim import InstructPix2PixModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import argparse
from omegaconf import OmegaConf
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from pytorch_lightning.strategies import DeepSpeedStrategy
import shutil
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
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
        except Exception as e:
            print(f"Warning: Could not get system information: {e}")

    torch.cuda.empty_cache()
    data_config = config["data"]
    data_module = FLUXDataModule(
        batch_size=data_config["batch_size"],
        val_batch_size=data_config["val_batch_size"],
        num_workers=data_config["num_workers"],
        data_dir=data_config["data_dir"],
        image_size=data_config["resize_res"]
    )

    # Initialize your FLUX model 
    model = InstructPix2PixModel(
        args=config["model"],
    )
    
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
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()