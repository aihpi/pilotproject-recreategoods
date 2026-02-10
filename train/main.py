from lightning import Trainer
from dataloader.data_module_redux import FLUXDataModule
from pipelines.train_redux_reverse_two import InstructPix2PixModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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
    torch.backends.cuda.matmul.allow_tf32 = True
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
        args=config,
    )
    # Initialize WandB Logger
    wandb_logger = WandbLogger(
        project="FLUX-Training",
        name=config["training"]["wandb_run_name"],
        log_model=True, 
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",        
        filename="{epoch}-{step}",     
        save_top_k=1,                 
        every_n_epochs=1,              
        monitor="val_lpips",          
        mode="min",
        save_last=False,              
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_lpips",
        patience=10,            
        mode="min",            
        verbose=True,
    )
    training_config = config["training"]
    trainer = Trainer(
        max_epochs=config["model"]["max_epochs"],
        devices="auto",
        accelerator="gpu",
        num_nodes=training_config["num_nodes"],
        strategy=DeepSpeedStrategy(stage=2),
        precision=training_config["precision"],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=training_config["check_val_every_n_epoch"],
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        log_every_n_steps=1, 
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(datamodule=data_module)

if __name__ == "__main__":
    main()