import os
import sys
sys.path.append("./")
import argparse
import lightning as pl
from lightning.pytorch.strategies import FSDPStrategy
from data_module import PromptDataModule
from inference_pipeline import PromptProcessor
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--max_out", type=int, default=10)
    parser.add_argument("--min_cfg", type=float, default=5.5)
    parser.add_argument("--max_cfg", type=float, default=12.5)
    parser.add_argument("--min_p2p_threshold", type=float, default=0.6)
    parser.add_argument("--max_p2p_threshold", type=float, default=0.85)
    parser.add_argument("--clip-threshold", type=float, default=0.12)
    parser.add_argument("--clip-dir-threshold", type=float, default=0.12)
    parser.add_argument("--clip-img-threshold", type=float, default=0.85)
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    clip_thresholds = {
        "max_out_samples": args.max_out,
        "clip_threshold": args.clip_threshold,
        "clip_dir_threshold": args.clip_dir_threshold,
        "clip_img_threshold": args.clip_img_threshold,
    }
    # Determine distributed settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision('high')
    # Initialize Data Module
    datamodule = PromptDataModule(
        prompts_file=args.prompts_file,
        n_samples=args.n_samples,
        world_size=world_size,
        local_rank=local_rank
    )

    # Initialize Processor
    model = PromptProcessor(
        out_dir=args.out_dir,
        steps=args.steps,
        min_cfg=args.min_cfg,
        max_cfg=args.max_cfg,
        min_threshold=args.min_p2p_threshold,
        max_threshold=args.max_p2p_threshold,
        mixed_precision=args.mixed_precision,
        clip_thresholds=clip_thresholds,
        is_main_process=local_rank == 0,
        rank = local_rank
    )

    # Configure Trainer for Inference
    trainer = pl.Trainer(
        devices='auto',
        accelerator="gpu",
        strategy=FSDPStrategy(),
        max_epochs=1,
        log_every_n_steps=10,
        precision="16-mixed" if args.mixed_precision else 32,
    )

    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()
