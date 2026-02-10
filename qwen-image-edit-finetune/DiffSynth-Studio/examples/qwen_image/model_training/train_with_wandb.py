import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, qwen_image_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        enable_fp8_training=False,
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=enable_fp8_training)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=enable_fp8_training,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task


    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}

        # CFG-unsensitive parameters
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
        }

        # Extra inputs
        controlnet_input, blockwise_controlnet_input = {}, {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("blockwise_controlnet_"):
                blockwise_controlnet_input[extra_input.replace("blockwise_controlnet_", "")] = data[extra_input]
            elif extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        if len(blockwise_controlnet_input) > 0:
            inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(**blockwise_controlnet_input)]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None, return_inputs=False):
        # Inputs
        if inputs is None: inputs = self.forward_preprocess(data)
        else: inputs = self.transfer_data_to_device(inputs, self.pipe.device)
        if return_inputs: return inputs

        # Loss
        if self.task == "sft":
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss = self.pipe.training_loss(**models, **inputs)
        elif self.task == "data_process":
            loss = inputs
        elif self.task == "direct_distill":
            loss = self.pipe.direct_distill_loss(**inputs)
        else:
            raise NotImplementedError(f"Unsupported task: {self.task}.")
        return loss


def launch_training_task_with_wandb(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    wandb_project: str = "qwen-image-edit",
    wandb_entity: str = None,
    wandb_name: str = None,
    wandb_tags: list = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        # W&B args from environment or args
        wandb_project = getattr(args, 'wandb_project', wandb_project)
        wandb_entity = getattr(args, 'wandb_entity', wandb_entity)
        wandb_name = getattr(args, 'wandb_name', wandb_name)
        wandb_tags = getattr(args, 'wandb_tags', wandb_tags)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    # Initialize W&B only on main process
    if accelerator.is_main_process:
        # Get config from args
        config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "dataset_size": len(dataset),
        }

        if args is not None:
            config.update({
                "lora_rank": getattr(args, 'lora_rank', None),
                "lora_base_model": getattr(args, 'lora_base_model', None),
                "lora_target_modules": getattr(args, 'lora_target_modules', None),
                "max_pixels": getattr(args, 'max_pixels', None),
                "dataset_repeat": getattr(args, 'dataset_repeat', None),
                "batch_size": 1,  # Since we use collate_fn=lambda x: x[0]
            })

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            tags=wandb_tags,
            config=config,
        )

        # Log model architecture info
        trainable_params = sum(p.numel() for p in model.trainable_modules() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({
            "model/trainable_parameters": trainable_params,
            "model/total_parameters": total_params,
            "model/trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        })

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    # Use ReduceLROnPlateau for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1000,  # steps to wait before reducing
        threshold=0.001,  # minimum change to qualify as improvement
        min_lr=1e-7
    )
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0

    for epoch_id in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch_id + 1}/{num_epochs}")

        for batch_idx, data in enumerate(progress_bar):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)

                # Step scheduler with loss for ReduceLROnPlateau
                if accelerator.is_main_process:
                    scheduler.step(loss.item())

                # Log to W&B
                if accelerator.is_main_process:
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    num_batches += 1

                    wandb.log({
                        "train/loss": loss_value,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                        "train/global_step": global_step,
                    }, step=global_step)

                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

        # Log epoch metrics
        if accelerator.is_main_process and num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch_id,
            }, step=global_step)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)

    # Finish W&B
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parser = qwen_image_parser()
    # Add W&B arguments
    parser.add_argument("--wandb_project", type=str, default="qwen-image-edit", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/username")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="W&B tags (comma-separated)")

    args = parser.parse_args()

    # Parse W&B tags
    if args.wandb_tags:
        args.wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",")]

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        enable_fp8_training=args.enable_fp8_training,
        task=args.task,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)

    launch_training_task_with_wandb(dataset, model, model_logger, args=args)