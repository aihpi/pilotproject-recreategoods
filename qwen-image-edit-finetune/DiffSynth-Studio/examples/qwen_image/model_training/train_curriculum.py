# Note: Import resolution errors and type checking issues in this file are due to
# the dynamic nature of the DiffSynth codebase and Pylance's limited visibility.
# These diagnostics do not affect runtime functionality.

import torch, os
from omegaconf import OmegaConf, DictConfig
from types import SimpleNamespace
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig  # type: ignore
from diffsynth.pipelines.flux_image_new import ControlNetInput  # type: ignore
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, qwen_image_parser, launch_data_process_task, launch_training_task_curriculum, launch_ranking_curriculum_training  # type: ignore
from diffsynth.trainers.unified_dataset import UnifiedDataset  # type: ignore
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config_with_omegaconf(config_path=None, args=None):
    """
    Load YAML configuration using OmegaConf and override with command-line arguments.
    Priority: CLI args > YAML config > defaults
    """
    # Register mathematical resolvers for OmegaConf
    # Register arithmetic resolvers
    if not OmegaConf.has_resolver("oc.div"):
        OmegaConf.register_new_resolver("oc.div", lambda x, y: int(x) // int(y))
    if not OmegaConf.has_resolver("oc.mul"):
        OmegaConf.register_new_resolver("oc.mul", lambda x, y: int(x) * int(y))
    if not OmegaConf.has_resolver("oc.add"):
        OmegaConf.register_new_resolver("oc.add", lambda x, y: int(x) + int(y))
    if not OmegaConf.has_resolver("oc.sub"):
        OmegaConf.register_new_resolver("oc.sub", lambda x, y: int(x) - int(y))
    # Start with defaults
    defaults = OmegaConf.create({
        'dataset': {
            'base_path': '',
            'metadata_path': None,
            'max_pixels': 1024*1024,
            'height': None,
            'width': None,
            'data_file_keys': 'image',
            'repeat': 1,
            'num_workers': 0
        },
        'model': {
            'paths': None,
            'id_with_origin_paths': None,
            'tokenizer_path': None,
            'processor_path': None,
            'trainable_models': None
        },
        'lora': {
            'base_model': None,
            'target_modules': 'to_q,to_k,to_v,to_out.0',
            'rank': 32,
            'checkpoint': None
        },
        'training': {
            'learning_rate': 1e-4,
            'num_epochs': 1,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'use_gradient_checkpointing': False,
            'use_gradient_checkpointing_offload': False,
            'find_unused_parameters': False,
            'enable_fp8_training': False,
            'task': 'sft',
            'extra_inputs': None
        },
        'output': {
            'path': './models',
            'remove_prefix_in_ckpt': 'pipe.dit.',
            'save_steps': None
        },
        'curriculum': {
            'ranking_csv': None,
            'schedule': '50:1.0,0.0,0.0,0.0;50:0.8,0.2,0.0,0.0;50:0.8,0.0,0.2,0.0;50:0.8,0.0,0.0,0.2',
            'batch_size': 32,
            'steps_per_epoch': 0,
            'sort_within_tier': False,
            'seed': 42,
            'max_epochs_per_phase': 10
        },
        'early_stopping': {
            'patience': 5,
            'threshold': 0.01
        },
        'validation': {
            'csv_path': None,
            'steps': None,
            'patience': 3,
            'min_delta': 0.001,
            'batch_size': 4,
            'samples': 50
        }
    })

    # Load config file if provided
    config = defaults
    if config_path and os.path.exists(config_path):
        file_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(defaults, file_config)
        print(f"Loaded configuration from: {config_path}")

    # Convert to flat namespace for compatibility with existing code
    merged_args = SimpleNamespace()

    # Map nested config to flat argparse names
    merged_args.dataset_base_path = config.dataset.base_path
    merged_args.dataset_metadata_path = config.dataset.metadata_path
    merged_args.max_pixels = config.dataset.max_pixels
    merged_args.height = config.dataset.height
    merged_args.width = config.dataset.width
    merged_args.data_file_keys = config.dataset.data_file_keys
    merged_args.dataset_repeat = config.dataset.repeat
    merged_args.dataset_num_workers = config.dataset.num_workers

    merged_args.model_paths = config.model.paths
    merged_args.model_id_with_origin_paths = config.model.id_with_origin_paths
    merged_args.tokenizer_path = config.model.tokenizer_path
    merged_args.processor_path = config.model.processor_path
    merged_args.trainable_models = config.model.trainable_models

    merged_args.lora_base_model = config.lora.base_model
    merged_args.lora_target_modules = config.lora.target_modules
    merged_args.lora_rank = config.lora.rank
    merged_args.lora_checkpoint = config.lora.checkpoint

    merged_args.learning_rate = config.training.learning_rate
    merged_args.num_epochs = config.training.num_epochs
    merged_args.weight_decay = config.training.weight_decay
    merged_args.gradient_accumulation_steps = config.training.gradient_accumulation_steps
    merged_args.use_gradient_checkpointing = config.training.use_gradient_checkpointing
    merged_args.use_gradient_checkpointing_offload = config.training.use_gradient_checkpointing_offload
    merged_args.find_unused_parameters = config.training.find_unused_parameters
    merged_args.enable_fp8_training = config.training.enable_fp8_training
    merged_args.task = config.training.task
    merged_args.extra_inputs = config.training.extra_inputs

    merged_args.output_path = config.output.path
    merged_args.remove_prefix_in_ckpt = config.output.remove_prefix_in_ckpt
    merged_args.save_steps = config.output.save_steps

    merged_args.ranking_csv = config.curriculum.ranking_csv
    merged_args.curriculum_schedule = config.curriculum.schedule
    merged_args.curriculum_batch_size = config.curriculum.batch_size
    merged_args.curriculum_steps_per_epoch = config.curriculum.steps_per_epoch
    merged_args.sort_within_tier = config.curriculum.sort_within_tier
    merged_args.curriculum_seed = config.curriculum.seed
    merged_args.max_epochs_per_phase = config.curriculum.max_epochs_per_phase

    merged_args.early_stop_patience = config.early_stopping.patience
    merged_args.early_stop_threshold = config.early_stopping.threshold

    merged_args.validation_csv_path = config.validation.csv_path
    merged_args.validation_steps = config.validation.steps if config.validation.steps is not None else None
    merged_args.validation_patience = config.validation.patience
    merged_args.validation_min_delta = config.validation.min_delta
    merged_args.validation_batch_size = config.validation.batch_size
    merged_args.validation_samples = config.validation.samples

    # Override with CLI args if provided (only for path-like arguments that are commonly overridden)
    if args:
        # Only override path arguments that users commonly want to change without editing config
        path_overrides = ['output_path']
        for key in path_overrides:
            if hasattr(args, key) and getattr(args, key) is not None:
                config_value = getattr(merged_args, key, None)
                cli_value = getattr(args, key)
                # Only override if different from config AND not the default CLI value
                if cli_value != config_value and cli_value != './models':  # ./models is the CLI default
                    setattr(merged_args, key, cli_value)
                    print(f"CLI override: {key} = {cli_value}")

    # Basic validation for required fields
    if not merged_args.dataset_base_path:
        raise ValueError("dataset_base_path is required. Set it in config file or use --dataset_base_path")

    return merged_args



class QwenImageTrainingModule(DiffusionTrainingModule):  # type: ignore
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
            # Following original template: input_image is the target/ground truth
            "input_image": data["image"],  # Target/ground truth (what model should learn to produce)
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
        }

        # Extra inputs (following original template)
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



if __name__ == "__main__":
    # First, check if config argument is provided
    import sys
    config_provided = '--config' in sys.argv

    parser = qwen_image_parser()
    # Add config file argument
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file. CLI args override config values.")

    # If config is provided, make dataset_base_path optional
    if config_provided:
        for action in parser._actions:
            if action.dest == 'dataset_base_path':
                action.required = False
                break

    cli_args = parser.parse_args()

    # Load config and merge with CLI args
    args = load_config_with_omegaconf(cli_args.config, cli_args)

    # Print configuration source
    if cli_args.config:
        print(f"Using configuration from: {cli_args.config}")
        print("CLI arguments will override config file values where specified.")

    # Check if using ranking-based curriculum instead of stage-based
    if args.ranking_csv and os.path.exists(args.ranking_csv):
        print(f"=== RANKING-BASED CURRICULUM TRAINING ===")
        print(f"Ranking CSV: {args.ranking_csv}")
        print(f"Using adaptive curriculum with validation-based phase advancement")

        # Print validation configuration if enabled
        if args.validation_csv_path:
            print(f"Validation CSV: {args.validation_csv_path}")
            print(f"Validation will run every {args.validation_steps} steps")
            print(f"Validation patience: {args.validation_patience} checks")
            print(f"Max epochs per phase: {args.max_epochs_per_phase}")
        else:
            print("No validation dataset provided - using fixed schedule")
       
        # Create model
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

        # Create model logger
        model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)

        # Launch ranking curriculum training
        launch_ranking_curriculum_training(
            ranking_csv_path=args.ranking_csv,
            model=model,
            model_logger=model_logger,
            args=args
        )

    elif args.curriculum_dir and os.path.exists(args.curriculum_dir):
        import glob
        import re
        
        # Find and sort stage files
        stage_pattern = os.path.join(args.curriculum_dir, "stage_*.csv")
        stage_files = glob.glob(stage_pattern)
        if stage_files:
            def extract_stage_number(filename):
                match = re.search(r'stage_(\d+)', filename)
                return int(match.group(1)) if match else 0
    
            stage_files = sorted(stage_files, key=extract_stage_number)
            base_output_path = args.output_path

            for i, stage_file in enumerate(stage_files):
                stage_num = i + 1
                print(f"\nCURRICULUM STAGE {stage_num}/{len(stage_files)}")

                args.dataset_metadata_path = stage_file
                args.output_path =f"{base_output_path}/stage_{stage_num:02d}"

                if i > 0:
                    prev_stage_path = f"{base_output_path}/stage_{stage_num - 1:02d}"
                    step_files = glob.glob(f"{prev_stage_path}/step-*.safetensors")
                    if step_files:
                        latest_checkpoint = max(step_files, key=lambda x: int(x.split('step-')[1].split('.')[0]))
                        args.lora_checkpoint = latest_checkpoint
                    else:
                        print(f"Error: No step checkpoint found in {prev_stage_path}")
                        break


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
                launcher_map = {
                    "sft": launch_training_task_curriculum,
                    "data_process": launch_data_process_task,
                    "direct_distill": launch_training_task_curriculum,
                }
                launcher_map[args.task](dataset, model, model_logger, args=args)

    else:
        # Neither ranking CSV nor curriculum directory provided
        print("Error: Either --ranking_csv or --curriculum_dir must be provided.")
        print("Usage:")
        print("  For ranking-based curriculum: --ranking_csv path/to/ranking.csv")
        print("  For stage-based curriculum: --curriculum_dir path/to/stage/files")
        exit(1)
