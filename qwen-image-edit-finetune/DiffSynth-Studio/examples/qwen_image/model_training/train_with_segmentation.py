'''Training script with segmentation mask loss.

This script mirrors ``train.py`` but uses ``UnifiedDatasetSegmentation`` to load
segmentation masks and adds a ``--mask_loss_weight`` CLI argument that controls the
relative contribution of the mask loss to the overall training objective.
'''

import os, argparse

from diffsynth.trainers.unified_dataset_segmentation import UnifiedDatasetSegmentation
from examples.qwen_image.model_training.train import QwenImageTrainingModule
from diffsynth.trainers.utils import qwen_image_parser, launch_training_task, launch_data_process_task, ModelLogger


def main():
    parser = qwen_image_parser()
    # Existing arguments are added by qwen_image_parser(); we add our own.
    parser.add_argument(
        "--mask_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the segmentation loss term (default: 1.0)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for dynamic filtering (default: 0.5)",
    )
    parser.add_argument(
        "--enable_dynamic_filtering",
        action="store_true",
        default=True,
        help="Enable dynamic similarity-based filtering (default: True)",
    )
    parser.add_argument(
        "--disable_dynamic_filtering",
        action="store_false",
        dest="enable_dynamic_filtering",
        help="Disable dynamic similarity-based filtering (use all predictions)",
    )
    args = parser.parse_args()

    # Build the dataset – we need to point to the segmentation folder.
    # The repository layout is:
    #   DiffSynth-Studio/segmentation/segmentation_output_parallel/predictions
    # The script lives in examples/qwen_image/model_training, so we go three
    # levels up to the repo root and then into ``segmentation``.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    segmentation_root = os.path.join(repo_root, "segmentation")

    dataset = UnifiedDatasetSegmentation(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDatasetSegmentation.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
        mask_root_dir=segmentation_root,
        mask_target_size=None,  # keep native resolution; the loss will resize if needed
        similarity_threshold=args.similarity_threshold,
        enable_dynamic_filtering=args.enable_dynamic_filtering,
    )

    # Initialise the model – we pass all original arguments.
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
    # Attach the mask loss weight so that the forward method can read it.
    model.mask_loss_weight = args.mask_loss_weight

    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launcher_map = {
        "sft": launch_training_task,
        "data_process": launch_data_process_task,
        "direct_distill": launch_training_task,
    }
    launcher_map[args.task](dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
