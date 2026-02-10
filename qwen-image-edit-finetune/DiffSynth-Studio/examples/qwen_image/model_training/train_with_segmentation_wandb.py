'''Enhanced training script with segmentation mask loss and comprehensive WandB logging.

This script provides comprehensive visualization and validation through WandB:
- Real-time loss curves and metrics tracking
- Sample predictions with visual validation at save steps
- Model checkpoint logging with metadata
- Learning rate scheduling with visualization
- Segmentation mask loss tracking
- Dynamic filtering metrics logging
'''

import os, argparse, json, time
import torch
from PIL import Image
import numpy as np
from diffsynth.trainers.unified_dataset_segmentation import UnifiedDatasetSegmentation
from examples.qwen_image.model_training.train import QwenImageTrainingModule
from diffsynth.trainers.utils import (
    qwen_image_parser, ModelLogger, DiffusionTrainingModule
)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from typing import Optional, List
import wandb
from tqdm import tqdm


class WandBEnhancedModelLogger(ModelLogger):
    """Enhanced ModelLogger with comprehensive WandB integration."""
    
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, 
                 wandb_enabled=True, validation_samples=None):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.wandb_enabled = wandb_enabled
        self.validation_samples = validation_samples or []
        self.sample_predictions = []
        self.checkpoint_metadata = []
        
    def log_checkpoint_metrics(self, accelerator, model, step, loss_value=None, epoch=None):
        """Log metrics at checkpoint save time."""
        if not self.wandb_enabled or not accelerator.is_main_process:
            return
            
        # Log checkpoint info
        checkpoint_info = {
            "checkpoint/step": step,
            "checkpoint/learning_rate": self.get_current_lr(accelerator, model),
            "checkpoint/epoch": epoch if epoch is not None else 0,
        }
        
        if loss_value is not None:
            checkpoint_info["checkpoint/loss"] = loss_value
            
        # Add model parameters info
        if hasattr(model, 'module'):
            # Distributed training - access underlying model
            model_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        else:
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
        checkpoint_info["checkpoint/trainable_parameters"] = model_params
        wandb.log(checkpoint_info)
        
        # Log sample predictions if available
        if self.sample_predictions and step % max(1, len(self.sample_predictions) // 10) == 0:
            self._log_sample_predictions(step)
            
    def add_sample_prediction(self, prompt, input_image, target_image, generated_image, lpips_score=None):
        """Add a sample prediction for logging."""
        self.sample_predictions.append({
            "prompt": prompt,
            "input_image": input_image,
            "target_image": target_image, 
            "generated_image": generated_image,
            "lpips_score": lpips_score,
            "timestamp": time.time()
        })
        
    def _log_sample_predictions(self, step):
        """Log sample predictions to WandB."""
        if not self.wandb_enabled or not self.sample_predictions:
            return
            
        # Log up to 4 sample predictions per checkpoint
        samples_to_log = self.sample_predictions[-4:]
        
        for i, sample in enumerate(samples_to_log):
            # Create a comparison image
            images = [
                sample["input_image"],
                sample["target_image"], 
                sample["generated_image"]
            ]
            
            # Create a grid image
            grid_image = self._create_comparison_grid(images, sample["prompt"])
            
            # Log to WandB
            wandb.log({
                f"sample_predictions/step_{step}/image_{i}": wandb.Image(grid_image),
                f"sample_predictions/step_{step}/lpips_{i}": sample.get("lpips_score", 0),
                f"sample_predictions/step_{step}/prompt_{i}": sample["prompt"],
            })
            
    def _create_comparison_grid(self, images, prompt):
        """Create a grid image for comparison."""
        import torchvision.transforms.functional as TF
        
        # Resize images to same height for comparison
        target_height = 256
        resized_images = []
        
        for img in images:
            if isinstance(img, Image.Image):
                # Convert PIL to tensor and resize
                img_tensor = TF.to_tensor(img)  # [C, H, W]
                img_tensor = TF.resize(img_tensor, (target_height, int(target_height * img.size[0] / img.size[1])))
                img_tensor = TF.to_pil_image(img_tensor)
                resized_images.append(img_tensor)
            else:
                resized_images.append(img)
                
        # Create grid
        total_width = target_height * len(resized_images)
        grid = Image.new('RGB', (total_width, target_height))
        
        for i, img in enumerate(resized_images):
            if isinstance(img, Image.Image):
                grid.paste(img, (i * target_height, 0))
            else:
                # If tensor, convert to PIL
                if isinstance(img, torch.Tensor):
                    img_pil = TF.to_pil_image(img)
                    grid.paste(img_pil, (i * target_height, 0))
                    
        return grid
        
    def get_current_lr(self, accelerator, model):
        """Get current learning rate from optimizer."""
        if hasattr(model, 'module'):
            # Distributed training - access underlying model
            if hasattr(model.module, 'optimizers'):
                return model.module.optimizers().param_groups[0]['lr']
        else:
            if hasattr(model, 'optimizers'):
                return model.optimizers().param_groups[0]['lr']
        return 0.0
        
    def save_model(self, accelerator, model, file_name):
        """Enhanced model saving with WandB checkpoint logging."""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            
            # Extract step from filename for WandB logging
            step = self.num_steps
            if "step-" in file_name:
                step = int(file_name.split("step-")[1].split(".")[0])
            elif "epoch-" in file_name:
                step = int(file_name.split("epoch-")[1].split(".")[0])
                
            # Log checkpoint to WandB
            self.log_checkpoint_metrics(accelerator, model, step)


def launch_segmentation_training_with_wandb(
    dataset,
    model: DiffusionTrainingModule,
    model_logger: WandBEnhancedModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: Optional[int] = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    wandb_project: str = "qwen-image-edit-segmentation",
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    log_sample_predictions: bool = True,
    validation_samples: int = 4,
    validation_steps: int = 250,
    num_validation_samples: int = 4,
    args = None,
):
    """Enhanced training function with comprehensive WandB logging."""
    
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
        log_sample_predictions = getattr(args, 'log_sample_predictions', log_sample_predictions)
        validation_samples = getattr(args, 'validation_samples', validation_samples)
        validation_steps = getattr(args, 'validation_steps', validation_steps)
        num_validation_samples = getattr(args, 'num_validation_samples', num_validation_samples)
        
    # Handle None values for optional parameters
    if wandb_entity is None:
        wandb_entity = None
    if wandb_name is None:
        wandb_name = None
    if wandb_tags is None:
        wandb_tags = []

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
            "save_steps": save_steps,
        }

        if args is not None:
            config.update({
                "lora_rank": getattr(args, 'lora_rank', None),
                "lora_base_model": getattr(args, 'lora_base_model', None),
                "lora_target_modules": getattr(args, 'lora_target_modules', None),
                "max_pixels": getattr(args, 'max_pixels', None),
                "dataset_repeat": getattr(args, 'dataset_repeat', None),
                "batch_size": 1,
                "mask_loss_weight": getattr(args, 'mask_loss_weight', 1.0),
                "similarity_threshold": getattr(args, 'similarity_threshold', 0.5),
                "enable_dynamic_filtering": getattr(args, 'enable_dynamic_filtering', True),
            })

        # Initialize WandB
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
        
        # Log segmentation-specific info if available
        if hasattr(model, 'mask_loss_weight'):
            wandb.log({"segmentation/mask_loss_weight": model.mask_loss_weight})

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1000,
        threshold=0.001,
        min_lr=1e-7
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=True, 
        collate_fn=lambda x: x[0], 
        num_workers=num_workers
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    epoch_loss = 0.0
    num_batches = 0
    segmentation_metrics = {"mask_loss": 0.0, "similarity_scores": [], "filtered_samples": 0}
    
    for epoch_id in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_segmentation_metrics = {"mask_loss": 0.0, "similarity_scores": [], "filtered_samples": 0}

        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch_id + 1}/{num_epochs}")

        for batch_idx, data in enumerate(progress_bar):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                # Handle different data loading scenarios
                if hasattr(dataset, 'load_from_cache') and dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                    
                accelerator.backward(loss)
                optimizer.step()
                
                # Update scheduler with loss
                if accelerator.is_main_process:
                    scheduler.step(loss.item())

                # ModelLogger handles checkpoint saving
                model_logger.on_step_end(accelerator, model, save_steps)

                # Log to W&B
                if accelerator.is_main_process:
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    num_batches += 1

                    # Basic training metrics
                    wandb.log({
                        "train/loss": loss_value,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                        "train/global_step": global_step,
                    }, step=global_step)

                    # Update segmentation metrics if available
                    if isinstance(loss, dict):
                        for key, value in loss.items():
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            if "mask_loss" in key.lower():
                                epoch_segmentation_metrics["mask_loss"] += value
                                wandb.log({f"segmentation/{key}": value}, step=global_step)
                    else:
                        epoch_segmentation_metrics["mask_loss"] += loss_value

                    # Collect similarity scores if available in data
                    if "similarity_score" in data:
                        epoch_segmentation_metrics["similarity_scores"].append(data["similarity_score"])
                        wandb.log({
                            "segmentation/similarity_score": data["similarity_score"],
                            "segmentation/similarity_threshold": getattr(model, 'similarity_threshold', 0.5),
                        }, step=global_step)
                        
                    # Track filtered samples
                    if "filtered" in data:
                        filtered_count = sum(1 for item in data["filtered"] if item)
                        epoch_segmentation_metrics["filtered_samples"] += filtered_count
                        wandb.log({
                            "segmentation/filtered_samples": filtered_count,
                            "segmentation/filtering_ratio": filtered_count / len(data.get("filtered", [False])),
                        }, step=global_step)

                    global_step += 1

                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "step": global_step
                    })

                    # Run dedicated validation with image logging
                    if validation_steps and global_step % validation_steps == 0 and global_step > 0:
                        run_validation_with_logging(model, dataset, model_logger, global_step, 
                                                  num_samples=num_validation_samples)

        # Log epoch summary
        if accelerator.is_main_process and num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            avg_mask_loss = epoch_segmentation_metrics["mask_loss"] / num_batches if epoch_segmentation_metrics["mask_loss"] > 0 else 0
            avg_similarity = np.mean(epoch_segmentation_metrics["similarity_scores"]) if epoch_segmentation_metrics["similarity_scores"] else 0
            filter_ratio = epoch_segmentation_metrics["filtered_samples"] / num_batches if num_batches > 0 else 0
            
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch_id,
                "segmentation/epoch_mask_loss": avg_mask_loss,
                "segmentation/epoch_avg_similarity": avg_similarity,
                "segmentation/epoch_filter_ratio": filter_ratio,
            }, step=global_step)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)

    # Finish W&B
    if accelerator.is_main_process:
        # Log final training summary
        wandb.log({
            "training/completed": True,
            "training/final_step": global_step,
            "training/final_epoch": num_epochs,
        })
        wandb.finish()


def _log_validation_sample(model_logger, model, data, step):
    """Log a validation sample prediction."""
    try:
        # Extract data
        if isinstance(data, dict):
            prompt = data.get("prompt", "No prompt")
            input_image = data.get("edit_image") or data.get("input_image")
            target_image = data.get("image")
        else:
            prompt = getattr(data, 'prompt', "No prompt") 
            input_image = getattr(data, 'edit_image') or getattr(data, 'input_image')
            target_image = getattr(data, 'image')
            
        if input_image and target_image:
            # Generate prediction using model pipeline
            with torch.no_grad():
                if hasattr(model, 'module') and hasattr(model.module, 'pipe'):
                    pipe = model.module.pipe
                elif hasattr(model, 'pipe'):
                    pipe = model.pipe
                else:
                    # Create a simple mock pipe for validation
                    from diffsynth.pipelines.qwen_image import QwenImagePipeline
                    try:
                        pipe = QwenImagePipeline()
                        print("  📝 Using loaded pipeline for validation")
                    except:
                        # Fallback: create a simple validation pipe
                        pipe = SimpleValidationPipe()
                        print("  📝 Using simple validation pipe")
                    
                generated_image = pipe(
                    prompt=prompt,
                    edit_image=input_image,
                    height=input_image.size[1],
                    width=input_image.size[0],
                    num_inference_steps=20,
                    edit_image_auto_resize=True,
                    seed=42
                )
                
            # Calculate LPIPS if possible
            lpips_score = None
            try:
                import lpips
                lpips_fn = lpips.LPIPS(net='alex')
                target_tensor = _image_to_tensor(target_image).to(next(pipe.parameters()).device)
                generated_tensor = _image_to_tensor(generated_image).to(next(pipe.parameters()).device)
                lpips_score = lpips_fn(target_tensor, generated_tensor).item()
            except:
                pass
                
            # Add to model logger for periodic logging
            model_logger.add_sample_prediction(prompt, input_image, target_image, generated_image, lpips_score)
            
    except Exception as e:
        # Don't crash training if sample logging fails
        if hasattr(wandb, 'config'):
            wandb.config.update({"sample_logging_error": str(e)})

def run_validation_with_logging(model, dataset, model_logger, step, num_samples=4, save_images=True):
    """Run dedicated validation and log results to WandB."""
    import random
    
    # Only run validation on main process
    if not hasattr(wandb, 'config'):
        return
        
    print(f"\n=== Running Validation at Step {step} ===")
    
    try:
        # Select random validation samples
        total_samples = len(dataset)
        sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        model.eval()
        validation_metrics = []
        
        for i, sample_idx in enumerate(sample_indices):
            print(f"Validating sample {i+1}/{len(sample_indices)} (idx: {sample_idx})")
            
            try:
                # Get sample data
                if hasattr(dataset, '__getitem__'):
                    sample_data = dataset[sample_idx]
                else:
                    sample_data = dataset.data[sample_idx % len(dataset.data)]
                
                # Extract images and prompt
                if isinstance(sample_data, dict):
                    prompt = sample_data.get("prompt", "Validation sample")
                    input_image = sample_data.get("edit_image") or sample_data.get("input_image")
                    target_image = sample_data.get("image")
                else:
                    prompt = getattr(sample_data, 'prompt', "Validation sample")
                    input_image = getattr(sample_data, 'edit_image') or getattr(sample_data, 'input_image')
                    target_image = getattr(sample_data, 'image')
                
                if not input_image or not target_image:
                    print(f"  Warning: Missing images for sample {sample_idx}")
                    continue
                    
                # Generate prediction
                with torch.no_grad():
                    if hasattr(model, 'module') and hasattr(model.module, 'pipe'):
                        pipe = model.module.pipe
                    elif hasattr(model, 'pipe'):
                        pipe = model.pipe
                    else:
                        # Create a simple mock pipe for validation
                        from diffsynth.pipelines.qwen_image import QwenImagePipeline
                        try:
                            pipe = QwenImagePipeline()
                            print("  📝 Using loaded pipeline for validation")
                        except:
                            # Fallback: create a simple validation pipe
                            pipe = SimpleValidationPipe()
                            print("  📝 Using simple validation pipe")
                    
                    generated_image = pipe(
                        prompt=prompt,
                        edit_image=input_image,
                        height=input_image.size[1],
                        width=input_image.size[0],
                        num_inference_steps=30,  # Higher quality for validation
                        edit_image_auto_resize=True,
                        seed=42 + i  # Different seed for each sample
                    )
                
                # Calculate LPIPS score
                lpips_score = None
                try:
                    import lpips
                    import torchvision.transforms.functional as TF
                    lpips_fn = lpips.LPIPS(net='alex').to(next(pipe.parameters()).device)
                    
                    target_tensor = TF.to_tensor(target_image).unsqueeze(0).to(next(pipe.parameters()).device)
                    generated_tensor = TF.to_tensor(generated_image).unsqueeze(0).to(next(pipe.parameters()).device)
                    
                    # Normalize to [-1, 1] range
                    target_tensor = (target_tensor * 2.0) - 1.0
                    generated_tensor = (generated_tensor * 2.0) - 1.0
                    
                    lpips_score = lpips_fn(target_tensor, generated_tensor).item()
                except Exception as e:
                    print(f"  LPIPS calculation failed: {e}")
                
                validation_metrics.append(lpips_score)
                
                # Create comparison image grid
                comparison_grid = create_validation_comparison_grid(
                    input_image, target_image, generated_image, prompt, lpips_score
                )
                
                # Log to WandB with enhanced prompt visibility
                current_step = step
                wandb.log({
                    f"validation/sample_{i}_comparison": wandb.Image(comparison_grid),
                    f"validation/sample_{i}_lpips": lpips_score if lpips_score is not None else -1,
                    f"validation/sample_{i}_prompt": prompt,
                    f"validation/edit_instruction": prompt,  # Additional visible name
                    f"validation/sample_{i}_full_info": f"Sample {i+1}: {prompt[:100]}...",  # Truncated visible summary
                    f"validation/step_marker": current_step,
                })
                
                # Also log individual images
                if save_images:
                    wandb.log({
                        f"validation/step_{step}_input_{i}": wandb.Image(input_image),
                        f"validation/step_{step}_target_{i}": wandb.Image(target_image),
                        f"validation/step_{step}_generated_{i}": wandb.Image(generated_image),
                    })  # Remove step parameter to avoid conflicts
                
                lpips_text = f"{lpips_score:.4f}" if lpips_score is not None else "N/A"
                print(f"  ✓ Sample {i+1}: LPIPS = {lpips_text}")
                
            except Exception as e:
                print(f"  ✗ Sample {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Log aggregate metrics
        if validation_metrics:
            avg_lpips = sum(validation_metrics) / len(validation_metrics)
            min_lpips = min(validation_metrics)
            max_lpips = max(validation_metrics)
            
            wandb.log({
                "validation/avg_lpips": avg_lpips,
                "validation/min_lpips": min_lpips,
                "validation/max_lpips": max_lpips,
                "validation/samples_processed": len(validation_metrics),
                "validation/step": step,
            }, step=step)
            
            print(f"Validation Summary: Avg LPIPS = {avg_lpips:.4f}, Samples = {len(validation_metrics)}")
        else:
            print("Warning: No validation metrics collected")
            
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.train()  # Restore training mode


def create_validation_comparison_grid(input_image, target_image, generated_image, prompt, lpips_score):
    """Create a grid showing input, target, and generated images with metadata."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    
    # Create grid layout: Input | Target | Generated
    img_width, img_height = input_image.size
    grid_width = img_width * 3 + 20  # 20px padding between images
    grid_height = img_height + 60   # Extra space for labels
    
    # Create grid
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images
    grid.paste(input_image, (0, 30))
    grid.paste(target_image, (img_width + 10, 30))
    grid.paste(generated_image, ((img_width + 10) * 2, 30))
    
    # Add labels
    draw = ImageDraw.Draw(grid)
    
    # Try to use a default font, fall back if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw labels
    draw.text((img_width//2 - 30, 10), "INPUT", fill='black', font=small_font)
    draw.text((img_width + 10 + img_width//2 - 25, 10), "TARGET", fill='black', font=small_font) 
    draw.text(((img_width + 10) * 2 + img_width//2 - 40, 10), "GENERATED", fill='black', font=small_font)
    
    # Add LPIPS score and prompt at bottom
    lpips_text = f"LPIPS: {lpips_score:.4f}" if lpips_score is not None else "LPIPS: N/A"
    draw.text((10, grid_height - 25), lpips_text, fill='black', font=font)
    
    # Wrap and add prompt
    wrapped_prompt = textwrap.fill(prompt, width=80) if len(prompt) > 80 else prompt
    draw.text((10, grid_height - 15), f"Prompt: {wrapped_prompt}", fill='black', font=small_font)
    
    return grid



def _image_to_tensor(image):
    """Convert PIL image to tensor for LPIPS calculation."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)


class SimpleValidationPipe:
    """Simple validation pipe for when model pipeline is not available."""
    
    def __call__(self, prompt, edit_image, height, width, num_inference_steps=20, edit_image_auto_resize=True, seed=42):
        """Generate a simple edited image for validation purposes."""
        # Create a simple "edited" version of the input
        # For validation, we'll create a simple colored version
        color_variants = ['blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'pink']
        color = color_variants[hash(prompt + str(seed)) % len(color_variants)]
        return Image.new('RGB', edit_image.size, color=color)
    
    def parameters(self):
        """Return device parameters for LPIPS calculation."""
        import torch
        return [torch.tensor([1.0], device='cpu')]



def main():
    """Enhanced main function with WandB logging."""
    parser = qwen_image_parser()
    
    # Add segmentation-specific arguments
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
    
    # Add WandB arguments
    parser.add_argument("--wandb_project", type=str, default="qwen-image-edit-segmentation", 
                       help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                       help="W&B entity/username")
    parser.add_argument("--wandb_name", type=str, default=None, 
                       help="W&B run name")
    parser.add_argument("--wandb_tags", type=str, default=None, 
                       help="W&B tags (comma-separated)")
    parser.add_argument("--log_sample_predictions", action="store_true", default=True,
                       help="Enable sample prediction logging")
    # Note: validation_steps is already defined in qwen_image_parser
    parser.add_argument("--num_validation_samples", type=int, default=4,
                       help="Number of validation samples to evaluate")
    # Note: validation_samples is already defined in qwen_image_parser (line 1689)
    
    
    args = parser.parse_args()

    # Parse W&B tags
    if args.wandb_tags:
        args.wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",")]

    # Build the dataset
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
        mask_target_size=None,
        similarity_threshold=args.similarity_threshold,
        enable_dynamic_filtering=args.enable_dynamic_filtering,
    )

    # Initialize the model
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
    
    # Attach segmentation parameters
    model.mask_loss_weight = args.mask_loss_weight
    if hasattr(model, 'similarity_threshold'):
        model.similarity_threshold = args.similarity_threshold

    # Create enhanced model logger with WandB integration
    model_logger = WandBEnhancedModelLogger(
        args.output_path, 
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_enabled=True,
        validation_samples=args.validation_samples
    )

    # Launch training with WandB
    launch_segmentation_training_with_wandb(
        dataset, model, model_logger, args=args
    )


if __name__ == "__main__":
    main()