import accelerate
import imageio, os, torch, warnings, torchvision, argparse, json
from ..utils import ModelConfig  # type: ignore
from ..models.utils import load_state_dict  # type: ignore
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from typing import Optional

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01, verbose=True):
        """
        Early stopping implementation for curriculum learning.
        
        Args:
            patience: Number of steps to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, current_loss):
        """
        Check if training should stop early.
        
        Args:
            current_loss: Current training loss
            
        Returns:
            bool: True if training should stop
        """
        if current_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = current_loss
            self.counter = 0
            if self.verbose:
                print(f"Loss improved to {current_loss:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} steps")
                
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} steps without improvement")
                
        return self.early_stop


class EarlyStopCurriculum:
    """
    Adaptive curriculum controller that automatically advances through difficulty phases
    based on validation performance convergence.
    """
    def __init__(self, phases, patience=2, min_delta_rel=0.005, ema_alpha=0.6):
        """
        Args:
            phases: List of phase configurations (tier mixes)
            patience: Number of validation checks without improvement before advancing
            min_delta_rel: Minimum relative improvement threshold (e.g., 0.005 = 0.5%)
            ema_alpha: EMA smoothing factor for noisy validation metrics
        """
        self.phases = phases
        self.patience = patience
        self.min_delta_rel = min_delta_rel
        self.ema_alpha = ema_alpha

        self.current_phase = 0
        self.best_score = float('inf')
        self.ema_score = None
        self.bad_epochs = 0
        self.is_final_phase = False

        # For compatibility with existing code
        self.phase_idx = 0

    def should_advance(self, current_score):
        """
        Check if curriculum should advance to next phase based on validation score.

        Args:
            current_score: Current validation LPIPS score

        Returns:
            bool: True if should advance to next phase
        """
        # Update EMA score
        if self.ema_score is None:
            self.ema_score = current_score
        else:
            self.ema_score = self.ema_alpha * current_score + (1 - self.ema_alpha) * self.ema_score

        # Check for improvement using EMA score
        # Handle initial case where best_score is infinity
        if self.best_score == float('inf') or self.ema_score < self.best_score - self.best_score * self.min_delta_rel:
            # Improvement detected
            self.best_score = self.ema_score
            self.bad_epochs = 0
            return False
        else:
            # No improvement
            self.bad_epochs += 1

            # Check if should advance
            if self.bad_epochs >= self.patience:
                if self.current_phase < len(self.phases) - 1:
                    # Advance to next phase
                    self.current_phase += 1
                    self.phase_idx = self.current_phase  # Keep in sync
                    self.best_score = float('inf')
                    self.ema_score = None
                    self.bad_epochs = 0
                    self.is_final_phase = (self.current_phase == len(self.phases) - 1)
                    return True
                else:
                    # Final phase - signal early stopping
                    self.is_final_phase = True
                    return False

            return False

    def get_current_mix(self):
        """Get current phase tier mix."""
        return self.phases[self.current_phase]

    def current_mix(self):
        """Alias for get_current_mix for compatibility."""
        return self.get_current_mix()

    def step(self, validation_score):
        """
        Process validation score and return curriculum decision.

        Args:
            validation_score: Current validation LPIPS score

        Returns:
            dict: Decision with action, phase info, and metrics
        """
        should_advance = self.should_advance(validation_score)

        if should_advance:
            return {
                "action": "advance",
                "phase": self.current_phase,
                "best": self.best_score,
                "ema": self.ema_score,
                "bad_epochs": self.bad_epochs
            }
        else:
            return {
                "action": "stay",
                "phase": self.current_phase,
                "best": self.best_score,
                "ema": self.ema_score,
                "bad_epochs": self.bad_epochs
            }

    def should_stop_training(self):
        """Check if training should stop (final phase plateaued)."""
        return self.is_final_phase and self.bad_epochs >= self.patience


class LPIPSEvaluator:
    def __init__(self, validation_dataset_or_csv, base_path="", device="cuda", lpips_net="alex"):
        """
        LPIPS evaluator for validation during curriculum training.

        Args:
            validation_dataset_or_csv: Either a ValidationDataset object or path to validation CSV file
            base_path: Base path for validation images (only used if CSV path provided)
            device: Device for LPIPS computation
            lpips_net: LPIPS network type ("alex", "vgg", "squeeze")
        """
        import lpips
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
        self.device = device
        self.base_path = base_path

        # Determine if we have a ValidationDataset or CSV path
        if isinstance(validation_dataset_or_csv, str):
            # CSV path provided - load validation data from CSV
            self.validation_data = pd.read_csv(validation_dataset_or_csv)
            self.validation_dataset = None
            print(f"Loaded {len(self.validation_data)} validation samples")
        else:
            print("Using provided ValidationDataset object...")
            # ValidationDataset object provided
            self.validation_dataset = validation_dataset_or_csv
            self.validation_data = None
            print(f"Using ValidationDataset with {len(self.validation_dataset)} validation samples")

    def evaluate_batch(self, model, validation_samples, batch_size=4, save_images=False, epoch=None, max_save=3, accelerator=None):
        """
        Evaluate a batch of validation samples and compute LPIPS scores.

        Args:
            model: The QwenImage model to evaluate
            validation_samples: List of validation sample indices to evaluate
            batch_size: Batch size for evaluation
            save_images: Whether to save validation images for inspection
            epoch: Current epoch number (for image naming)
            max_save: Maximum number of images to save per validation run
            accelerator: Accelerator object for distributed training (optional)

        Returns:
            float: Average LPIPS score for the batch
        """
        # import torchvision.transforms.functional as TF  # Unused

        model.eval()

        # Distribute validation samples across GPUs if using distributed training
        if accelerator is not None and accelerator.num_processes > 1:
            # Split validation samples across ranks
            rank = accelerator.process_index
            world_size = accelerator.num_processes

            # Distribute samples round-robin across ranks
            my_samples = [validation_samples[i] for i in range(rank, len(validation_samples), world_size)]
            if accelerator.is_main_process:
                print(f"Distributed validation: {world_size} GPUs, rank {rank} processing {len(my_samples)} samples")
        else:
            my_samples = validation_samples

        lpips_scores = []
        total_samples = len(my_samples)
        processed_samples = 0

        with torch.no_grad():
            for i in range(0, len(my_samples), batch_size):
                batch_indices = my_samples[i:i+batch_size]
                batch_lpips = []

                for idx in batch_indices:
                    processed_samples += 1
                    # Only print progress on rank 0 to avoid duplicate output in distributed training
                    try:
                        from accelerate import PartialState
                        if PartialState().is_main_process:
                            print(f"  Evaluating sample {processed_samples}/{total_samples}...", end='\r')
                    except:
                        # Fallback if accelerate not available or in single GPU mode
                        print(f"  Evaluating sample {processed_samples}/{total_samples}...", end='\r')
                    try:
                        if self.validation_dataset is not None:
                            # Using ValidationDataset - get pre-loaded images
                            sample = self.validation_dataset[idx]
                            # Fixed: edit_image is the input TO BE EDITED, image is the target result
                            input_image = sample['edit_image']  # Already a PIL Image
                            target_image = sample['image']  # Already a PIL Image
                            prompt = sample['prompt']
                        else:
                            # Using pandas DataFrame - load images from paths
                            row = self.validation_data.iloc[idx]

                            # Load input image
                            input_image_path = os.path.join(self.base_path, row['image'])
                            input_image = Image.open(input_image_path).convert('RGB')

                            # Load target (ground truth) image
                            target_image_path = os.path.join(self.base_path, row['edit_image'])
                            target_image = Image.open(target_image_path).convert('RGB')
                            prompt = row['prompt']

                        # Prepare data for model inference
                        # FIXED: input_image is the source to edit, target_image is desired result
                        data = {
                            "image": target_image,      # Target/ground truth (what we want to achieve)
                            "edit_image": input_image,  # Source image (what we want to edit)
                            "prompt": prompt
                        }

                        # Generate image using direct pipeline inference (bypass training preprocessing)
                        # Handle both regular model and DistributedDataParallel wrapped model
                        if hasattr(model, 'module'):
                            # Distributed training - access underlying model
                            pipe = model.module.pipe
                        else:
                            # Single GPU training
                            pipe = model.pipe

                        # FIXED: Use direct pipeline inference instead of training preprocessing
                        with torch.no_grad():
                            generated_image = pipe(
                                prompt=prompt,
                                edit_image=input_image,  # Source image to edit
                                height=input_image.size[1],
                                width=input_image.size[0],
                                num_inference_steps=40,  # Higher quality validation inference
                                edit_image_auto_resize=True,
                                seed=42  # Consistent validation results
                            )

                        # Convert images to tensors for LPIPS calculation
                        target_tensor = self._image_to_tensor(target_image).to(self.device)
                        generated_tensor = self._image_to_tensor(generated_image).to(self.device)

                        # Calculate LPIPS score
                        lpips_score = self.lpips_fn(target_tensor, generated_tensor).item()
                        batch_lpips.append(lpips_score)

                        # Save validation images if requested (only save first few samples)
                        if save_images and len(lpips_scores) + len(batch_lpips) <= max_save and epoch is not None:
                            self.save_validation_images(input_image, target_image, generated_image, prompt, idx, epoch)

                    except Exception as e:
                        import traceback
                        print(f"Error evaluating validation sample {idx}: {e}")
                        print(f"Exception type: {type(e).__name__}")
                        print(f"Full traceback:")
                        traceback.print_exc()
                        # If it's a ValidationDataset, also try to debug the sample
                        if self.validation_dataset is not None:
                            try:
                                print(f"Attempting to debug ValidationDataset sample {idx}...")
                                sample_debug = self.validation_dataset[idx]
                                print(f"Sample keys: {list(sample_debug.keys())}")
                                print(f"Sample types: {[(k, type(v).__name__) for k, v in sample_debug.items()]}")
                            except Exception as debug_e:
                                print(f"Failed to debug sample {idx}: {debug_e}")
                        continue

                if batch_lpips:
                    lpips_scores.extend(batch_lpips)

        # Clear the progress line (only on rank 0)
        try:
            from accelerate import PartialState
            if PartialState().is_main_process:
                print(" " * 50, end='\r')
        except:
            # Fallback if accelerate not available or in single GPU mode
            print(" " * 50, end='\r')

        model.train()

        # Calculate average LPIPS for this rank
        local_avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else float('inf')

        # Gather results from all GPUs if using distributed training
        if accelerator is not None and accelerator.num_processes > 1:
            # Convert to tensor for gathering
            local_avg_tensor = torch.tensor(local_avg_lpips, device=accelerator.device)
            # Gather all LPIPS scores from all ranks
            gathered_lpips = accelerator.gather(local_avg_tensor)
            # Calculate overall average (only meaningful on main process)
            if accelerator.is_main_process:
                # Filter out inf values in case some ranks had no samples
                valid_scores = [score.item() for score in gathered_lpips if score.item() != float('inf')]
                final_avg_lpips = sum(valid_scores) / len(valid_scores) if valid_scores else float('inf')
                if accelerator.num_processes > 1:
                    print(f"Distributed validation complete: {len(valid_scores)} ranks contributed scores")
                    print(f"Individual rank LPIPS: {[f'{score:.4f}' for score in valid_scores]}")
                print(f"Final averaged LPIPS: {final_avg_lpips:.6f}")
                return final_avg_lpips
            else:
                # Non-main processes return the gathered result (though it won't be used)
                return gathered_lpips[0].item()
        else:
            return local_avg_lpips

    def _inference_single_image(self, model, inputs):
        """
        DEPRECATED: This method is no longer used in validation.
        Validation now uses direct pipeline calls for proper inference.
        """
        raise NotImplementedError("This method is deprecated. Use direct pipeline calls in validation.")

    def save_validation_images(self, input_image, target_image, generated_image, prompt, sample_idx, epoch, save_dir="validation_outputs"):
        """
        Save validation images for visual inspection.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Create filename with epoch and sample info
        base_name = f"epoch_{epoch}_sample_{sample_idx}"

        # Save input image
        input_image.save(os.path.join(save_dir, f"{base_name}_input.png"))

        # Save target (ground truth) image
        target_image.save(os.path.join(save_dir, f"{base_name}_target.png"))

        # Save generated image
        generated_image.save(os.path.join(save_dir, f"{base_name}_generated.png"))

        # Save prompt as text file
        with open(os.path.join(save_dir, f"{base_name}_prompt.txt"), 'w') as f:
            f.write(prompt)

    def _image_to_tensor(self, image):
        """
        Convert PIL image to tensor for LPIPS calculation.
        """
        from torchvision import transforms

        # LPIPS expects images in [-1, 1] range
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # LPIPS works well with 256x256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
        ])

        return transform(image).unsqueeze(0)


class ValidationEarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, verbose=True, mode='min'):
        """
        Early stopping based on validation LPIPS scores.

        Args:
            patience: Number of validation checks to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print early stopping messages
            mode: 'min' for metrics where lower is better (like LPIPS)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, current_score):
        """
        Check if training should stop early based on validation score.

        Args:
            current_score: Current validation score (e.g., LPIPS)

        Returns:
            bool: True if training should stop
        """
        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta

        if improved:
            # Improvement detected
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                print(f"Validation score improved to {current_score:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"No validation improvement for {self.counter}/{self.patience} checks (current: {current_score:.6f}, best: {self.best_score:.6f})")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Validation early stopping triggered after {self.patience} checks without improvement")

        return self.early_stop




class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        # import torchvision.transforms.functional as TF  # Unused
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = TF.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = TF.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        # import torchvision.transforms.functional as TF  # Unused
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = TF.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = TF.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        # Increment by world_size to show global step numbers
        world_size = accelerator.num_processes if accelerator is not None else 1
        self.num_steps += world_size
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
def launch_training_task(
    dataset: torch.utils.data.Dataset,  # type: ignore
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
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
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # CHANGE 1: Disable shuffle for curriculum learning
    shuffle_setting = getattr(args, 'dataloader_shuffle', True) if args else True
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle_setting, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # CHANGE 2: Add early stopping variables
    early_stop_patience = getattr(args, 'early_stop_patience', 5) if args else 5
    early_stop_threshold = getattr(args, 'early_stop_threshold', 0.01) if args else 0.01
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch_id in range(num_epochs):
        epoch_loss = 0.0
        step_count = 0
        
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                
                # CHANGE 3: Track loss for early stopping
                loss_value = loss.item()
                epoch_loss += loss_value
                step_count += 1
                
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                
                # CHANGE 4: Check early stopping every save_steps
                if save_steps and step_count % save_steps == 0:
                    avg_loss = epoch_loss / step_count
                    if avg_loss < best_loss - early_stop_threshold:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping: no improvement for {early_stop_patience} evaluations")
                        model_logger.on_training_end(accelerator, model, save_steps)
                        return
                        
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def check_early_stopping(current_loss, best_loss, patience_counter, patience, min_delta):
    if current_loss < best_loss - min_delta:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1
    early_stop = patience_counter >= patience
    return best_loss, patience_counter, early_stop

def launch_training_task_curriculum(
    dataset: torch.utils.data.Dataset,  # type: ignore
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    early_stop_patience: int = 5,
    early_stop_threshold: float = 0.01,
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
        early_stop_patience = args.early_stop_patience
        early_stop_threshold = args.early_stop_threshold
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch_id in range(num_epochs):
        epoch_loss = 0.0
        step_count = 0
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                epoch_loss += loss.item()
                step_count += 1
                
                accelerator.backward(loss)
                optimizer.step()
                print(f"DEBUG: on_step_end called - step {step_count}, save_steps={save_steps}")
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                
                best_loss, patience_counter, early_stop = check_early_stopping(epoch_loss / step_count, best_loss, patience_counter, early_stop_patience, early_stop_threshold)
                if early_stop:
                    print(f"Early stopping: no improvement for {early_stop_patience} evaluations")
                    model_logger.on_training_end(accelerator, model, save_steps)
                    return
                
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task(
    dataset: torch.utils.data.Dataset,  # type: ignore
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
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
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_ranking_curriculum_training(
    ranking_csv_path: str,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    # curriculum_schedule: str = "50:1.0,0.0,0.0,0.0;50:0.8,0.2,0.0,0.0;50:0.8,0.0,0.2,0.0;50:0.8,0.0,0.0,0.2",  # Not used in adaptive curriculum
    curriculum_batch_size: int = 32,
    curriculum_steps_per_epoch: int = 0,
    sort_within_tier: bool = False,
    curriculum_seed: int = 42,
    base_path: str = "",
    max_pixels: int = 1024*1024,
    height: int = None,
    width: int = None,
    # New validation parameters
    validation_csv_path: str = None,
    validation_steps: Optional[int] = None,
    validation_patience: int = 3,
    validation_min_delta: float = 0.001,
    validation_batch_size: int = 4,
    validation_samples: int = 50,
    max_epochs_per_phase: int = 10,
    args = None,
):
    """
    Launch ranking-based curriculum training with validation-based early stopping.

    This function creates a curriculum sampler that progressively introduces harder
    samples according to a predefined schedule, with each phase continuing until
    validation LPIPS scores converge.
    """
    import math, os, random
    import pandas as pd
    from diffsynth.curriculum import build_curriculum_pools, CurriculumSampler, DistributedCurriculumSampler  # type: ignore
    from diffsynth.ranking_dataset import RankingDataset  # type: ignore

    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        # curriculum_schedule = args.curriculum_schedule  # Not used in adaptive curriculum
        curriculum_batch_size = args.curriculum_batch_size
        curriculum_steps_per_epoch = args.curriculum_steps_per_epoch
        sort_within_tier = args.sort_within_tier
        curriculum_seed = args.curriculum_seed
        base_path = args.dataset_base_path
        max_pixels = args.max_pixels
        height = args.height
        width = args.width
        # Get validation parameters from args
        validation_csv_path = getattr(args, 'validation_csv_path', validation_csv_path)
        validation_steps = getattr(args, 'validation_steps', validation_steps)
        validation_patience = getattr(args, 'validation_patience', validation_patience)
        validation_min_delta = getattr(args, 'validation_min_delta', validation_min_delta)
        validation_batch_size = getattr(args, 'validation_batch_size', validation_batch_size)
        validation_samples = getattr(args, 'validation_samples', validation_samples)
        max_epochs_per_phase = getattr(args, 'max_epochs_per_phase', max_epochs_per_phase)
    
    if validation_steps is None:
        validation_steps = curriculum_steps_per_epoch // 2
    # Load ranking dataset
    print(f"Loading ranking dataset from: {ranking_csv_path}")
    dataset = RankingDataset(
        ranking_csv_path=ranking_csv_path,
        base_path=base_path,
        max_pixels=max_pixels,
        height=height,
        width=width,
    )

    print(f"Dataset statistics:")
    stats = dataset.get_tier_stats()
    for tier, count in stats["tier_counts"].items():
        print(f"  Tier {tier}: {count} samples")

    # Load validation dataset if provided
    validation_evaluator = None
    if validation_csv_path and os.path.exists(validation_csv_path):
        print(f"Loading validation dataset from: {validation_csv_path}")
        from diffsynth.ranking_dataset import create_validation_dataset  # type: ignore
        # Use the CSV file's directory as base path for relative image paths
        validation_base_path = os.path.dirname(validation_csv_path)
        validation_dataset = create_validation_dataset(
            validation_csv_path=validation_csv_path,
            base_path=validation_base_path,
            max_pixels=1024*1024,  # Match training config
            height=None,
            width=None,
        )
        # Use the same device as the training model for validation
        # In distributed training, each process gets its own GPU
        model_device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validation_evaluator = LPIPSEvaluator(
            validation_dataset_or_csv=validation_dataset,
            device=model_device
        )
        print(f"Validation evaluation will run every {validation_steps} steps on {validation_samples} samples")

        # Test validation dataset integrity
        print("Testing validation dataset integrity...")
        test_indices = [0, 1, 2]  # Test first few samples
        for test_idx in test_indices:
            try:
                print(f"Testing validation sample {test_idx}...")
                sample = validation_dataset[test_idx]
                print(f"  Sample keys: {list(sample.keys())}")
                print(f"  Sample types: {[(k, type(v).__name__) for k, v in sample.items()]}")
                if 'image_path' in sample and 'edit_image_path' in sample:
                    print(f"  Image path: {sample['image_path']}")
                    print(f"  Edit image path: {sample['edit_image_path']}")
                print(f"  Sample {test_idx} OK")
            except Exception as e:
                print(f"  Sample {test_idx} FAILED: {e}")
                import traceback
                traceback.print_exc()
                break
        print("Validation dataset integrity test completed.")
    else:
        print("Warning: No validation dataset provided. Training will use fixed epoch schedule.")

    # Build curriculum pools
    df = pd.read_csv(ranking_csv_path)
    pools = build_curriculum_pools(df, tier_col="tier", difficulty_col="difficulty",
                                   sort_within_tier=sort_within_tier)

    # Define adaptive curriculum phases - more gradual progression
    phases = [
        {0: 0.8, 1: 0.2, 2: 0.0, 3: 0.0},  # Phase 1: Start with some tier 1 diversity
        {0: 0.6, 1: 0.3, 2: 0.1, 3: 0.0},  # Phase 2: Introduce tier 2 earlier
        {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},  # Phase 3: Introduce tier 3 earlier
        {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.1},  # Phase 4: More balanced mix
        {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},  # Phase 5: Equal mix of all tiers
    ]

    print(f"Adaptive curriculum: {len(phases)} phases")
    for i, phase_mix in enumerate(phases):
        print(f"  Phase {i+1}: mix = {phase_mix}")

    # Create adaptive curriculum controller
    curriculum_controller = EarlyStopCurriculum(
        phases=phases,
        patience=validation_patience,  # Use validation patience (default 3)
        min_delta_rel=0.005,  # 0.5% relative improvement threshold
        ema_alpha=0.6,  # Smooth noisy LPIPS scores
    )

    print(f"Curriculum controller: patience={curriculum_controller.patience}, min_delta_rel={curriculum_controller.min_delta_rel}")
    print(f"Starting with phase 1 mix: {curriculum_controller.current_mix()}")

    # Calculate steps per epoch
    if curriculum_steps_per_epoch <= 0:
        curriculum_steps_per_epoch = math.ceil(len(dataset) / curriculum_batch_size)


    # Set up distributed training if applicable
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    # Create dummy schedule for sampler initialization (will be overridden by adaptive controller)
    dummy_schedule = [{"epochs": 999, "mix": curriculum_controller.current_mix()}]

    if world_size > 1:
        print(f"Using DistributedCurriculumSampler (rank {rank}/{world_size})")
        sampler = DistributedCurriculumSampler(
            pools=pools,
            schedule=dummy_schedule,
            batch_size=curriculum_batch_size,
            steps_per_epoch=curriculum_steps_per_epoch,
            world_size=world_size,
            rank=rank,
            seed=curriculum_seed
        )
    else:
        print("Using CurriculumSampler (single GPU)")
        sampler = CurriculumSampler(
            pools=pools,
            schedule=dummy_schedule,
            batch_size=curriculum_batch_size,
            steps_per_epoch=curriculum_steps_per_epoch,
            seed=curriculum_seed
        )

    # Set initial curriculum mix
    sampler.set_mix_dict(curriculum_controller.current_mix())  # type: ignore

    # Create dataloader with curriculum sampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=curriculum_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Allow uneven batches to get full epoch coverage
        collate_fn=lambda x: x[0],  # Compatibility with DiffSynth training
    )

    # Set up optimizer and scheduler BEFORE accelerator.prepare to avoid DDP wrapping issues
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Set up training - use simpler approach like regular DiffSynth to avoid CUDA OOM
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model = accelerator.prepare(model)
    # Adaptive curriculum training loop
    print(f"\n{'='*60}")
    print("STARTING ADAPTIVE CURRICULUM TRAINING")
    print(f"{'='*60}")

    total_epochs = 0
    training_complete = False
    advance_curriculum = False  # Flag for step-based curriculum advancement
    total_step_counter = 0
    while not training_complete:
        # Update curriculum for this epoch
        sampler.set_epoch(total_epochs)

        current_mix = curriculum_controller.current_mix()
        print(f"\nEpoch {total_epochs + 1} - Phase {curriculum_controller.phase_idx + 1}/{len(curriculum_controller.phases)}")
        print(f"Current tier mix: {current_mix}")

        # Train one epoch
        model.train()
        epoch_loss = 0.0
        step_count = 0

        # Create progress bar that shows global steps
        pbar = tqdm(total=curriculum_steps_per_epoch, desc=f"Epoch {total_epochs + 1}")
        global_step_counter = 0

        for batch_idx, data in enumerate(dataloader):
            # Zero gradients at start of each accumulation cycle
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            loss = model(data)
            epoch_loss += loss.item()
            step_count += 1
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            # Update progress bar with global step increment
            world_size = accelerator.num_processes
            global_step_counter += accelerator.num_processes
            total_step_counter += accelerator.num_processes
            pbar.update(world_size)
            # Update progress bar with comprehensive information
            postfix_info: dict[str, str | int] = {"total_step": total_step_counter}
            # Add curriculum information (curriculum_controller is always available in this training loop)
            postfix_info["bad_val"] = f"{curriculum_controller.bad_epochs}/{curriculum_controller.patience}"
            postfix_info["phase"] = f"{curriculum_controller.current_phase + 1}/{len(curriculum_controller.phases)}"
            pbar.set_postfix(postfix_info)
            # Step optimizer at end of each accumulation cycle
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model_logger.on_step_end(accelerator, model, save_steps)

            # Run validation every validation_steps (global steps)
            if validation_evaluator is not None and total_step_counter % validation_steps == 0:
                print(f"\nRunning validation at global step {total_step_counter}...")

                model.eval()
                # Get total validation samples based on data source type
                if validation_evaluator.validation_dataset is not None:
                    total_val_samples = len(validation_evaluator.validation_dataset)
                elif validation_evaluator.validation_data is not None:
                    total_val_samples = len(validation_evaluator.validation_data)
                else:
                    raise ValueError("No validation data available in evaluator")
                val_indices = random.sample(range(total_val_samples),
                                            min(validation_samples, total_val_samples))

                # Evaluate LPIPS score (lower is better)
                with torch.no_grad():
                    # Only save images on rank 0 to avoid duplicates from each GPU
                    should_save_images = accelerator.is_main_process if hasattr(accelerator, 'is_main_process') else True
                    val_lpips = validation_evaluator.evaluate_batch(
                        model, val_indices, validation_batch_size,
                        save_images=should_save_images, epoch=total_epochs+1, max_save=1,
                        accelerator=accelerator
                    )

                # CRITICAL: Ensure scheduler is reset to training mode after validation
                model.train()
                # Reset scheduler timesteps for training - validation sets it to 20 steps
                # but training needs the full num_train_timesteps range
                if hasattr(model, 'module'):
                    # Distributed training - access underlying model
                    model.module.pipe.scheduler.set_timesteps(1000, training=True)
                else:
                    # Single GPU training
                    model.pipe.scheduler.set_timesteps(1000, training=True)

                if val_lpips is not None:
                    print(f"Step {global_step_counter} validation LPIPS: {val_lpips:.6f}")

                    # Update curriculum controller with validation score
                    decision = curriculum_controller.step(val_lpips)

                    if decision.get("advance"):
                        print(f"Step-based validation: Advancing to next phase (LPIPS: {val_lpips:.6f})")
                        if curriculum_controller.phase_idx < len(curriculum_controller.phases) - 1:
                            advance_curriculum = True
                        else:
                            print("All curriculum phases completed via step-based validation!")
                            training_complete = True
                else:
                    print(f"Step {global_step_counter}: Validation failed - continuing training")

        # Close progress bar
        pbar.close()

        avg_loss = epoch_loss / step_count if step_count > 0 else 0.0
        print(f"Epoch {total_epochs + 1} average loss: {avg_loss:.6f}")

        # Note: Validation now runs based on validation_steps during training (step-based)
        # Check if we need to advance curriculum or stop training based on step-based decisions
        if advance_curriculum:
            # Advance curriculum based on step-based validation
            if curriculum_controller.phase_idx < len(curriculum_controller.phases) - 1:
                new_mix = curriculum_controller.current_mix()
                sampler.set_mix_dict(new_mix)  # type: ignore
                print(f"[Step-based Curriculum] Advanced to phase {curriculum_controller.phase_idx + 1}: mix={new_mix}")
                advance_curriculum = False  # Reset flag
            else:
                print("All curriculum phases completed!")
                training_complete = True

        # Fallback: No validation provided
        if validation_evaluator is None:
            # No validation - check max epochs per phase fallback
            if total_epochs >= max_epochs_per_phase:
                print(f"No validation provided. Stopping after {max_epochs_per_phase} epochs (max_epochs_per_phase).")
                training_complete = True

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, total_epochs)

        total_epochs += 1

        # Safety check to prevent infinite training
        if total_epochs >= 1000:
            print("Warning: Reached 1000 epochs safety limit. Stopping training.")
            training_complete = True

    model_logger.on_training_end(accelerator, model, save_steps)
    print(f"\nRanking curriculum training completed! Total epochs: {total_epochs}")


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,  # type: ignore
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")

    # Ranking-based curriculum learning arguments
    parser.add_argument("--ranking_csv", type=str, default=None, help="Path to ranking CSV with tier information for curriculum learning.")
    parser.add_argument("--curriculum_schedule", type=str, default="50:1.0,0.0,0.0,0.0;50:0.8,0.2,0.0,0.0;50:0.8,0.0,0.2,0.0;50:0.8,0.0,0.0,0.2",
                        help="Curriculum schedule as 'epochs:p0,p1,p2,p3;...' format for tier mixing ratios. Each phase trains for specified epochs or until early stopping.")
    parser.add_argument("--curriculum_batch_size", type=int, default=32, help="Batch size for curriculum sampling.")
    parser.add_argument("--curriculum_steps_per_epoch", type=int, default=0, help="Steps per epoch for curriculum (0 = auto-calculate).")
    parser.add_argument("--sort_within_tier", default=False, action="store_true", help="Sort samples by difficulty within each tier.")
    parser.add_argument("--curriculum_seed", type=int, default=42, help="Random seed for curriculum sampling.")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience for curriculum stages.")
    parser.add_argument("--early_stop_threshold", type=float, default=0.01, help="Early stopping threshold for curriculum stages.")

    # Validation-based early stopping arguments
    parser.add_argument("--validation_csv_path", type=str, default=None, help="Path to validation CSV for LPIPS-based early stopping.")
    parser.add_argument("--validation_steps", type=int, default=None, help="Steps between validation evaluations.")
    parser.add_argument("--validation_patience", type=int, default=3, help="Validation patience for curriculum phase early stopping.")
    parser.add_argument("--validation_min_delta", type=float, default=0.001, help="Minimum LPIPS improvement for validation early stopping.")
    parser.add_argument("--validation_batch_size", type=int, default=4, help="Batch size for validation evaluation.")
    parser.add_argument("--validation_samples", type=int, default=50, help="Number of validation samples to evaluate each time.")
    parser.add_argument("--max_epochs_per_phase", type=int, default=10, help="Maximum epochs per curriculum phase before forced transition.")

    return parser
