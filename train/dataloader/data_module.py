from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import os
import random
import torchvision.transforms.functional as F
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL
from pipelines.tokenize import tokenize_prompt, encode_prompt
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


# TODO: Flipped versions of the data are currently disabled to save disk space.
# When more disk space is available, re-enable flipped versions by:
# 1. Uncommenting the flipped version code in _process_single_item
# 2. Updating the _precompute_and_save method to check for flipped versions
# 3. Updating the __getitem__ method to randomly choose between original and flipped versions


class EditDataset(Dataset):
    def __init__(self, path, metadata_file, width_resize, height_resize, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, preprocess=False):
        """
        Dataset for image editing tasks with precomputed latents and embeddings.

        Args:
            path (str or Path): Path to the dataset split directory (e.g., train, val, test).
            metadata_file (str or Path): Path to the metadata JSON file corresponding to the split.
            vae: Pretrained VAE model for encoding images to latents.
            tokenizer: Tokenizer for the first set of prompts.
            tokenizer_2: Tokenizer for the second set of prompts.
            text_encoder: Text encoder for the first tokenizer.
            text_encoder_2: Text encoder for the second tokenizer.
            device: Device to run computations (e.g., 'cuda' or 'cpu').
            preprocess (bool): Whether to preprocess and save latents/embeddings.
        """
        # TODO: Re-enable flipped versions when disk space is available
        self.use_flipped_versions = False  # Flag to control whether to use flipped versions
        
        self.data_dir = Path(path)
        self.metadata_file = Path(metadata_file)
        
        # Store original device but keep models on CPU until needed
        self.device = device
        self.vae = vae.to("cpu")
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder.to("cpu")
        self.text_encoder_2 = text_encoder_2.to("cpu")
        
        self.width_resize = width_resize
        self.height_resize = height_resize
        self.preprocess = preprocess

        # Load metadata
        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)

        if self.preprocess:
            self._precompute_and_save()

    def _precompute_and_save(self):
        current_rank = torch.distributed.get_rank()
        metadata_per_rank = self.metadata[current_rank::torch.distributed.get_world_size()]
        
        # Process in smaller batches to save memory
        import gc
        
        # Process in batches of 10 items
        batch_size = 10
        # Limit to 10 batches for testing
        max_batches = 10
        total_batches = min(max_batches, (len(metadata_per_rank) + batch_size - 1) // batch_size)
        
        for batch_idx in range(0, min(max_batches * batch_size, len(metadata_per_rank)), batch_size):
            batch_items = metadata_per_rank[batch_idx:batch_idx + batch_size]
            
            print(f"Processing batch {batch_idx//batch_size + 1}/{total_batches}")
            
            for item in tqdm(batch_items, desc=f"Rank {current_rank} Precomputing batch {batch_idx//batch_size + 1}"):
                input_image_path = self.data_dir / item["input_image"]
                output_image_path = self.data_dir / item["output_image"]

                # Paths for original and augmented data
                latent_data_path = input_image_path.with_suffix(f".latent_data_{self.width_resize}_{self.height_resize}.pt")
                
                # TODO: Re-enable flipped versions when disk space is available
                latent_data_flipped_path = None
                if self.use_flipped_versions:
                    latent_data_flipped_path = input_image_path.with_suffix(f".latent_data_{self.width_resize}_{self.height_resize}_flipped.pt")
                    # Skip if both original and flipped latent data exist
                    if latent_data_path.exists() and latent_data_flipped_path.exists():
                        continue
                else:
                    # Skip if original latent data exists
                    if latent_data_path.exists():
                        continue
                    
                try:
                    # Process this item
                    self._process_single_item(item, latent_data_path, latent_data_flipped_path)
                except Exception as e:
                    print(f"Error processing item {item}: {e}")
                    continue
            
            # Clear cache after each batch to prevent memory buildup
            torch.cuda.empty_cache()
            gc.collect()
            
        print(f"Processed {total_batches} batches out of {(len(metadata_per_rank) + batch_size - 1) // batch_size} total batches")
    
    def _process_single_item(self, item, latent_data_path, latent_data_flipped_path):
        # Remove any existing latent data
        for path in latent_data_path.parent.glob(f"{latent_data_path.stem}.latent_data*.pt"):
            path.unlink()

        input_image_path = self.data_dir / item["input_image"]
        output_image_path = self.data_dir / item["output_image"]
        
        # Move all models to CPU for preprocessing to avoid GPU memory issues
        device = "cpu"
        vae_cpu = self.vae.to(device)
        text_encoder_cpu = self.text_encoder.to(device)
        text_encoder_2_cpu = self.text_encoder_2.to(device)
        
        try:
            # Load and preprocess images
            input_image = Image.open(input_image_path).convert("RGB")
            output_image = Image.open(output_image_path).convert("RGB")
            
            # Resize images
            input_image = input_image.resize((self.width_resize, self.height_resize), Image.LANCZOS)
            output_image = output_image.resize((self.width_resize, self.height_resize), Image.LANCZOS)
            
            # TODO: Re-enable flipped versions when disk space is available
            input_image_flipped = None
            output_image_flipped = None
            if self.use_flipped_versions and latent_data_flipped_path is not None:
                input_image_flipped = input_image.transpose(Image.FLIP_LEFT_RIGHT)
                output_image_flipped = output_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Convert to tensors
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            
            # Process on CPU
            input_tensor = transform(input_image).unsqueeze(0).to(device)
            output_tensor = transform(output_image).unsqueeze(0).to(device)
            
            input_tensor_flipped = None
            output_tensor_flipped = None
            if self.use_flipped_versions and input_image_flipped is not None:
                input_tensor_flipped = transform(input_image_flipped).unsqueeze(0).to(device)
                output_tensor_flipped = transform(output_image_flipped).unsqueeze(0).to(device)
            
            # Encode images to latent space on CPU
            with torch.no_grad():
                cond_input = vae_cpu.encode(input_tensor).latent_dist.sample()
                model_input = vae_cpu.encode(output_tensor).latent_dist.sample()
                
                cond_input_flipped = None
                model_input_flipped = None
                if self.use_flipped_versions and input_tensor_flipped is not None:
                    cond_input_flipped = vae_cpu.encode(input_tensor_flipped).latent_dist.sample()
                    model_input_flipped = vae_cpu.encode(output_tensor_flipped).latent_dist.sample()

            # VAE scale factor
            vae_scale_factor = 2 ** (len(vae_cpu.config.block_out_channels) - 1)

            # Tokenize prompts
            tokens_one = tokenize_prompt(self.tokenizer, item["edit_instruction"], max_sequence_length=77)
            tokens_two = tokenize_prompt(self.tokenizer_2, item["edit_instruction"], max_sequence_length=256)
            
            # Process text on CPU
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders=[text_encoder_cpu, text_encoder_2_cpu],
                    tokenizers=[None, None],
                    text_input_ids_list=[tokens_one, tokens_two],
                    max_sequence_length=256,
                    prompt=item["edit_instruction"],
                )

            # Save processed data
            torch.save({
                "model_input": model_input.detach().cpu(),
                "cond_input": cond_input.detach().cpu(),
                "prompt_embeds": prompt_embeds.detach().cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
                "text_ids": text_ids.detach().cpu(),
                "vae_scale_factor": vae_scale_factor,
            }, latent_data_path)

            # TODO: Re-enable flipped versions when disk space is available
            if self.use_flipped_versions and latent_data_flipped_path is not None and model_input_flipped is not None:
                torch.save({
                    "model_input": model_input_flipped.detach().cpu(),
                    "cond_input": cond_input_flipped.detach().cpu(),
                    "prompt_embeds": prompt_embeds.detach().cpu(),
                    "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
                    "text_ids": text_ids.detach().cpu(),
                    "vae_scale_factor": vae_scale_factor,
                }, latent_data_flipped_path)
        finally:
            # Move models back to original device
            self.vae.to(self.device)
            self.text_encoder.to(self.device)
            self.text_encoder_2.to(self.device)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        input_image_path = self.data_dir / item["input_image"]
        
        # Path for original latent data
        latent_data_path = input_image_path.with_suffix(f".latent_data_{self.width_resize}_{self.height_resize}.pt")
        
        # Load original latent data
        if latent_data_path.exists():
            latent_data = torch.load(latent_data_path)
            return latent_data
        else:
            # If latent data doesn't exist, process it on the fly
            print(f"Warning: Latent data not found for {input_image_path}, processing on the fly")
            output_image_path = self.data_dir / item["output_image"]
            
            # Process on CPU to save memory
            device = "cpu"
            
            # Load and preprocess images
            input_image = Image.open(input_image_path).convert("RGB")
            output_image = Image.open(output_image_path).convert("RGB")
            
            # Resize images
            input_image = input_image.resize((self.width_resize, self.height_resize), Image.LANCZOS)
            output_image = output_image.resize((self.width_resize, self.height_resize), Image.LANCZOS)
            
            # Convert to tensors
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            
            input_tensor = transform(input_image).unsqueeze(0).to(device)
            output_tensor = transform(output_image).unsqueeze(0).to(device)
            
            # Encode images to latent space
            with torch.no_grad():
                cond_input = self.vae.encode(input_tensor).latent_dist.sample()
                model_input = self.vae.encode(output_tensor).latent_dist.sample()
            
            # VAE scale factor
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            
            # Tokenize prompts
            tokens_one = tokenize_prompt(self.tokenizer, item["edit_instruction"], max_sequence_length=77)
            tokens_two = tokenize_prompt(self.tokenizer_2, item["edit_instruction"], max_sequence_length=256)
            
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders=[self.text_encoder, self.text_encoder_2],
                    tokenizers=[None, None],
                    text_input_ids_list=[tokens_one, tokens_two],
                    max_sequence_length=256,
                    prompt=item["edit_instruction"],
                )
            
            return {
                "model_input": model_input.detach(),
                "cond_input": cond_input.detach(),
                "prompt_embeds": prompt_embeds.detach(),
                "pooled_prompt_embeds": pooled_prompt_embeds.detach(),
                "text_ids": text_ids.detach(),
                "vae_scale_factor": vae_scale_factor,
            }

    def __len__(self):
        return len(self.metadata)

class EditDatasetVal(Dataset):
    def __init__(self, path, metadata_file, width_resize, height_resize):
        """
        Dataset for image editing tasks.

        Args:
            path (str or Path): Path to the dataset split directory (e.g., train, val, test).
            metadata_file (str or Path): Path to the metadata JSON file corresponding to the split.
            min_resize_res (int): Minimum resize resolution for images.
            max_resize_res (int): Maximum resize resolution for images.
            crop_res (int): Crop resolution for images.
            flip_prob (float): Probability of horizontal flipping during augmentation.
        """
        self.data_dir = Path(path)
        self.metadata_file = Path(metadata_file)
        self.height = height_resize
        self.width = width_resize
        # Load metadata
        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)
    
    def paired_transform(self, input_image, output_image, normalize=True):
        input_image = F.resize(input_image, (self.width, self.height))
        output_image = F.resize(output_image, (self.width, self.height))

        input_image = F.to_tensor(input_image)
        output_image = F.to_tensor(output_image)

        if normalize:
            input_image = 2.0 * input_image - 1.0
            output_image = 2.0 * output_image - 1.0

        return input_image, output_image
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict: A dictionary containing:
                - "input_image": Transformed input image tensor.
                - "output_image": Transformed output image tensor.
                - "edit_instruction": Tokenized edit instruction.
        """
        item = self.metadata[idx]

        # Load and transform input and output images
        input_image_path = self.data_dir / item["input_image"]
        output_image_path = self.data_dir / item["output_image"]

        input_image = Image.open(input_image_path).convert("RGB")
        output_image = Image.open(output_image_path).convert("RGB")
        input_image, output_image = self.paired_transform(input_image, output_image, normalize=False)

        edit_instruction = item["edit_instruction"]

        return {
            "input_image": input_image,
            "output_image": output_image,
            "edit_instruction": edit_instruction,
        }
    
class FLUXDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, data_dir, model_name, image_size : tuple = (512, 512), valid_test_res=512):
        """
        Data module for FLUX training with latent preprocessing.

        Args:
            batch_size (int): Batch size for training and validation.
            num_workers (int): Number of data loading workers.
            data_dir (str or Path): Path to the dataset.
            latent_dir (str or Path): Directory to save precomputed latents.
            valid_test_res (int): Resize resolution for validation/testing.
            flip_prob (float): Probability of applying random horizontal flip.
        """
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.width_resize = image_size[0]
        self.height_resize = image_size[1]

        self.valid_test_res = valid_test_res
        self.model_name = model_name

        
    def _load_models(self, ckpt_name):
        models = {}
        model_components = ["vae", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2"]

        # Determine which cache directory to use
        cache_dir = "/tmp/huggingface"
        if os.path.exists("/workspace/hf_cache"):
            cache_dir = "/workspace/hf_cache"
        
        print(f"Loading models from {ckpt_name} using cache directory: {cache_dir}")
        
        try:
            # Try loading with DiffusionPipeline which can handle various model architectures
            from diffusers import DiffusionPipeline
            import torch
            
            print(f"Attempting to load {ckpt_name} as a DiffusionPipeline...")
            pipe = DiffusionPipeline.from_pretrained(
                ckpt_name,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16"
            )
            
            # Extract components from the pipeline
            models["vae"] = pipe.vae
            models["text_encoder"] = pipe.text_encoder
            models["tokenizer"] = pipe.tokenizer
            models["text_encoder_2"] = pipe.text_encoder_2
            models["tokenizer_2"] = pipe.tokenizer_2
            
            print("Successfully loaded model as DiffusionPipeline")
            
        except Exception as e:
            print(f"Failed to load as DiffusionPipeline: {e}")
            
            # Try loading with FluxPipeline specifically
            try:
                print(f"Attempting to load {ckpt_name} as a FluxPipeline...")
                # First check if FluxPipeline is available
                try:
                    from diffusers import FluxPipeline
                    import torch
                    
                    pipe = FluxPipeline.from_pretrained(
                        ckpt_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        variant="bf16"
                    )
                    
                    # Extract components from the pipeline
                    models["vae"] = pipe.vae
                    models["text_encoder"] = pipe.text_encoder
                    models["tokenizer"] = pipe.tokenizer
                    models["text_encoder_2"] = pipe.text_encoder_2
                    models["tokenizer_2"] = pipe.tokenizer_2
                    
                    print("Successfully loaded model as FluxPipeline")
                except ImportError:
                    print("FluxPipeline not available in this version of diffusers")
                    raise
                    
            except Exception as flux_error:
                print(f"Failed to load as FluxPipeline: {flux_error}")
                print("Falling back to loading individual components...")
                
                # Fall back to loading individual components
                for component in model_components:
                    try:
                        model_class = {
                            "vae": AutoencoderKL,
                            "text_encoder": CLIPTextModel,
                            "tokenizer": CLIPTokenizer,
                            "text_encoder_2": T5EncoderModel,
                            "tokenizer_2": T5Tokenizer,
                        }[component]

                        # Load each component using the corresponding subfolder and explicit cache_dir
                        models[component] = model_class.from_pretrained(
                            ckpt_name, 
                            subfolder=component,
                            cache_dir=cache_dir,
                            use_safetensors=True
                        )
                    except Exception as component_error:
                        print(f"Error loading {component}: {component_error}")
                        
                        # Try manual download as a last resort
                        try:
                            print(f"Attempting manual download for {component}...")
                            from huggingface_hub import hf_hub_download
                            import os
                            
                            # Map components to their typical filenames
                            filename_map = {
                                "vae": "diffusion_pytorch_model.safetensors",
                                "text_encoder": "model.safetensors",
                                "tokenizer": "tokenizer_config.json",
                                "text_encoder_2": "model.safetensors",
                                "tokenizer_2": "tokenizer_config.json"
                            }
                            
                            # Create component directory
                            component_dir = os.path.join(cache_dir, ckpt_name, component)
                            os.makedirs(component_dir, exist_ok=True)
                            
                            # Download the file
                            file_path = hf_hub_download(
                                repo_id=ckpt_name,
                                filename=f"{component}/{filename_map[component]}",
                                cache_dir=cache_dir
                            )
                            
                            # Try loading again
                            models[component] = model_class.from_pretrained(
                                component_dir,
                                cache_dir=cache_dir,
                                use_safetensors=True
                            )
                            
                            print(f"Successfully loaded {component} after manual download")
                        except Exception as download_error:
                            print(f"Manual download failed for {component}: {download_error}")
                            raise component_error
        
        return models
    
    def setup(self, stage=None):
        """Preprocess latents and prepare datasets."""
        
        if stage in (None, "fit"):
            # Load models
            models = self._load_models(self.model_name)
            
            # Process in smaller batches to save memory
            import gc
            
            print("TESTING MODE: Using limited dataset (10 batches only)")
            
            # First process validation dataset
            print("Setting up validation dataset...")
            self.val_dataset = EditDatasetVal(
                path=self.data_dir / "val",
                metadata_file=self.data_dir / "val/val_metadata.json",
                width_resize=self.width_resize,
                height_resize=self.height_resize,
            )
            
            # Clear memory before processing training dataset
            torch.cuda.empty_cache()
            gc.collect()
            
            # Then process training dataset
            print("Setting up training dataset...")
            self.train_dataset = EditDataset(
                path=self.data_dir / "train",
                metadata_file=self.data_dir / "train/train_metadata.json",
                width_resize=self.width_resize,
                height_resize=self.height_resize,
                vae=models["vae"],
                tokenizer=models["tokenizer"],
                tokenizer_2=models["tokenizer_2"],
                text_encoder=models["text_encoder"],
                text_encoder_2=models["text_encoder_2"],
                device="cuda",
                preprocess=True,
            )

            # Clear CUDA cache after preprocessing
            torch.cuda.empty_cache()
            gc.collect()
            
            # Move models to CPU and clear memory
            to_cpu = ["vae", "text_encoder", "text_encoder_2"]
            for component in to_cpu: 
                models[component].to("cpu")
            
            # Clear references to free memory
            models = None
            torch.cuda.empty_cache()
            gc.collect()


        if stage in (None, "test"):
            self.test_dataset = EditDatasetVal(
                path=self.data_dir / "test",
                metadata_file=self.data_dir / "test/test_metadata.json",
                width_resize=self.width_resize,
                height_resize=self.height_resize,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)