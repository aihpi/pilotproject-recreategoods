import inspect
import lightning as pl
import torch
from torchvision.utils import make_grid
import json
import wandb
from diffusers import FluxPipeline
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from deepspeed.ops.adam import FusedAdam

from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
import numpy as np
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from pipelines.tokenize import tokenize_prompt, encode_prompt
from diffusers.utils.torch_utils import randn_tensor
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pipelines.inference_pipeline import FluxPix2PixPipeline
import copy


@staticmethod
def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

@staticmethod
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

@staticmethod
def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

class InstructPix2PixModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transformer = None
        self.vae = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.weight_dtype = torch.bfloat16
    
    
    def setup(self, stage=None):
        models = {}
        model_components = ["transformer", "vae", "scheduler", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2"]
        for component in model_components:
            model_class = {
                "scheduler": FlowMatchEulerDiscreteScheduler,
                "vae": AutoencoderKL,
                "text_encoder": CLIPTextModel,
                "tokenizer": CLIPTokenizer,
                "text_encoder_2": T5EncoderModel,
                "tokenizer_2": T5TokenizerFast,
                "transformer": FluxTransformer2DModel,
            }[component]

            # Load each component using the corresponding subfolder
            models[component] = model_class.from_pretrained(
                self.args['name'], subfolder=component
            )
        self.transformer = models["transformer"].to(self.device)
        self.vae = models["vae"].to("cpu")
        self.noise_scheduler = models["scheduler"]
        self.text_encoder = models["text_encoder"]
        self.tokenizer = models["tokenizer"]
        self.text_encoder_2 = models["text_encoder_2"]
        self.tokenizer_2 = models["tokenizer_2"]

        # Update the x_embedder layer in the transformer
        original_x_embedder = self.transformer.x_embedder

        new_x_embedder = nn.Linear(
            in_features=128,  # Updated input size
            out_features=original_x_embedder.out_features,  # Dynamically use the original out_features
            bias=original_x_embedder.bias is not None  # Preserve bias configuration
        )
         # Copy the weights for the first 64 input channels and initialize the remaining input channels to zero
        new_x_embedder.weight.data[:, :original_x_embedder.in_features] = original_x_embedder.weight.data
        new_x_embedder.weight.data[:, original_x_embedder.in_features:] = 0
        if original_x_embedder.bias is not None:
            new_x_embedder.bias.data = original_x_embedder.bias.data.clone()
        self.transformer.x_embedder = new_x_embedder
        # Freeze non-trainable components
        self.transformer.train()
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
         # Initialize the FluxImg2ImgPipeline
        with torch.no_grad():
            scheduler_copy = copy.deepcopy(self.noise_scheduler)
            self.pipeline = FluxPix2PixPipeline(
                transformer=self.transformer,
                vae=self.vae,
                scheduler=scheduler_copy,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                text_encoder_2=self.text_encoder_2,
                tokenizer_2=self.tokenizer_2,
            )
        self.logger.experiment.define_metric("Validation Images", step_metric="global_step")

    def get_sigmas(self,timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    

    def forward(self, batch):
        in_pixel_values = batch["input_image"].to(dtype=self.vae.dtype)
        out_pixel_values = batch["output_image"].to(dtype=self.vae.dtype)
        prompts = batch["edit_instruction"]
        tokens_one = tokenize_prompt(self.tokenizer, prompts, max_sequence_length=77)
        tokens_two = tokenize_prompt(
            self.tokenizer_2, prompts, max_sequence_length=256
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders=[self.text_encoder, self.text_encoder_2],
            tokenizers=[None, None],
            text_input_ids_list=[tokens_one, tokens_two],
            max_sequence_length=256,
            prompt=prompts,
        )
        # Convert images to latent space
        model_input = self.vae.encode(out_pixel_values).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        model_input = model_input.to(dtype=self.weight_dtype)

        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        bsz = out_pixel_values.shape[0]
           # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)
     
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        noise = torch.randn_like(model_input)
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
       
        cond_model_input = self.vae.encode(in_pixel_values).latent_dist.sample()
        cond_model_input = (cond_model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        cond_model_input = cond_model_input.to(dtype=self.weight_dtype)
        packed_noisy_model_input = torch.cat([noisy_model_input, cond_model_input], dim=1)

        latent_image_ids = _prepare_latent_image_ids(
            packed_noisy_model_input.shape[0],
            packed_noisy_model_input.shape[2] // 2,
            packed_noisy_model_input.shape[3] // 2,
            self.device,
            self.weight_dtype,
        )
        packed_noisy_model_input = _pack_latents(
            packed_noisy_model_input,
            batch_size=packed_noisy_model_input.shape[0],
            num_channels_latents=packed_noisy_model_input.shape[1],
            height=packed_noisy_model_input.shape[2],
            width=packed_noisy_model_input.shape[3],
        )
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([self.args.guidance_scale], device=self.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        model_pred = self.transformer(
                hidden_states=packed_noisy_model_input,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,)[0]
         # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
        model_pred = _unpack_latents(
            model_pred,
            height=model_input.shape[2] * self.vae_scale_factor,
            width=model_input.shape[3] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="sigma_sqrt", sigmas=sigmas)
        target = noise - model_input
        return model_pred, target, weighting
        

    def training_step(self, batch, batch_idx):
        pred_noise, target_noise, weight = self(batch)
        loss = (weight * F.mse_loss(pred_noise, target_noise)).mean()
        self.log("train_loss", loss, prog_bar=True)
        wandb_logs = {
        "batch_idx": batch_idx,
        "train_loss": loss.detach().item(),  # Log the scalar value
        }
        self.logger.experiment.log(wandb_logs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Perform reverse diffusion using FluxImg2ImgPipeline
        if batch_idx == 0:  # Log only for the first batch in validation
            with torch.no_grad():
                
                # Prepare inputs
                in_pixel_values = batch["input_image"].to(dtype=self.vae.dtype)
                prompts = batch["edit_instruction"]

                # Generate images using the pipeline
                generated_output = self.pipeline(
                    prompt=prompts,
                    image=in_pixel_values,
                    height=512,
                    width=512,
                    strength=1,  # Adjust strength to control transformation extent
                    num_inference_steps=20,  # Number of denoising steps
                    guidance_scale=4.5,  # Guidance scale
                    num_images_per_prompt=1,
                    generator=None,  # Optionally set for deterministic results
                    output_type="pt",  # Get images as PIL objects
                )
                if self.trainer.is_global_zero:
                    # Create a grid for generated images
                    generated_grid = make_grid(generated_output, nrow=4)

                    # Create a grid for input images
                    input_grid = make_grid(batch["input_image"].to(dtype=torch.float32), nrow=4)

                    # Create a grid for output images (ground truth)
                    output_grid = make_grid(batch["output_image"].to(dtype=torch.float32), nrow=4)

                    # Prepare the captions for the edit instructions
                    edit_instructions = batch["edit_instruction"]

                    # Log all images to WandB
                    self.logger.experiment.log({
                        "Validation Generated Images": wandb.Image(generated_grid, caption=edit_instructions),
                        "Validation Input Images": wandb.Image(input_grid, caption="Input Images"),
                        "Validation Output Images": wandb.Image(output_grid, caption="Ground Truth Images"),
                    })
            del generated_output, in_pixel_values, prompts
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.transformer.train()
        return {}
    
    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.transformer.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs)
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "epoch",  
            "frequency": 1        
        }
        return [optimizer], [scheduler_dict]

    def log_sample_images(self, input_images, reconstructed_images, prompts):
        """
        Log original and reconstructed images to WandB.

        Args:
            input_images (torch.Tensor): Input images.
            reconstructed_images (torch.Tensor): Reconstructed/generated images.
            prompts (List[str]): Corresponding prompts.
        """
        # Convert images to grid
        input_grid = make_grid(input_images, normalize=True, scale_each=True)
        reconstructed_grid = make_grid(reconstructed_images, normalize=True, scale_each=True)

        # Log to WandB
        self.logger.experiment.log({
            "Input Images": [wandb.Image(input_grid, caption="Input Images")],
            "Reconstructed Images": [wandb.Image(reconstructed_grid, caption="Reconstructed Images")],
            "Prompts": prompts,
        })

    @staticmethod
    def save_results(results, output_dir):
        """
        Save generated images and metadata.

        Args:
            results (Dict): Results dictionary containing generated images and metadata.
            output_dir (str): Directory to save the results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for seed, result in results.items():
            result["image"].save(output_dir / f"{seed}.jpg", quality=100)
            with open(output_dir / "metadata.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

