import torch

# model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16)
# processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")


# inputs = processor(images=image, return_tensors="pt")
# print(f"GOOGLE inputs: {inputs}")
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# print(f"GOOGLE last hidden state: {last_hidden_state}, shape: {last_hidden_state.shape}")
# pooled_output = outputs.pooler_output  # pooled features
# print(f"GOOGLE MODEL: {model}")
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

# pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16).to("cuda")
# pipe = FluxPipeline.from_pretrained(
#     "shuttleai/shuttle-3.1-aesthetic" , 
#     text_encoder=None,
#     text_encoder_2=None,
#     torch_dtype=torch.bfloat16
# ).to("cuda")
# # print(f"Image Embedding model: {pipe_prior_redux.image_embedder}")
# pipe_prior_output = pipe_prior_redux(image)
# # print(f"pipe_prior_output: {pipe_prior_output}")
# images = pipe(
#     guidance_scale=2.5,
#     num_inference_steps=50,
#     generator=torch.Generator("cpu").manual_seed(0),
#     **pipe_prior_output,
# ).images
# images[0].save("flux-dev-redux.png")
from diffusers import FluxPipeline
from pipelines.fluxpriorpipeline import FluxPriorReduxPipeline
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from transformers import SiglipImageProcessor, SiglipVisionModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as pl
import torch
from torchvision.utils import make_grid
import json
import wandb
from diffusers import FluxPipeline
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam
import deepspeed
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
import numpy as np
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
# from pipelines.tokenize import tokenize_prompt, encode_prompt
from diffusers.utils import BaseOutput
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pipelines.inference_pipeline import FluxPix2PixPipeline
from diffusers import FluxImg2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor
import copy
import lpips
import torch.distributed as dist
# from peft import LoraConfig, set_peft_model_state_dict
# from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from typing import List, Optional, Tuple, Union
import inspect

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

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
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class ReduxImageEncoderOutput(BaseOutput):
    image_embeds = None

class ReduxImageEncoder(ModelMixin, ConfigMixin):
    def __init__(self, redux_dim: int = 1152, txt_in_features: int = 4096,) -> None:
        super().__init__()
        self.redux_up = nn.Linear(redux_dim, txt_in_features * 3)
        self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features)

    def forward(self, x: torch.Tensor):
        projected_x = self.redux_down(nn.functional.silu(self.redux_up(x)))

        return ReduxImageEncoderOutput(image_embeds=projected_x)

class InstructPix2PixModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args['model']
        self.data_args = args['data']
        self.transformer = None
        self.noise_scheduler = None
        self.weight_dtype = torch.bfloat16
    
    def setup(self, stage=None):
        models = {}
        model_components = ["transformer", "scheduler"]
        for component in model_components:
            model_class = {
                "scheduler": FlowMatchEulerDiscreteScheduler,
                "transformer": FluxTransformer2DModel,
                "vae": AutoencoderKL,
                "text_encoder": CLIPTextModel,
                "tokenizer": CLIPTokenizer,
                "text_encoder_2": T5EncoderModel,
                "tokenizer_2": T5TokenizerFast,
            }[component]
            # Load each component using the corresponding subfolder
            models[component] = model_class.from_pretrained(
                self.args['name'], subfolder=component
            )
        self.image_embedder = ReduxImageEncoder()
        self.image_embedder.requires_grad_(True)
        self.image_embedder.to(self.device)
        self.image_embedder.train()

        self.transformer = models["transformer"]
        self.noise_scheduler = models["scheduler"]
        self.transformer.eval()
        for params in self.transformer.parameters():
            params.requires_grad = False
        self.transformer.gradient_checkpointing = True
        self.transformer.to('cpu')
        with torch.no_grad():
            self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        
        # torch.utils.checkpoint.checkpoint_sequential(self.transformer.modules(), segments=2)
        
        
       
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids
    

    def forward(self, batch):
        # Extract batch data
        model_input = batch["model_input"]
        cond_input = batch["cond_input"]
        prompt_embeds = batch["prompt_embeds"].squeeze(1)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].squeeze(1)
        text_ids = batch["text_ids"][0]
        vae_scale_factor = batch["vae_scale_factor"][0].item()
        height = self.data_args.resize_res[0]
        width = self.data_args.resize_res[1]
        self.vae_scale_factor = vae_scale_factor
        bsz = model_input.shape[0]
      

        image_embeds = self.image_embedder(cond_input).image_embeds
         # max_sequence_length is 512, t5 encoder hidden size is 4096
        prompt_embeds = torch.zeros((bsz, 512, 4096), device=self.device, dtype=image_embeds.dtype)
        # pooled_prompt_embeds is 768, clip text encoder hidden size
        pooled_prompt_embeds = torch.zeros((bsz, 768), device=self.device, dtype=image_embeds.dtype)
        prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)
        # weighted sum
        # prompt_embeds = torch.sum(prompt_embeds, dim=0, keepdim=True)
        # pooled_prompt_embeds = torch.sum(pooled_prompt_embeds, dim=0, keepdim=True)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=self.device, dtype=self.transformer.dtype)
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            None,
            None,
        )
        num_inference_steps = 6
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.noise_scheduler.config.base_image_seq_len,
            self.noise_scheduler.config.max_image_seq_len,
            self.noise_scheduler.config.base_shift,
            self.noise_scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.noise_scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)
        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=None,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        target = model_input
        return latents, target
        
    def training_step(self, batch, batch_idx):
        pred, target = self(batch)
        loss = (F.mse_loss(pred, target)).mean()
        self.log("train_loss", loss, prog_bar=True)
        wandb_logs = {
        "batch_idx": batch_idx,
        "train_loss": loss.detach().item(),  # Log the scalar value
        }
        self.logger.experiment.log(wandb_logs)
        return loss
    
    def _get_lpips_mean(self, gen_images, gt_images):
        gen_images_lpips = 2.0 * gen_images - 1.0
        gt_images_lpips = 2.0 * gt_images - 1.0
        lpips_values = self.lpips_fn.forward(gen_images_lpips, gt_images_lpips)
        lpips_mean = lpips_values.mean()
        return lpips_mean

    def _get_inference_steps(self):
        random_inference_steps = torch.randint(low=4, high=21, size=(1,), device=self.device)
        gathered_steps = self.trainer.strategy.all_gather(random_inference_steps)
        num_inference_steps = int(gathered_steps[0].item())
        return num_inference_steps
    
    
    def on_validation_start(self):
        self.transformer.to("cpu")
        with torch.no_grad():
            torch.cuda.empty_cache()
            image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16).to("cpu")
            feature_extractor = SiglipImageProcessor(size={"height": 384, "width": 384}).from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16)
            self.pipe_prior_redux = FluxPriorReduxPipeline(image_encoder=image_encoder, feature_extractor=feature_extractor, image_embedder=self.image_embedder, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None).to("cuda")
            self.pipe = FluxPipeline.from_pretrained(
                "shuttleai/shuttle-3.1-aesthetic" , 
                text_encoder=None,
                text_encoder_2=None,
                torch_dtype=torch.bfloat16
            ).to("cuda")
            self.pipe.set_progress_bar_config(disable=True)
            
        
    
    def on_validation_end(self):
        self.pipe = None
        self.pipe_prior_redux = None
        self.lpips_fn = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        self.transformer.to(self.device)


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            in_pixel_values = batch["input_image"].to(dtype=torch.float32)
            in_pixel_values = (in_pixel_values * 255).byte()
            prompts = batch["edit_instruction"]
            gt_images = batch["output_image"]
            pipe_prior_output = self.pipe_prior_redux(in_pixel_values)
            generated_output = self.pipe(
                height=gt_images.shape[2],
                width=gt_images.shape[3],
                num_inference_steps=6,
                guidance_scale=4.5,
                num_images_per_prompt=1,
                generator=None,
                output_type="pt",
                **pipe_prior_output,
            ).images
            gt_images = gt_images.to(dtype=torch.bfloat16, device=generated_output.device)
            gen_images = generated_output.to(dtype=torch.bfloat16, device=generated_output.device)
            lpips_mean = self._get_lpips_mean(gen_images, gt_images)
            self.log("val_lpips", lpips_mean, on_step=False, on_epoch=True, sync_dist=True, batch_size=in_pixel_values.shape[0])

            if batch_idx == 0 and self.trainer.is_global_zero:
                generated_grid = make_grid(gen_images, nrow=4)
                input_grid = make_grid(batch["input_image"].float(), nrow=4)
                output_grid = make_grid(gt_images, nrow=4)
                edit_instructions = batch["edit_instruction"]
                # Log to WandB
                self.logger.experiment.log({
                    "Validation Generated Images": wandb.Image(generated_grid, caption=edit_instructions),
                    "Validation Input Images": wandb.Image(input_grid, caption="Input Images"),
                    "Validation Output Images": wandb.Image(output_grid, caption="Ground Truth Images"),
                })

        # Cleanup
        del generated_output, in_pixel_values, prompts, gen_images, gt_images
        return {}
    

    
    def configure_optimizers(self):
        optimizer = FusedAdam(self.image_embedder.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
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

