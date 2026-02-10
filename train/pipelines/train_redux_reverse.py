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
from diffusers import FluxPriorReduxPipeline, FluxPipeline
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
import copy
import lpips
import torch.distributed as dist
# from peft import LoraConfig, set_peft_model_state_dict
# from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


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
        self.data_config = args['data']
        self.pipe = None
        self.weight_dtype = torch.bfloat16
    
    def setup(self, stage=None):
        
        # models = {}
        # model_components = ["text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2"]
        # for component in model_components:
        #     model_class = {
        #         "text_encoder": CLIPTextModel,
        #         "tokenizer": CLIPTokenizer,
        #         "text_encoder_2": T5EncoderModel,
        #         "tokenizer_2": T5TokenizerFast,
        #     }[component]
        #     # Load each component using the corresponding subfolder
        #     models[component] = model_class.from_pretrained(
        #         self.args['name'], subfolder=component
        #     )
        # self.text_encoder = models["text_encoder"]
        # self.tokenizer = models["tokenizer"]
        # self.text_encoder_2 = models["text_encoder_2"]
        # self.tokenizer_2 = models["tokenizer_2"]
        
        self.image_embedder = ReduxImageEncoder()
        self.image_embedder.requires_grad_(True)
        self.image_embedder.to(self.device)
        self.image_embedder.train()
 
         # Initialize the FluxImg2ImgPipeline
        
        
       
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
        # Extract batch data
        model_input = batch["model_input"]
        cond_input = batch["cond_input"]
        prompt_embeds = batch["prompt_embeds"].squeeze(1)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].squeeze(1)
        # text_ids = batch["text_ids"][0]
        # vae_scale_factor = batch["vae_scale_factor"][0].item()
        bsz = model_input.shape[0]
        image_embeds = self.image_embedder(cond_input).image_embeds
        prompt_embeds = torch.zeros((bsz, 256, 4096), device=self.device, dtype=image_embeds.dtype)
        # pooled_prompt_embeds is 768, clip text encoder hidden size
        pooled_prompt_embeds = torch.zeros((bsz, 768), device=self.device, dtype=image_embeds.dtype)
         # max_sequence_length is 512, t5 encoder hidden size is 4096
        prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)
        # weighted sum
        # prompt_embeds = torch.sum(prompt_embeds, dim=0, keepdim=True)
        # pooled_prompt_embeds = torch.sum(pooled_prompt_embeds, dim=0, keepdim=True)
        
        target = model_input
        return prompt_embeds, pooled_prompt_embeds, target
        
    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.pipe is None:
                self.pipe = FluxPipeline.from_pretrained(
                    "shuttleai/shuttle-3.1-aesthetic" , 
                    text_encoder=None,
                    text_encoder_2=None,
                    torch_dtype=torch.bfloat16
                ).to(self.device)
                self.pipe.set_progress_bar_config(disable=True)

        prompt_embeds, pooled_prompt_embeds, target_noise = self(batch)
        guidance_scale = np.random.uniform(3.5, 15)
        generated_output = self.pipe(
            height=self.data_config.resize_res[0],
            width=self.data_config.resize_res[1],
            num_inference_steps=6,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=None,
            output_type="latent",
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        ).images
        pred_noise = self.pipe._unpack_latents(generated_output, self.data_config.resize_res[0], self.data_config.resize_res[1], batch["vae_scale_factor"][0].item())
        loss = (F.mse_loss(pred_noise, target_noise)).mean()
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
        with torch.no_grad():
            torch.cuda.empty_cache()
            if self.pipe is None:
                self.pipe = FluxPipeline.from_pretrained(
                    "shuttleai/shuttle-3.1-aesthetic" , 
                    text_encoder=None,
                    text_encoder_2=None,
                    torch_dtype=torch.bfloat16
                ).to(self.device)
                self.pipe.set_progress_bar_config(disable=True)
                
            with torch.no_grad():
                self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
                image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16).to("cpu")
                feature_extractor = SiglipImageProcessor(size={"height": 384, "width": 384}).from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16)
            self.pipe_prior_redux = FluxPriorReduxPipeline(image_encoder=image_encoder, feature_extractor=feature_extractor, image_embedder=self.image_embedder, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None).to("cuda")
            
            
        
    
    def on_validation_end(self):
        self.pipe_prior_redux = None
        self.lpips_fn = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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

