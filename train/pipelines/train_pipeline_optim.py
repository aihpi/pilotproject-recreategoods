import inspect
import pytorch_lightning as pl
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
from diffusers.utils.torch_utils import randn_tensor
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pipelines.inference_pipeline import FluxPix2PixPipeline
from diffusers import FluxImg2ImgPipeline
import copy
import lpips
import torch.distributed as dist
# from peft import LoraConfig, set_peft_model_state_dict
# from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from peft import LoraConfig


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

class InstructPix2PixModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transformer = None
        self.noise_scheduler = None
        self.weight_dtype = torch.bfloat16
    
    def _exchange_layer(self, model):
        original_x_embedder = model.x_embedder
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
        return new_x_embedder
    
    def setup(self, stage=None):
        models = {}
        model_components = ["transformer", "scheduler", "vae", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2"]
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
        transformer = models["transformer"]
        self.noise_scheduler = models["scheduler"]
        transformer.x_embedder = self._exchange_layer(transformer)
        self.transformer = transformer.train()
        self.transformer.gradient_checkpointing = True

        self.vae = models["vae"].to("cpu")
        self.text_encoder = models["text_encoder"].to("cpu")
        self.text_encoder_2 = models["text_encoder_2"].to("cpu")
        self.tokenizer = models["tokenizer"]
        self.tokenizer_2 = models["tokenizer_2"]
        self.transformer.requires_grad_(True)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        # target_modules = [
        #     "attn.to_k",
        #     "attn.to_q",
        #     "attn.to_v",
        #     "attn.to_out.0",
        #     "attn.add_k_proj",
        #     "attn.add_q_proj",
        #     "attn.add_v_proj",
        #     "attn.to_add_out",
        #     "ff.net.0.proj",
        #     "ff.net.2",
        #     "ff_context.net.0.proj",
        #     "ff_context.net.2",
        # ]
        # lora_rank = 32
        # transformer_lora_config = LoraConfig(
        #     r=lora_rank,
        #     lora_alpha=lora_rank,
        #     init_lora_weights="gaussian",
        #     target_modules=target_modules,
        # )
        # transformer.x_embedder.requires_grad_(True)
        # self.transformer.add_adapter(transformer_lora_config)
        # self.transformer_lora_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
         # Initialize the FluxImg2ImgPipeline
        with torch.no_grad():
            self.lpips_fn = lpips.LPIPS(net='alex')
            self.lpips_fn.net.requires_grad_(False)
        
       
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
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        text_ids = batch["text_ids"][0]
        vae_scale_factor = batch["vae_scale_factor"][0].item()

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=bsz,
            logit_mean=0.0, 
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)
        
        sigmas = self.get_sigmas(timesteps, n_dim=cond_input.ndim, dtype=cond_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        noisy_cond_model_input = torch.cat([noisy_model_input, cond_input], dim=1)

        latent_image_ids = _prepare_latent_image_ids(
            noisy_cond_model_input.shape[0],
            noisy_cond_model_input.shape[2] // 2,
            noisy_cond_model_input.shape[3] // 2,
            self.device,
            self.weight_dtype,
        )
        packed_noisy_cond_model_input = _pack_latents(
            noisy_cond_model_input,
            batch_size=noisy_cond_model_input.shape[0],
            num_channels_latents=noisy_cond_model_input.shape[1],
            height=noisy_cond_model_input.shape[2],
            width=noisy_cond_model_input.shape[3],
        )

        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([self.args.guidance_scale], device=self.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        model_pred = self.transformer(
                hidden_states=packed_noisy_cond_model_input,
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
            height=int(model_input.shape[2] * vae_scale_factor),
            width=int(model_input.shape[3] * vae_scale_factor),
            vae_scale_factor=vae_scale_factor,
        )
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)
        target = noise - model_input
        return model_pred, target, weighting
        
    def compute_l2sp_loss(self):
        """
        Compute the L2-SP loss to encourage model weights to stay close to pre-trained weights.
        """
        l2sp_loss = 0.0
        for param, pre_param in zip(self.model.parameters(), self.pretrained_model.parameters()):
            l2sp_loss += torch.sum((param - pre_param) ** 2)
        return l2sp_loss * self.args["l2sp_weight"]
    
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
        self.transformer.eval()
        with torch.no_grad():
            self.vae.to(self.device)
            self.text_encoder.to(self.device)
            self.text_encoder_2.to(self.device)
            torch.cuda.empty_cache()
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
            self.pipeline.set_progress_bar_config(disable=True)
            
        
    
    def on_validation_end(self):
        self.pipeline = None
        import gc
        gc.collect()
        self.vae.to("cpu")
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        self.transformer.train()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            in_pixel_values = batch["input_image"].to(dtype=self.vae.dtype)
            prompts = batch["edit_instruction"]
            strength = 1
            generated_output = self.pipeline(
                prompt=prompts,
                image=in_pixel_values,
                height=in_pixel_values.shape[2],
                width=in_pixel_values.shape[3],
                strength=strength,
                num_inference_steps=6,
                guidance_scale=4.5,
                num_images_per_prompt=1,
                generator=None,
                output_type="pt",
            ).images

            gt_images = batch["output_image"].to(dtype=torch.bfloat16, device=generated_output.device)
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
        optimizer = FusedAdam(self.transformer.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
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

