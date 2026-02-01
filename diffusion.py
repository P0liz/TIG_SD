# %%writefile diffusion.py
import torch
from config import (
    DEVICE,
    MODEL_ID_PATH,
    LORA_PATH,
    LORA_WEIGHTS,
    DTYPE,
    VARIANT,
    PROMPTS,
)

# standard
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

# custom
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm
from PIL import Image


# Using Stable diffusion pipeline
class SDPipelineManager:
    """Singleton per gestire la pipeline Stable Diffusion"""

    _instance = None
    _pipe = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SDPipelineManager, cls).__new__(cls)
        return cls._instance

    # ------------------------------------------------------
    #   STANDARD IMPLEMENTATION
    # ------------------------------------------------------

    def initialize(self, num_inference_steps=15):
        """Inizializza la pipeline solo la prima volta"""
        if self._initialized:
            print("✓ Using cached Stable Diffusion pipeline")
            return self._pipe

        print("Loading Stable Diffusion pipeline...")

        # Load pipeline
        self._pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID_PATH, variant=VARIANT, torch_dtype=DTYPE, safety_checker=None
        ).to(DEVICE)
        print("Loaded Stable Diffusion model")

        # Load LoRA weights
        self._pipe.load_lora_weights(LORA_PATH, weight_name=LORA_WEIGHTS)
        print("Loaded LoRA weights")

        # Configure scheduler (only once)
        self._pipe.scheduler = DDIMScheduler.from_config(
            self._pipe.scheduler.config, rescale_betas_zero_snr=True
        )
        print("Configured scheduler")

        self.num_inference_steps = num_inference_steps
        self._initialized = True

        print("Pipeline ready for inference")
        return self._pipe

    def get_pipe(self):
        """Ottieni la pipeline (inizializza se necessario)"""
        if not self._initialized:
            self.initialize()
        return self._pipe

    # ------------------------------------------------------
    #   CUSTOM IMPLEMENTATION
    # ------------------------------------------------------
    vae = None
    tokenizer = None
    text_encoder = None
    unet = None
    scheduler = None

    def initialize_custom(self):
        """Inizializza i componenti personalizzati della pipeline"""

        print("Loading custom components...")

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            MODEL_ID_PATH,
            subfolder="vae",
            variant=VARIANT,
            torch_dtype=DTYPE,
            use_safetensors=True,
        ).to(DEVICE)
        print("Loaded VAE")

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_ID_PATH, subfolder="tokenizer", variant=VARIANT
        )
        print("Loaded tokenizer")

        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID_PATH,
            subfolder="text_encoder",
            variant=VARIANT,
            torch_dtype=DTYPE,
            use_safetensors=True,
        ).to(DEVICE)
        print("Loaded text encoder")

        # Load unet
        self.unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID_PATH,
            subfolder="unet",
            variant=VARIANT,
            torch_dtype=DTYPE,
            use_safetensors=True,
        ).to(DEVICE)
        print("Loaded unet")

        # TODO: Understand differences between schedulers
        # TODO: should I change it ?
        # Load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            MODEL_ID_PATH,
            subfolder="scheduler",
            rescale_betas_zero_snr=True,
        )
        print("Loaded scheduler")


# Istanza globale singleton
pipeline_manager = SDPipelineManager()


def get_pipeline():
    """Funzione helper per ottenere la pipeline"""
    return pipeline_manager.get_pipe()


def text_embeddings(prompt):
    BATCH_SIZE = len(PROMPTS)
    # Generate embeddings for the prompt
    text_input = pipeline_manager.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline_manager.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = pipeline_manager.text_encoder(
            text_input.input_ids.to(DEVICE)
        )[0]

    # Generate the unconditional text embeddings
    max_length = text_input.input_ids.shape[-1]
    uncond_input = pipeline_manager.tokenizer(
        [""] * BATCH_SIZE,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipeline_manager.text_encoder(
        uncond_input.input_ids.to(DEVICE)
    )[0]
    # Concatenate the unconditional and conditional embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def denoise(latent, text_embeddings, guidance_scale=2.5, num_inference_steps=15):
    scheduler = pipeline_manager.scheduler
    # scaling the input with the initial noise distribution
    latent = latent * scheduler.init_noise_sigma

    # TODO: set custom timesteps
    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latent] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = pipeline_manager.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latent = scheduler.step(noise_pred, t, latent).prev_sample
        return latent


def decode_image(latents):
    # Scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = pipeline_manager.vae.decode(latents).sample
    # Convert to PIL Image
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image
