# %%writefile diffusion.py
import torch
from config import *

# standard
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

# custom
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin
from PIL import Image


# Using Stable diffusion pipeline
class SDPipelineManager:
    """Singleton per gestire la pipeline Stable Diffusion"""

    _instance = None
    _initialized = False
    _mode = None

    # Singleton pattern
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SDPipelineManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, mode="standard", num_inference_steps=15):
        """
        Initialize pipeline
        Args:
            mode: 'standard' (uses StableDiffusionPipeline) or 'custom' (loads components separately)
        """
        if self._initialized:
            if self._mode == mode:
                print(f"Using cached {mode} pipeline")
                return self.pipe
            else:
                raise ValueError(f"Pipeline already initialized in {self._mode} mode, cannot switch to {mode}")

        print(f"Loading {mode} Stable Diffusion pipeline...")

        if mode == "standard":
            self._init_standard()
        elif mode == "custom":
            self._init_custom()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'standard' or 'custom'")

        self._mode = mode
        self.num_inference_steps = num_inference_steps
        self._initialized = True
        print("Pipeline ready for inference")
        return self.pipe

    # ------------------------------------------------------
    #   STANDARD IMPLEMENTATION
    # ------------------------------------------------------
    pipe = None

    def _init_standard(self):
        """Load Standard pipeline"""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID_PATH, variant=VARIANT, torch_dtype=DTYPE, safety_checker=None
        ).to(DEVICE)

        # Load LoRA weights
        self.pipe.load_lora_weights(LORA_PATH, weight_name=LORA_WEIGHTS)

        # Configure scheduler
        if TRYNEW:
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="leading"
            )
        else:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, rescale_betas_zero_snr=True)

    # ------------------------------------------------------
    #   CUSTOM IMPLEMENTATION
    # ------------------------------------------------------
    vae = None
    tokenizer = None
    text_encoder = None
    unet = None
    scheduler = None

    def _init_custom(self):
        """Load Custom pipeline components"""
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            MODEL_ID_PATH, subfolder="vae", variant=VARIANT, torch_dtype=DTYPE, use_safetensors=True
        ).to(DEVICE)

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID_PATH, subfolder="tokenizer", variant=VARIANT)

        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID_PATH, subfolder="text_encoder", variant=VARIANT, torch_dtype=DTYPE, use_safetensors=True
        ).to(DEVICE)

        # Load unet
        self.unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID_PATH, subfolder="unet", variant=VARIANT, torch_dtype=DTYPE, use_safetensors=True
        ).to(DEVICE)

        # Load LoRA weights into UNet and text encoder
        state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(LORA_PATH, weight_name=LORA_WEIGHTS)
        LoraLoaderMixin.load_lora_into_unet(state_dict=state_dict, network_alphas=network_alphas, unet=self.unet)
        LoraLoaderMixin.load_lora_into_text_encoder(
            state_dict=state_dict, network_alphas=network_alphas, text_encoder=self.text_encoder
        )

        # Load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            MODEL_ID_PATH, subfolder="scheduler", rescale_betas_zero_snr=False, timestep_spacing="leading"
        )
        # Possible alternative schedulers (while testing these gave less noisy images)
        """
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            MODEL_ID_PATH, 
            subfolder="scheduler", 
            timestep_spacing="leading" 
        )
        
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            MODEL_ID_PATH, 
            subfolder="scheduler",
            timestep_spacing="leading" 
        )
        """

    def text_embeddings(self, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]  # prompt must be a list
        BATCH_SIZE = len(prompt)
        # Generate embeddings for the prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        # Generate the unconditional text embeddings
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * BATCH_SIZE, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
        # Concatenate the unconditional and conditional embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def denoise(self, isMutating=True, guidance_scale=2.5):
        raise NotImplementedError("Use denoise_and_decode() of digit_mutator instead")

    def decode_image(self, latent):
        # Scale and decode the image latents with vae
        latent = 1 / 0.18215 * latent
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        return image


# Singleton global instance
# Remeber to initialize the pipeline before using it
pipeline_manager = SDPipelineManager()
