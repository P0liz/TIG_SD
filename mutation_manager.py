# %%writefile mutation_manager.py
from PIL import Image
import numpy as np
import cv2
import torch
import math

from torchvision import transforms
from config import (
    DEVICE,
    MODEL_ID_PATH,
    LORA_PATH,
    LORA_WEIGHTS,
    DTYPE,
    VARIANT,
    CIRC_STEPS,
    NOISE_SCALE,
)
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler


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
    # TODO: Load/Init VAE, tokenizer, text_encoder, unet and scheduler and make the accessible from here


# Istanza globale singleton
pipeline_manager = SDPipelineManager()


def get_pipeline():
    """Funzione helper per ottenere la pipeline"""
    return pipeline_manager.get_pipe()


def mutate(z_orig, perturbation_size):
    """
    Muta il latent code aggiungendo rumore gaussiano

    Args:
        z_orig: torch.Tensor - latent originale
        delta: float - intensità della perturbazione

    Returns:
        z_mut: torch.Tensor - latent mutato
    """
    epsilon = torch.randn_like(z_orig)  # randn to have ε ~ N(0, I) noise
    z_mut = z_orig + perturbation_size * epsilon
    return z_mut


# Circular walk mutation
def mutate_circular(
    z_orig, step, noise_x, noise_y, total_steps=CIRC_STEPS, noise_scale=NOISE_SCALE
):
    """
    Muta il latent seguendo un punto specifico del percorso circolare

    Args:
        step: int - quale step del walk (0 a total_steps-1)
        total_steps: int - numero totale di step nel cerchio
        noise_x, noise_y: torch.Tensor - direzioni di rumore (opzionali, se vuoi riusarle)
    """
    t = (step / total_steps) * 2 * math.pi
    scale_x = math.cos(t)
    scale_y = math.sin(t)

    z_mut = z_orig + noise_scale * (scale_x * noise_x + scale_y * noise_y)

    return z_mut


def generate(prompt, mutated_latent=None, guidance_scale=2.5):
    """
    Genera un'immagine usando Stable Diffusion

    Args:
        prompt: str - prompt testuale
        mutated_latent: torch.Tensor - latent code (opzionale)

    Returns:
        mutated_latent: torch.Tensor - latent usato
        image_tensor: torch.Tensor - immagine preprocessata [1, 1, 28, 28]
        image: PIL.Image - visual image
    """
    pipe = get_pipeline()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=pipeline_manager.num_inference_steps,
            latents=mutated_latent,
        )["images"][0]

    # preprocess image to 28x28 grayscale tensor
    image_tensor = (
        process_image(image).unsqueeze(0).to(DEVICE)
    )  # Add batch dimension and move to device
    return mutated_latent, image_tensor, image


def process_image(image):
    """
    Convert a 3-channel RGB PIL Image to grayscale, resize it to 28x28 pixels,
    and convert it to a PyTorch tensor.

    Parameters:
    - image (PIL.Image): The input RGB image.

    Returns:
    - tensor (torch.Tensor): The processed image as a PyTorch tensor.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The provided image needs to be a PIL Image.")

    # Convert PIL Image to numpy array (RGB)
    img_np = np.array(image)

    # Convert the image from RGB to grayscale
    gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Resize the image to 28x28 pixels
    # Could use cv2.INTER_AREA for better quality when shrinking, but might cause more blurry results
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    # Convert the numpy array back to PIL Image (to use torchvision transforms)
    img_pil = Image.fromarray(resized_image)

    # Convert PIL Image to PyTorch Tensor
    transform = transforms.ToTensor()
    tensor = transform(img_pil)

    return tensor
