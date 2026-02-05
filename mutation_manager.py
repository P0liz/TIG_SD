# %%writefile mutation_manager.py
from PIL import Image
import numpy as np
import cv2
import torch
import math
import random

from torchvision import transforms
from diffusion import pipeline_manager, get_pipeline
from config import DEVICE, CIRC_STEPS, NOISE_SCALE, DEVICE


def mutate(z_orig, perturbation_size):
    """
    Muta il latent code aggiungendo rumore gaussiano

    Args:
        z_orig: torch.Tensor - latent originale
        delta: float - intensità della perturbazione

    Returns:
        z_mut: torch.Tensor - latent mutato
    """
    epsilon = torch.randn_like(z_orig, device=DEVICE)  # randn to have ε ~ N(0, I) noise
    z_mut = z_orig + perturbation_size * epsilon
    # z_mut = apply_mutation_op1(z_orig)
    return z_mut


# Applying mutation only on a random channel, and only on some of its columns
# TODO: consider applying perturbation_size to have less aggressive mutations?
def apply_mutation_op1(org_latent, device=DEVICE):
    mutated_latent = org_latent.clone()
    target_channel = random.randint(0, 3)
    num_columns_to_mutate = random.randint(1, 16)
    selected_columns = random.sample(range(64), num_columns_to_mutate)

    for col in selected_columns:
        random_factor = torch.randn(1, device=device)
        while abs(random_factor.item()) > 1.6:
            random_factor = torch.randn(1, device=device)
        mutated_latent[:, target_channel, :, col] *= random_factor
    mutated_latent = torch.clamp(mutated_latent, min=-4.0, max=4.0)
    return mutated_latent


# Circular walk mutation
def mutate_circular(z_orig, step, noise_x, noise_y, total_steps=CIRC_STEPS, noise_scale=NOISE_SCALE):
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
    image_tensor = process_image(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
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
