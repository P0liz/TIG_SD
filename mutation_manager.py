# %%writefile mutation_manager.py
from PIL import Image
import numpy as np
import cv2
import torch
import math
import random

from torchvision import transforms
from diffusion import pipeline_manager
from config import DEVICE, DELTA, STANDING_STEP_LIMIT, DATASET

# Local config
CLAMP_MIN = -5.41362476348877
CLAMP_MAX = 5.43081117630005

# TODO: understand this
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def mutate(z_orig, perturbation_size, generator=None):
    """
    Muta il latent code aggiungendo rumore gaussiano

    Args:
        z_orig: torch.Tensor - latent originale
        delta: float - intensità della perturbazione

    Returns:
        z_mut: torch.Tensor - latent mutato
    """
    epsilon = torch.randn(
        z_orig.shape, device=z_orig.device, dtype=z_orig.dtype, generator=generator
    )  # randn to have ε ~ N(0, I) noise
    z_mut = z_orig + perturbation_size * epsilon
    z_mut = torch.clamp(z_mut, min=CLAMP_MIN, max=CLAMP_MAX)
    return z_mut


def calculate_perturbation_size(standing_steps):
    # Progressive intensification of perturbation size
    # It is increased by one time every STANDING_STEP_LIMIT standing steps
    if standing_steps >= STANDING_STEP_LIMIT:
        perturbation_size = DELTA * (standing_steps / STANDING_STEP_LIMIT + 1)
    else:
        perturbation_size = DELTA
    print(f"Perturbation size: {perturbation_size:.3f}")
    return perturbation_size


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


def generate(prompts, mutated_latents, guidance_scale=2.5, generator=None):
    # Using batch generation to speed up the process, since the pipeline can handle multiple latents at once
    pipe = pipeline_manager.pipe

    with torch.inference_mode():
        if pipeline_manager.optimization:
            images = pipe.tgate(
                prompt=prompts,
                guidance_scale=guidance_scale,
                gate_step=10,
                num_inference_steps=pipeline_manager.num_inference_steps,
                latents=mutated_latents,
                generator=generator,
            )["images"]
        else:
            images = pipe(
                prompt=prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=pipeline_manager.num_inference_steps,
                latents=mutated_latents,
                generator=generator,
            )["images"]
    torch.cuda.empty_cache()

    image_tensors = []
    if DATASET == "mnist":
        # preprocess image to 28x28 grayscale tensor
        for image in images:
            image_tensors.append(process_mnist_image(image).to(DEVICE))
    elif DATASET == "imagenet":
        # preprocess image to 224x224 RGB tensor
        for image in images:
            image_tensors.append(transform(image).to(DEVICE))
    else:
        raise ValueError("Unsupported dataset specified in config")
    return image_tensors, images


def process_mnist_image(image):
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
