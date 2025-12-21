from PIL import Image
import numpy as np
import cv2
import torch

from torchvision import transforms
from config import DEVICE, MODEL_ID_PATH, LORA_WEIGHTS_PATH
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

# Extras
#import keras_cv

def mutate(z_orig, delta):
    """
    z_orig: ndarray (latent)
    delta: float (perturbation step) adjusted based on confidence
    """
    epsilon = np.random.randn(*z_orig.shape).astype(z_orig.dtype)  # ε ~ N(0, I) noise
    z_mut = z_orig + delta * epsilon
    return z_mut

def generate(prompt, mutated_latent=None):
    """
    prompt: str
    mutated_latent: ndarray (latent)
    """
    # Using Stable diffusion pipeline
    num_inference_steps = 15
    
    # Loading model
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID_PATH, 
        variant="fp16", 
        torch_dtype=torch.float16,     # reduce memory usage
        safety_checker=None).to(DEVICE)
    print("Loaded Stable Diffusion model")
    pipe.load_lora_weights(LORA_WEIGHTS_PATH)
    print("Loaded LORA weights")
    
    # (Opzionale) Cambia scheduler per velocità
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True
    )
    print("Loaded scheduler")
    
    
    with torch.inference_mode():
        image = pipe(prompt=prompt, guidance_scale=3.5, num_inference_steps=num_inference_steps, latents=mutated_latent)["images"][0]
        # preprocess image to 28x28 grayscale tensor
        image_tensor = process_image(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
    return mutated_latent, image_tensor
    
    # Using diffusion library 
    #cond_latent, image = diffusion.generate(prompt, mutate = mutate, mutated_latent = mutated_latent)
    #return cond_latent, image 
    
    # Using KerasCV Stable Diffusion model
    # NOT ENOUGH MEMORY
    """
    model = keras_cv.models.StableDiffusion(jit_compile=True)

    image = model.generate_image(
        prompt,
        batch_size=1,
        num_steps=10,        # Riduci da 50 a 10
        diffusion_noise=mutated_latent,
    )

    return mutated_latent, image
    """

# TODO: capire come funziona e cosa succede ancora prima di passare l'immagine al classificatore
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
    resized_image = cv2.resize(gray_image, (28, 28))

    # Convert the numpy array back to PIL Image (to use torchvision transforms)
    img_pil = Image.fromarray(resized_image)

    # Convert PIL Image to PyTorch Tensor
    transform = transforms.ToTensor()
    tensor = transform(img_pil)

    return tensor