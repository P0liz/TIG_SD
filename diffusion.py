"""
# Currently not used in TIG SD project
# but might be useful for future reference (caching pipeline, diffusion process, etc.)
import torch
import inspect
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
DATASET = 'MNIST'
TORCH_DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'
# TODO: set the correct model paths (find models)
SD_MODEL_PATH='?'
SD_LORA_PATH='?'

_cached_pipeline = None

@torch.no_grad()
def diffuse(
        pipeline,
        cond_embeddings, # text conditioning, should be (1, 77, 768)
        cond_latents,    # image conditioning, should be (1, 4, 64, 64)
        num_inference_steps,
        guidance_scale,
        eta,
    ):
    '''
    conducts diffusion process
    returns: result img in numpy.ndarray shape:(512, 512, 3)
    '''
    torch_device = cond_latents.get_device()

    # classifier guidance: add the unconditional embedding
    max_length = cond_embeddings.shape[1] # 77
    uncond_input = pipeline.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    # if we use LMSDiscreteScheduler, make sure latents are mulitplied by sigmas
    if isinstance(pipeline.scheduler, LMSDiscreteScheduler):
        cond_latents = cond_latents * pipeline.scheduler.sigmas[0]

    # init the scheduler
    accepts_offset = "offset" in set(inspect.signature(pipeline.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1
    pipeline.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    accepts_eta = "eta" in set(inspect.signature(pipeline.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    for i, t in enumerate(pipeline.scheduler.timesteps):
        # expand the latents for classifier free guidance
        latent_model_input = torch.cat([cond_latents] * 2)
        if isinstance(pipeline.scheduler, LMSDiscreteScheduler):
            sigma = pipeline.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # cfg
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        cond_latents = pipeline.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]

    # scale and decode the latents to get the image tensor
    image = pipeline.vae.decode(cond_latents / pipeline.vae.config.scaling_factor).sample

    # generate output numpy image as uint8
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).astype(np.uint8)

    return image

def load_pipeline():
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_PATH, scheduler=lms, safety_checker=None)
    if DATASET == 'MNIST':
        pipeline.load_lora_weights(SD_LORA_PATH)
    pipeline.to(TORCH_DEVICE)

    return pipeline

def get_pipeline():
    global _cached_pipeline
    if _cached_pipeline is None:
        _cached_pipeline = load_pipeline()
    return _cached_pipeline

def generate(prompt, seed=42, mutate='False',
             mutated_latent=None, num_inference_steps=40, guidance_scale=7.5, eta=0.0, width=512, height=512):
    '''
    create image by diffusion (for each digit)
    returns:
        cond_latent: latent space (1, 4, 64, 64)
        result img in numpy.ndarray shape:(512, 512, 3)
    '''
    #assert torch.cuda.is_available()
    assert height % 8 == 0 and width % 8 == 0
    pipeline = get_pipeline()
    text_input = pipeline.tokenizer(prompt, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    cond_embeddings = pipeline.text_encoder(text_input.input_ids.to(TORCH_DEVICE))[0]  # shape [1, 77, 768]
    if mutate == 'True':
        cond_latent = mutated_latent
    else:
        torch.manual_seed(seed)
        cond_latent = torch.randn((1, pipeline.unet.in_channels, height // 8, width // 8), device=TORCH_DEVICE)
    image = diffuse(pipeline, cond_embeddings, cond_latent, num_inference_steps, guidance_scale, eta)
    return cond_latent, image
"""
