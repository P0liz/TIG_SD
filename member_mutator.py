# %%writefile digit_mutator.py
import mutation_manager
from config import DEVICE
from mnist_member import MnistMember
import torch
from diffusion import pipeline_manager


class MemberMutator:

    def __init__(self, prompt, member, mutation_step=4):
        self.prompt = prompt
        self.member: "MnistMember" = member
        self.initial_latent = member.og_latent
        self.mutation_step = mutation_step  # Step to cache
        self.cached_latent = None

        if pipeline_manager._mode == "custom":
            self.text_embeddings = pipeline_manager.text_embeddings(self.prompt)
            # First pass: denoise until mutation_step is reached, then cache it
            self.cache_denoising_steps()

    def cache_denoising_steps(self, guidance_scale=3.5):
        scheduler = pipeline_manager.scheduler
        latent = self.initial_latent * scheduler.init_noise_sigma
        scheduler.set_timesteps(self.inference_steps)

        print(f"Timesteps: {scheduler.timesteps}")
        print(f"Will cache at step {self.mutation_step}, timestep={scheduler.timesteps[self.mutation_step]}")

        for step_idx, t in enumerate(scheduler.timesteps):
            latent_model_input = torch.cat([latent] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = pipeline_manager.unet(
                    latent_model_input, t, encoder_hidden_states=self.text_embeddings
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latent = scheduler.step(noise_pred, t, latent).prev_sample

            # Cache latent at mutation_step
            if step_idx == self.mutation_step:
                self.cached_latent = latent.clone()
                break

    def clone(self):
        """Clone the mutator with a new member instance but shared cache"""
        cloned_member = self.member.clone()
        cloned_mutator = MemberMutator(self.prompt, cloned_member, self.mutation_step)

        # Shared cache between original and clone
        if pipeline_manager._mode == "custom":
            cloned_mutator.text_embeddings = self.text_embeddings
            cloned_mutator.cached_latent = self.cached_latent

        return cloned_mutator

    def denoise(self, isMutating=True, guidance_scale=2.5, generator=None):
        """Resume from mutation_step, apply noise, continue denoising"""
        scheduler = pipeline_manager.scheduler
        # TODO: custom timesteps?
        scheduler.set_timesteps(self.inference_steps)

        # Latent walk on cached vector
        if isMutating:
            # Mutate the cached latent vector
            perturbation_size = mutation_manager.calculate_perturbation_size(self.member.standing_steps)
            self.cached_latent = mutation_manager.mutate(self.cached_latent, perturbation_size, generator)
            # Reset prediction status to trigger re-evaluation
            self.member.reset()
        latent = self.cached_latent.clone()
        print(f"Latent stats - min: {latent.min():.2f}, max: {latent.max():.2f}, std: {latent.std():.2f}")

        # Only denoise from mutation_step onwards
        for t in scheduler.timesteps[self.mutation_step :]:
            latent_model_input = torch.cat([latent] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = pipeline_manager.unet(
                    latent_model_input, t, encoder_hidden_states=self.text_embeddings
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latent = scheduler.step(noise_pred, t, latent).prev_sample

        return latent

    def initial_mutation(self, generator=None):
        perturbation_size = mutation_manager.calculate_perturbation_size(self.member.standing_steps)

        # Latent mutation
        mutated_latent = mutation_manager.mutate(self.member.latent, perturbation_size, generator)
        print(
            f"Latent stats - min: {mutated_latent.min():.2f}, max: {mutated_latent.max():.2f}, std: {mutated_latent.std():.2f}"
        )
        # Update state
        self.member.latent = mutated_latent
        # Reset prediction status to trigger re-evaluation
        self.member.reset()

    def generate(self, guidance_scale=2.5, generator=None):
        assert pipeline_manager._mode == "standard", "generate() only works in standard mode"

        _, mutated_tensor, image = mutation_manager.generate(self.prompt, self.member.latent, guidance_scale, generator)
        # Update state
        self.member.image_tensor = mutated_tensor
        self.member.image = image

    def denoise_and_decode(self, isMutating=True, guidance_scale=2.5, generator=None):
        assert pipeline_manager._mode == "custom", "denoise_decode() only works in custom mode"
        # Denoise with optional mutation
        latent = self.denoise(isMutating, guidance_scale, generator)

        # Decode image
        image = pipeline_manager.decode_image(latent)

        # preprocess image to 28x28 grayscale tensor
        # Add batch dimension and move to device
        image_tensor = mutation_manager.process_image(image).unsqueeze(0).to(DEVICE)

        # Update state
        self.member.image = image
        self.member.latent = latent
        self.member.image_tensor = image_tensor
