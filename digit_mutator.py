from math import exp
import mutation_manager
from config import DELTA


class DigitMutator:

    def __init__(self, digit):
        self.digit = digit

    def mutate(self, prompt):
        """
        Mutate the member assigning a delta value based on the number of standing steps
        The higher the standing steps, the higher the delta
        Args:
            prompt (_type_): _description_
        """
        # def mutate(self, prompt, step, noise_x, noise_y): # Circular walk
        # Progressive intensification of perturbation size
        # It is increased by one time every 5 standing steps
        base_delta = DELTA
        if self.digit.standing_steps >= 5:
            base_delta = DELTA * (self.digit.standing_steps / 5 + 1)
        # perturbation_size = base_delta * exp(self.digit.confidence)   # Old method
        perturbation_size = base_delta  # understand if this should change
        print(f"perturbation_size: {perturbation_size}")

        # Mutazione nel latent space
        mutated_latent = mutation_manager.mutate(self.digit.latent, perturbation_size)

        # Circular walk mutation
        # mutated_latent = mutation_manager.mutate_circular( self.digit.latent, step, noise_x, noise_y)

        # Generazione immagine dal latente mutato
        _, mutated_tensor, image = mutation_manager.generate(
            prompt, mutated_latent=mutated_latent
        )

        # Aggiornamento stato
        self.digit.latent = mutated_latent
        self.digit.image_tensor = mutated_tensor
        self.digit.image = image

    def generate(self, prompt, guidance_scale):
        _, mutated_tensor, image = mutation_manager.generate(
            prompt,
            self.digit.latent,
            guidance_scale,
        )
        self.digit.image_tensor = mutated_tensor
        self.digit.image = image
