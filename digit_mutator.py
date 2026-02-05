# %%writefile digit_mutator.py
from math import exp
import mutation_manager
from config import DELTA, STANDING_STEP_LIMIT
from mnist_member import MnistMember


class DigitMutator:

    def __init__(self, digit):
        self.digit: MnistMember = digit

    def mutate(self, prompt):
        # def mutate(self, prompt, step, noise_x, noise_y): # Circular walk
        """
        Mutate the member assigning a delta value based on the number of standing steps
        The higher the standing steps, the higher the delta
        Args:
            prompt (_type_): _description_
        """
        # Progressive intensification of perturbation size
        # It is increased by one time every STANDING_STEP_LIMIT standing steps
        if self.digit.standing_steps >= STANDING_STEP_LIMIT:
            perturbation_size = DELTA * (self.digit.standing_steps / STANDING_STEP_LIMIT + 1)
        else:
            perturbation_size = DELTA
        print(f"Perturbation size: {perturbation_size:.3f}")

        # Mutazione nel latent space
        # TODO: consider clamping latent vector values to (-5,5) to avoid strange values
        mutated_latent = mutation_manager.mutate(self.digit.latent, perturbation_size)
        print(
            f"Latent stats - min: {mutated_latent.min():.2f}, max: {mutated_latent.max():.2f}, std: {mutated_latent.std():.2f}"
        )

        # Circular walk mutation
        # mutated_latent = mutation_manager.mutate_circular( self.digit.latent, step, noise_x, noise_y)

        # Generazione immagine dal latente mutato
        _, mutated_tensor, image = mutation_manager.generate(prompt, mutated_latent=mutated_latent)

        # Aggiornamento stato
        self.digit.latent = mutated_latent
        self.digit.image_tensor = mutated_tensor
        self.digit.image = image
        # Reset prediction status to trigger re-evaluation
        self.digit.reset()

    def generate(self, prompt, guidance_scale):
        _, mutated_tensor, image = mutation_manager.generate(prompt, self.digit.latent, guidance_scale)
        self.digit.image_tensor = mutated_tensor
        self.digit.image = image
