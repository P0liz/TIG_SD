from math import exp
import mutation_manager
from config import DELTA


class DigitMutator:

    def __init__(self, digit):
        self.digit = digit
        # self.seed = digit.seed

    def mutate(self, prompt, step, noise_x, noise_y):
        # Intensità progressiva della mutazione
        # Fare in modo che più alta è la confidence e maggiore diventa il delta
        # Vedi anche predictor.py per modificare calcolo della confidence
        base_delta = DELTA
        delta = base_delta * exp(self.digit.confidence)

        # Mutazione nel latent space
        # mutated_latent = mutation_manager.mutate(self.digit.latent, delta)

        # Circular walk mutation
        mutated_latent = mutation_manager.mutate_circular(
            self.digit.latent, step, noise_x, noise_y
        )

        # Generazione immagine dal latente mutato
        _, mutated_tensor, image = mutation_manager.generate(
            prompt, mutated_latent=mutated_latent
        )

        # Aggiornamento stato
        self.digit.latent = mutated_latent
        self.digit.image_tensor = mutated_tensor
        self.digit.image = image

    def generate(self, prompt):
        _, mutated_tensor, image = mutation_manager.generate(
            prompt, mutated_latent=self.digit.latent
        )
        self.digit.image_tensor = mutated_tensor
        self.digit.image = image
