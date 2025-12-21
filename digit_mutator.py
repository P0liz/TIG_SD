from math import exp
import mutation_manager

class DigitMutator:

    def __init__(self, digit):
        self.digit = digit
        #self.seed = digit.seed

    def mutate(self, prompt):
        # Intensità progressiva della mutazione
        # TODO: cambiare delta in base alla fitness 
        base_delta = 0.01
        delta = base_delta * exp(-self.digit.confidence)

        # Mutazione nel latent space
        mutated_latent = mutation_manager.mutate(self.digit.latent, delta)

        # Generazione immagine dal latente mutato
        _, mutated_image = mutation_manager.generate(prompt=prompt, mutated_latent=mutated_latent)

        # Aggiornamento stato
        self.digit.latent = mutated_latent
        self.digit.image = mutated_image


    def generate(self, prompt):
        _, img = mutation_manager.generate(prompt, mutated_latent=self.digit.latent)
        return img