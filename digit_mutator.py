from math import exp
import mutation_manager
from config import DELTA

class DigitMutator:

    def __init__(self, digit):
        self.digit = digit
        #self.seed = digit.seed

    #TODO: attualmente non sempre si raggiunge il flipping del label
    # quindi si potrebbe aumentare il DELTA di base, oppure cambiare il calcolo della confidence (predictor.py)
    def mutate(self, prompt, step, noise_x, noise_y):
        # Intensità progressiva della mutazione
        # Fare in modo che più alta è la confidence e maggiore diventa il delta 
        base_delta = DELTA
        delta = base_delta * exp(self.digit.confidence)

        # Mutazione nel latent space
        #mutated_latent = mutation_manager.mutate(self.digit.latent, delta)

        # Circular walk mutation
        mutated_latent = mutation_manager.mutate_circular(step, noise_x, noise_y)

        # Generazione immagine dal latente mutato
        _, mutated_image = mutation_manager.generate(prompt=prompt, mutated_latent=mutated_latent)

        # Aggiornamento stato
        self.digit.latent = mutated_latent
        self.digit.image = mutated_image


    def generate(self, prompt):
        _, img = mutation_manager.generate(prompt, mutated_latent=self.digit.latent)
        self.digit.image = img