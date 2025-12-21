import numpy as np

class MnistMember:
    COUNT = 0

    def __init__(self, latent, expected_label):
        #self.seed = seed
        # latent vector
        self.image = None
        self.latent = latent
        # label
        self.expected_label = expected_label
        self.predicted_label = None
        self.confidence = None
        self.correctly_classified = None

    def clone(self):
        clone_digit = MnistMember(self.latent, self.expected_label)
        return clone_digit

    def cosine_similarity(self, other):
        return np.dot(self.latent, other.latent) / (np.linalg.norm(self.latent) * np.linalg.norm(other.latent))

    def image_distance(self, other):
        return np.linalg.norm(self.image - other.image)
        
