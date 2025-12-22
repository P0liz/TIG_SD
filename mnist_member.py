import torch

class MnistMember:
    COUNT = 0

    def __init__(self, latent, expected_label):
        #self.seed = seed
        self.id = MnistMember.COUNT
        # latent vector
        self.image = None
        self.latent = latent
        # label
        self.expected_label = expected_label
        self.predicted_label = None
        self.confidence = None
        self.correctly_classified = None
        MnistMember.COUNT += 1

    def clone(self):
        clone_digit = MnistMember(self.latent, self.expected_label)
        return clone_digit

    def cosine_similarity(self, other):
        return torch.dot(self.latent, other.latent) / (torch.norm(self.latent) * torch.norm(other.latent))

    def image_distance(self, other):
        return torch.linalg.norm(self.image - other.image)
        
