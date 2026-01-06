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
        clone_digit.image = self.image
        clone_digit.predicted_label = self.predicted_label
        clone_digit.confidence = self.confidence
        clone_digit.correctly_classified = self.correctly_classified
        return clone_digit

    def cosine_similarity(self, other):
        # Flatten and use cosine_similarity function
        self_flat = self.latent.flatten().unsqueeze(0)
        other_flat = other.latent.flatten().unsqueeze(0)
        return torch.nn.functional.cosine_similarity(self_flat, other_flat).item()

    def image_distance(self, other):
        return torch.linalg.norm(self.image - other.image)
        
