# %%writefile member.py
import utils


class Member:

    def __init__(self, latent, expected_label):
        self.image = None  # PIL Image in rgb
        self.image_tensor = None  # Torch Tensor
        self.og_latent = latent.clone()  # Original latent vector (pure gaussian noise)
        self.latent = latent  # Latent vector modified through latent walk by mutation
        self.denoised_latent = None  # Latent vector after denoising
        self.expected_label = expected_label
        self.predicted_label = None
        self.confidence = None
        self.confidence_history = []  # plotting confidence over time
        self.correctly_classified = None
        self.standing_steps = 0

    def clone(self):
        clone_member = Member(self.latent, self.expected_label)
        clone_member.image = self.image
        clone_member.image_tensor = self.image_tensor
        clone_member.denoised_latent = self.denoised_latent
        clone_member.predicted_label = self.predicted_label
        clone_member.confidence = self.confidence
        clone_member.confidence_history = list(self.confidence_history)
        clone_member.correctly_classified = self.correctly_classified
        clone_member.standing_steps = self.standing_steps
        return clone_member

    def reset(self):
        self.predicted_label = None
        self.correctly_classified = None

    def distance(self, other):
        return utils.get_distance(self, other)
