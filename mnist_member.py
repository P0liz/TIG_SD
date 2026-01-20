# %%writefile mnist_member.py
import utils


class MnistMember:

    def __init__(self, latent, expected_label):
        self.image = None  # PIL Image in rgb
        self.image_tensor = None  # Torch Tensor [1, 1, 28, 28] in grayscale
        self.latent = latent
        self.expected_label = expected_label
        self.predicted_label = None
        self.confidence = None
        self.confidence_history = []  # for plotting confidence over time
        self.correctly_classified = None
        self.standing_steps = 0

    def clone(self):
        clone_member = MnistMember(self.latent, self.expected_label)
        clone_member.image = self.image
        clone_member.image_tensor = self.image_tensor
        clone_member.predicted_label = self.predicted_label
        clone_member.confidence = self.confidence
        clone_member.confidence_history = list(self.confidence_history)  # deep copy
        clone_member.correctly_classified = self.correctly_classified
        return clone_member

    def reset(self):
        self.predicted_label = None
        self.confidence = None
        self.correctly_classified = None

    def distance(self, other):
        return utils.get_distance(self, other)
