import utils


class MnistMember:
    COUNT = 0

    def __init__(self, latent, expected_label):
        self.id = MnistMember.COUNT
        self.image = None  # PIL Image in rgb
        self.image_tensor = None  # Torch Tensor [1, 1, 28, 28] in grayscale
        self.latent = latent
        self.expected_label = expected_label
        self.predicted_label = None
        self.confidence = None
        self.confidence_history = []  # for plotting confidence over time
        self.correctly_classified = None
        self.standing_steps = 0
        MnistMember.COUNT += 1

    def clone(self):
        clone_digit = MnistMember(self.latent, self.expected_label)
        clone_digit.image = self.image
        clone_digit.image_tensor = self.image_tensor
        clone_digit.predicted_label = self.predicted_label
        clone_digit.confidence = self.confidence
        clone_digit.correctly_classified = self.correctly_classified
        return clone_digit

    def distance(self, other):
        return utils.get_distance(self, other)
