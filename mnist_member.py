import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import makedirs
from os.path import join
from folder import Folder
from utils import get_distance


class MnistMember:
    COUNT = 0

    def __init__(self, latent, expected_label):
        self.id = MnistMember.COUNT
        self.image = None  # PIL Image
        self.image_tensor = None  # Torch Tensor [1, 1, 28, 28] in grayscale
        self.latent = latent
        self.expected_label = expected_label
        self.predicted_label = None
        self.confidence = None
        self.confidence_history = []
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

    def cosine_similarity(self, other):
        # Flatten and use cosine_similarity function
        self_flat = self.latent.flatten().unsqueeze(0)
        other_flat = other.latent.flatten().unsqueeze(0)
        return torch.nn.functional.cosine_similarity(self_flat, other_flat).item()

    def image_distance(self, other):
        return torch.linalg.norm(self.image_tensor - other.image_tensor).item()

    def distance(self, other):
        return torch.linalg.norm(self.latent - other.latent).item()
