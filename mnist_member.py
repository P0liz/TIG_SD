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
        self.image = None
        self.image_tensor = None
        self.latent = latent
        self.expected_label = expected_label
        self.predicted_label = None
        self.confidence = None
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

    def export(self, ind_id=None):
        # Create directory
        if ind_id is not None:
            member_dir = join(Folder.DST_IND, f"ind{ind_id}")
        else:
            member_dir = join(Folder.DST_IND, f"member{self.id}")
        makedirs(member_dir, exist_ok=True)

        # Save metadata
        data = {
            "id": str(self.id),
            "expected_label": str(self.expected_label),
            "predicted_label": str(self.predicted_label),
            "confidence": str(self.confidence),
            "correctly_classified": str(self.correctly_classified),
        }
        with open(join(member_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=4)

        # Save image
        base_name = f"member{self.id}_l_{self.predicted_label}"
        png_path = join(member_dir, base_name + ".png")
        npy_path = join(member_dir, base_name + ".npy")

        img_np = self.image_tensor.detach().cpu().numpy().squeeze()
        plt.imsave(png_path, img_np, cmap=cm.gray, format="png")
        np.save(npy_path, img_np)
