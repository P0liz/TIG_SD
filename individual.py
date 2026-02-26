# %%writefile individual.py
import json
from os import makedirs
from os.path import join

import numpy as np
from PIL import Image
from numpy import mean
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import evaluator
from folder import Folder
from data_visualization import plot_confidence, plot_distance
from config import DATASET, DISTANCE_METRIC


class Individual:
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    USED_LABELS = set()

    def __init__(self, member1, member2, prompt, latent):
        Individual.COUNT += 1
        self.id = Individual.COUNT
        self.prompt = prompt
        self.original_noise = latent
        self.members_distance = 0
        self.members_distances = [0]
        self.members_img_euc_dist = 0
        self.members_img_euc_dists = [0]
        self.members_latent_cos_sim = 0
        self.members_latent_cos_sims = [0]
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.archive_candidate = None
        self.m1 = member1
        self.m2 = member2
        self.misstep = 0
        self.bad_prediction = None

    def reset(self):
        # do not reset id (same throughout life)
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.archive_candidate = None

    def to_dict(self):
        return {
            "id": str(self.id),
            "m1": {
                "expected_label": str(self.m1.expected_label),
                "predicted_label": str(self.m1.predicted_label),
                "confidence": str(self.m1.confidence),
            },
            "m2": {
                "expected_label": str(self.m2.expected_label),
                "predicted_label": str(self.m2.predicted_label),
                "confidence": str(self.m2.confidence),
            },
            "members_distance": str(self.members_distance),
            "members_img_euc_dist": str(self.members_img_euc_dist),
            "members_latent_cos_sim": str(self.members_latent_cos_sim),
            "aggregate_ff": str(self.aggregate_ff),
            "sparseness": str(self.sparseness),
            "misstep": str(self.misstep),
            "bad_prediction": str(self.bad_prediction),
        }

    def export(self):
        # Base directory for individuals
        makedirs(Folder.DST_IND, exist_ok=True)

        # Individual directory
        ind_dir = join(Folder.DST_IND, f"ind{self.id}")
        makedirs(ind_dir, exist_ok=True)
        # Save metadata
        filedest = join(ind_dir, "data.json")
        with open(filedest, "w") as f:
            json.dump(self.to_dict(), f, sort_keys=False, indent=2)

        # --- Helper to save a member ---
        def save_member(member, idx):
            base_name = f"m{idx}_pred{member.predicted_label}"
            png_path = join(ind_dir, base_name + ".png")
            conf_path = join(ind_dir, base_name + "_conf.png")
            latent_path = join(ind_dir, f"m{idx}_latent.npy")
            image_path = join(ind_dir, f"m{idx}_image.npy")

            lat_np = member.latent.detach().cpu().numpy().squeeze()

            # Save image (visual)
            if DATASET == "mnist":
                # Rememeber to save image from the preprocessed tensor (already 28x28 grayscale)
                img_np = member.image_tensor.detach().cpu().numpy().squeeze()
                plt.imsave(png_path, img_np, cmap=cm.gray, format="png", vmin=0, vmax=1)
            elif DATASET == "imagenet":
                img_np = save_imagenet_image_np(member, png_path)
                member.image.save(png_path)  # PIL save
            else:
                raise ValueError("Unsupported dataset specified in config")

            # Save latent and image as .npy files (raw tensors)
            np.save(latent_path, lat_np)
            np.save(image_path, img_np)

            # Save confidence plot
            plot_confidence(member.confidence_history, conf_path)

            # Consistency check
            assert np.array_equal(img_np, np.load(image_path))

        # Save members
        save_member(self.m1, idx=1)
        save_member(self.m2, idx=2)

        # Plotting distances
        dist_path = join(ind_dir, "members_distance.png")
        dist_img_path = join(ind_dir, "members_img_euc_dist.png")
        cos_sim_path = join(ind_dir, "members_latent_cos_sim.png")
        plot_distance(self.members_distances, dist_path, "Members Latent Euclidean Distance")
        plot_distance(self.members_img_euc_dists, dist_img_path, "Members Image Euclidean Distance")
        plot_distance(self.members_latent_cos_sims, cos_sim_path, "Members Latent Cosine Similarity")

    def evaluate(self, archive, step):
        if self.misclass is None:
            # Calculate fitness function 2
            self.misclass = evaluator.evaluate_ff2(self.m1.confidence, self.m2.confidence)

            if self.m1.correctly_classified != self.m2.correctly_classified:
                self.archive_candidate = True
                self.misstep = step
                self.bad_prediction = (
                    self.m1.predicted_label if not self.m1.correctly_classified else self.m2.predicted_label
                )

        # Get the appropriate distance based on config
        distance_attr = {
            "latent_euclidean": "members_distance",
            "image_euclidean": "members_img_euc_dist",
            "latent_cosine": "members_latent_cos_sim",
        }[DISTANCE_METRIC]

        # Calculate fitness function 1 (distance between members)
        setattr(self, distance_attr, evaluator.evaluate_ff1(self.m1, self.m2))

        # Recalculate sparseness at each iteration (to reflect changes in archive)
        self.sparseness, _ = evaluator.evaluate_sparseness(self, archive)
        if self.sparseness == 0.0:
            print(self.sparseness)
            print("BUG")

        self.aggregate_ff = evaluator.evaluate_aggregate_ff(self.sparseness, getattr(self, distance_attr))

        return self.aggregate_ff, self.misclass

    def mutate(self):
        raise NotImplemented()

    # Generic distance between two individuals
    def distance(self, i2):
        i1 = self
        a = i1.m1.distance(i2.m1)
        b = i1.m1.distance(i2.m2)
        c = i1.m2.distance(i2.m1)
        d = i1.m2.distance(i2.m2)

        dist = mean([min(a, b), min(c, d), min(a, c), min(b, d)])
        return dist


def save_imagenet_image_np(member, png_path):
    img = member.image_tensor.detach().cpu()
    # Rimuovi batch se presente
    if img.ndim == 4:
        img = img[0]
    # Se formato CHW → converti a HWC
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    img_np = img.numpy()
    """
    # Denormalizza ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    plt.imsave(png_path, img_np, vmin=0, vmax=1)
    """
    return img_np
