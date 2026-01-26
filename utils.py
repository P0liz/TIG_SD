# %%writefile utils.py
import numpy as np
import torch
from config import DISTANCE_METRIC


def get_distance(m1, m2, metric=None):
    if metric is None:
        metric = DISTANCE_METRIC
    if metric == "latent_cosine":
        v1 = m1.latent.flatten().unsqueeze(0)
        v2 = m2.latent.flatten().unsqueeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(v1, v2)
        return (1 - cos_sim).item()  # Convert similarity to distance
    elif metric == "image_euclidean":
        return torch.linalg.norm(m1.image_tensor - m2.image_tensor).item()
    elif metric == "latent_euclidean":
        return torch.linalg.norm(m1.latent - m2.latent).item()
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def print_archive_experiment(archive):
    for ind in archive:
        ind.export()


def get_radius_reference(solution, reference):
    # Calculate the distance between each misclassified digit and the seed (mindist metric)
    min_distances = list()
    for sol in solution:
        dist = get_distance(sol, reference)
        min_distances.append(dist)
    mindist = np.mean(min_distances)
    return mindist


def get_diameter(solution):
    # Calculate the distance between each misclassified digit and the farthest element of the solution (diameter metric)
    max_distances = list()
    for d1 in solution:
        maxdist = float(0)
        for d2 in solution:
            if d1 != d2:
                dist = get_distance(d1, d2)
                if dist > maxdist:
                    maxdist = dist
        max_distances.append(maxdist)
    diameter = np.mean(max_distances)
    return diameter
