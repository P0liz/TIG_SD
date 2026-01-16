from os import makedirs
from os.path import exists, basename, join
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from folder import Folder
import numpy as np
import torch

# TODO: converto to use torch tensors instead of numpy arrays
# or remember to always convert to numpy arrays


def get_distance(v1, v2):
    if isinstance(v1, torch.Tensor):
        v1 = v1.detach().cpu().numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.detach().cpu().numpy()
    return np.linalg.norm(v1 - v2)


def print_archive(archive):
    dst = Folder.DST_ARC + "_DJ"
    if not exists(dst):
        makedirs(dst)
    for i, ind in enumerate(archive):
        filename1 = join(
            dst,
            basename(
                "archived_"
                + str(i)
                + "_mem1_l_"
                + str(ind.m1.predicted_label)
                + "_prompt_"
                + str(ind.prompt)
            ),
        )
        plt.imsave(
            filename1 + ".png",
            ind.m1.image_tensor.detach().cpu().numpy().squeeze(),
            cmap=cm.gray,
            format="png",
        )
        np.save(filename1, ind.m1.image_tensor.detach().cpu().numpy().squeeze())
        assert np.array_equal(
            ind.m1.image_tensor.detach().cpu().numpy().squeeze(),
            np.load(filename1 + ".npy"),
        )

        filename2 = join(
            dst,
            basename(
                "archived_"
                + str(i)
                + "_mem2_l_"
                + str(ind.m2.predicted_label)
                + "_prompt_"
                + str(ind.prompt)
            ),
        )
        plt.imsave(
            filename2 + ".png",
            ind.m2.image_tensor.detach().cpu().numpy().squeeze(),
            cmap=cm.gray,
            format="png",
        )
        np.save(filename2, ind.m2.image_tensor.detach().cpu().numpy().squeeze())
        assert np.array_equal(
            ind.m2.image_tensor.detach().cpu().numpy().squeeze(),
            np.load(filename2 + ".npy"),
        )


# TODO: understand why there is this and the previous one used both
def print_archive_experiment(archive):
    for ind in enumerate(archive):
        ind.export()


def get_radius_reference(solution, reference):
    # Calculate the distance between each misclassified digit and the seed (mindist metric)
    min_distances = list()
    for sol in solution:
        digit = sol.image_tensor
        dist = torch.linalg.norm(digit - reference).item()
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
                dist = torch.linalg.norm(d1.image_tensor - d2.image_tensor).item()
                if dist > maxdist:
                    maxdist = dist
        max_distances.append(maxdist)
    diameter = np.mean(max_distances)
    return diameter
