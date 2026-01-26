# %%writefile evaluator.py
import numpy as np
import utils
from config import K, K_SD
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from individual import Individual


def evaluate_ff1(A, B):
    dist = utils.get_distance(A, B)
    return dist


def evaluate_ff2(confidence1, confidence2):
    P3 = confidence1 * confidence2

    if P3 < 0:
        P3 = -1
    return P3


def evaluate_aggregate_ff(sparseness, distance):
    result = sparseness - (K_SD * distance)
    return result


def dist_from_nearest_archived(ind: Individual, population, k):
    neighbors = list()
    for ind_pop in population:
        if ind_pop.id != ind.id:
            d = ind.distance(ind_pop)
            if d > 0.0:
                neighbors.append((d, ind_pop))

    if len(neighbors) == 0:
        assert len(population) > 0
        # assert (population[0].id == ind.id)
        return -1.0, ind

    neighbors = sorted(neighbors, key=lambda x: x[0])
    nns = neighbors[:k]
    # k > 1 is not handeled yet
    if k > 1:
        dist = np.mean(nns)
    elif k == 1:
        dist = nns[0][0]
    if dist == 0.0:
        print("bug")
    return dist, nns[0][1]


def evaluate_sparseness(ind: Individual, individuals: list[Individual]):
    N = len(individuals)
    # Sparseness is evaluated only if the archive is not empty
    # Otherwise the sparseness is 1
    if (N == 0) or (N == 1 and individuals[0] == ind):
        ind.sparseness = np.inf
        closest = ind
    elif N == 2:
        ind.sparseness, closest = dist_from_nearest_archived(ind, individuals, K)
        individuals[0].sparseness = ind.sparseness
        individuals[1].sparseness = ind.sparseness
    else:
        ind.sparseness, closest = dist_from_nearest_archived(ind, individuals, K)
    return ind.sparseness, closest


# Distance between two individuals for archive management
# This is different from the generic distance method in individual.py
# It considers the correctly classified and misclassified members separately
def eval_archive_dist(ind1, ind2):
    """
    Determines the distance between two individuals for archive management
    Args:
        ind1 (_type_): _description_
        ind2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if ind1.m1.predicted_label == ind1.m1.expected_label:
        ind1_correct = ind1.m1
        ind1_misclass = ind1.m2
    else:
        ind1_correct = ind1.m2
        ind1_misclass = ind1.m1

    if ind2.m1.predicted_label == ind2.m1.expected_label:
        ind2_correct = ind2.m1
        ind2_misclass = ind2.m2
    else:
        ind2_correct = ind2.m2
        ind2_misclass = ind2.m1

    dist1 = utils.get_distance(ind1_correct, ind2_correct)
    dist2 = utils.get_distance(ind1_misclass, ind2_misclass)

    dist = np.mean([dist1, dist2])
    return dist
