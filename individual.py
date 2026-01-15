import json
from os import makedirs
from os.path import join, exists
from posixpath import basename

import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import evaluator
from folder import Folder


class Individual:
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    USED_LABELS = set()

    def __init__(self, member1, member2):
        self.id = Individual.COUNT
        self.prompt = None
        self.members_distance = None
        self.members_img_euc_dist = None
        self.members_latent_cos_sim = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.archive_candidate = None
        self.m1 = member1
        self.m2 = member2
        self.misstep = 0

    # TODO: why would I need this?
    def reset(self):
        self.id = Individual.COUNT
        self.members_distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.archive_candidate = None

    def to_dict(self):
        return {
            "id": str(self.id),
            "expected_label": str(self.m1.expected_label),
            "misclass": str(self.misclass),
            "steps": str(self.misstep),
            "members_distance": str(self.members_distance),
            "members_img_euc_dist": str(self.members_img_euc_dist),
            "members_latent_cos_sim": str(self.members_latent_cos_sim),
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
            json.dump(self.to_dict(), f, sort_keys=False, indent=4)

        # --- Helper to save a member ---
        def save_member(member, idx):
            base_name = f"archived_{idx}_mem{idx}_l_{member.predicted_label}"
            png_path = join(ind_dir, base_name + ".png")
            npy_path = join(ind_dir, base_name + ".npy")

            img_np = member.image_tensor.detach().cpu().numpy().squeeze()

            # Save image (visual)
            plt.imsave(png_path, img_np, cmap=cm.gray, format="png")

            # Save raw tensor (scientific)
            np.save(npy_path, img_np)

            # Consistency check
            assert np.array_equal(img_np, np.load(npy_path))

        # Save members
        save_member(self.m1, idx=1)
        save_member(self.m2, idx=2)

    """ old version
    def export(self):
        if not exists(Folder.DST_IND):
            makedirs(Folder.DST_IND)
        dst = join(Folder.DST_IND, "ind"+str(self.id))
        data = self.to_dict()
        filedest = dst + ".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))
        # Save member images and latents
        filename1 = join(dst, basename(
            'archived_' + str(1) +
            '_mem1_l_' + str(self.m1.predicted_label)))
        img_np = self.m1.image_tensor.detach().cpu().numpy()
        plt.imsave(filename1, img_np,
                   cmap=cm.gray,
                   format='png')
        np.save(filename1, img_np)
        assert (np.array_equal(img_np,
                               np.load(filename1 + '.npy')))

        filename2 = join(dst, basename(
            'archived_' + str(2) +
            '_mem2_l_' + str(self.m2.predicted_label)))
        img_np = self.m2.image_tensor.detach().cpu().numpy()
        plt.imsave(filename2, img_np,
                   cmap=cm.gray,
                   format='png')
        np.save(filename2, img_np)
        assert (np.array_equal(img_np,
                               np.load(filename2 + '.npy')))
    """

    def evaluate(self, archive):
        self.sparseness = None

        if self.misclass is None:
            # Calculate fitness function 2
            self.misclass = evaluator.evaluate_ff2(
                self.m1.confidence, self.m2.confidence
            )

            self.archive_candidate = (
                self.m1.correctly_classified != self.m2.correctly_classified
            )

        if self.members_distance is None:
            # Calculate fitness function 1
            self.members_distance = evaluator.evaluate_ff1(
                self.m1.purified, self.m2.purified
            )

        # Recalculate sparseness at each iteration
        self.sparseness = evaluator.evaluate_sparseness(self, archive)
        if self.sparseness == 0.0:
            print(self.sparseness)
            print("BUG")

        self.aggregate_ff = evaluator.evaluate_aggregate_ff(
            self.sparseness, self.members_distance
        )

        return self.aggregate_ff, self.misclass

    def mutate(self):
        raise NotImplemented()

    def distance(self, i2):
        i1 = self
        a = i1.m1.distance(i2.m1)
        b = i1.m1.distance(i2.m2)
        c = i1.m2.distance(i2.m1)
        d = i1.m2.distance(i2.m2)

        dist = mean([min(a, b), min(c, d), min(a, c), min(b, d)])
        return dist
