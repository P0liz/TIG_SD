# %%writefile archive_manager.py
import csv
import json
from os.path import join

from folder import Folder
from timer import Timer
from utils import get_distance, get_diameter, get_radius_reference
from evaluator import eval_archive_dist, evaluate_sparseness
import numpy as np
from individual import Individual

from config import (
    ARCHIVE_THRESHOLD,
    POPSIZE,
    NGEN,
    RESEEDUPPERBOUND,
    K_SD,
    CLASSIFIER_WEIGHTS_PATH,
    STOP_CONDITION,
    RUNTIME,
    REPORT_NAME,
    STEPSIZE,
    DISTANCE_METRIC,
    TARGET_SIZE,
)


class Archive:

    def __init__(self):
        self.archive = list()
        self.archived_labels = set()
        self.target_size = TARGET_SIZE
        self.distance_input = {
            "latent_euclidean": "members_distance",
            "image_euclidean": "members_img_euc_dist",
            "latent_cosine": "members_latent_cos_sim",
        }[DISTANCE_METRIC]

    def get_archive(self):
        return self.archive

    def update_size_based_archive(self, ind: Individual):
        if ind not in self.archive:
            # archive is empty
            if len(self.archive) == 0:
                print(
                    f"ind {ind.id} with exp->{ind.m1.expected_label} and pred->({ind.m1.predicted_label},{ind.m2.predicted_label}, sparseness {ind.sparseness} and distance {getattr(ind, self.distance_input)} added to archive"
                )
                self.archive.append(ind)
                self.archived_labels.add(ind.m1.expected_label)
            else:
                # Find the member of the archive that is closest to the candidate.
                d_min = evaluate_sparseness(ind, self.archive)
                # archive is not full
                if len(self.archive) / self.target_size < 1:
                    # not the same sparseness
                    if d_min > 0:
                        print(
                            f"ind {ind.id} with exp->{ind.m1.expected_label} and pred->({ind.m1.predicted_label},{ind.m2.predicted_label}, sparseness {ind.sparseness} and distance {getattr(ind, self.distance_input)} added to archive"
                        )
                        self.archive.append(ind)
                        self.archived_labels.add(ind.m1.expected_label)

                # archive is full
                else:
                    # find the individual with the most distant members and smallest sparseness
                    c: Individual = sorted(
                        self.archive,
                        key=lambda x: (getattr(x, self.distance_input), -x.sparseness),
                        reverse=True,
                    )[0]
                    # replace c if ind has closer members
                    if getattr(c, self.distance_input) > getattr(
                        ind, self.distance_input
                    ):
                        print(
                            f"ind {ind.id} with exp->{ind.m1.expected_label} and pred->({ind.m1.predicted_label},{ind.m2.predicted_label}, sparseness {ind.sparseness} and distance {getattr(ind, self.distance_input)} added to archive"
                        )
                        print(
                            f"ind {c.id} with exp->{c.m1.expected_label} and pred->({c.m1.predicted_label},{c.m2.predicted_label}, sparseness {c.sparseness} and distance {getattr(c, self.distance_input)} removed from archive"
                        )
                        self.archive.remove(c)
                        self.archive.append(ind)
                        self.archived_labels.add(ind.m1.expected_label)
                    elif getattr(c, self.distance_input) == getattr(
                        ind, self.distance_input
                    ):
                        # ind has better performance
                        if ind.misclass < c.misclass:
                            print(
                                f"ind {ind.id} with exp->{ind.m1.expected_label} and pred->({ind.m1.predicted_label},{ind.m2.predicted_label}, sparseness {ind.sparseness} and distance {getattr(ind, self.distance_input)} added to archive"
                            )
                            print(
                                f"ind {c.id} with exp->{c.m1.expected_label} and pred->({c.m1.predicted_label},{c.m2.predicted_label}, sparseness {c.sparseness} and distance {getattr(c, self.distance_input)} removed from archive"
                            )
                            self.archive.remove(c)
                            self.archive.append(ind)
                            self.archived_labels.add(ind.m1.expected_label)
                        # c and ind have the same performance
                        elif ind.misclass == c.misclass:
                            # ind has better sparseness
                            if d_min > c.sparseness:
                                print(
                                    f"ind {ind.id} with exp->{ind.m1.expected_label} and pred->({ind.m1.predicted_label},{ind.m2.predicted_label}, sparseness {ind.sparseness} and distance {getattr(ind, self.distance_input)} added to archive"
                                )
                                print(
                                    f"ind {c.id} with exp->{c.m1.expected_label} and pred->({c.m1.predicted_label},{c.m2.predicted_label}, sparseness {c.sparseness} and distance {getattr(c, self.distance_input)} removed from archive"
                                )
                                self.archive.remove(c)
                                self.archive.append(ind)
                                self.archived_labels.add(ind.m1.expected_label)

    def update_dist_based_archive(self, ind: Individual):
        if ind not in self.archive:
            if len(self.archive) == 0:
                self.archive.append(ind)
                self.archived_labels.add(ind.m1.expected_label)
                print("Added first member to the archive")
            else:
                # Find the member of the archive that is closest to the candidate.
                closest_archived = None
                d_min = np.inf
                i = 0
                # TODO: why not just use sparseness evaluation here?
                while i < len(self.archive):
                    distance_archived = eval_archive_dist(ind, self.archive[i])
                    if distance_archived < d_min:
                        closest_archived = self.archive[i]
                        d_min = distance_archived
                        print(f"New min distance: {d_min}")
                    i += 1
                # Decide whether to add the candidate to the archive
                # Verify whether the candidate is close to the existing member of the archive
                # Note: 'close' is defined according to a user-defined threshold
                if d_min <= ARCHIVE_THRESHOLD:
                    # The candidate replaces the closest archive member if its members' distance is better
                    print(
                        "Candidate is close to an existing archive member, updating archive..."
                    )
                    print(
                        f"Old ind labels: exp->{closest_archived.m1.expected_label}, pred->{closest_archived.m1.predicted_label}, {closest_archived.m2.predicted_label}"
                    )
                    print(
                        f"New ind labels: exp->{ind.m1.expected_label}, pred->{ind.m1.predicted_label}, {ind.m2.predicted_label}"
                    )
                    dist_ind = getattr(ind, self.distance_input)
                    dist_archived_ind = get_distance(
                        closest_archived.m1,
                        closest_archived.m2,
                    )
                    if dist_ind <= dist_archived_ind:
                        self.archive.remove(closest_archived)
                        self.archive.append(ind)
                        self.archived_labels.add(ind.m1.expected_label)
                else:
                    # Add the candidate to the archive if it is distant from all the other archive members
                    print("Adding new member to the archive...")
                    print(
                        f"New ind labels: exp->{ind.m1.expected_label}, pred->{ind.m1.predicted_label}, {ind.m2.predicted_label}"
                    )
                    self.archive.append(ind)
                    self.archived_labels.add(ind.m1.expected_label)

    # TODO: review the entire method
    def create_report(self, labels, generation, logbook=None):
        # Retrieve the solutions belonging to the archive.
        if generation == STEPSIZE:
            dst = join(Folder.DST, REPORT_NAME)
            with open(dst, mode="w") as report_file:
                report_writer = csv.writer(
                    report_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                # TODO: add stuff
                report_writer.writerow(
                    [
                        "run",
                        "iteration",
                        "timestamp",
                        "archive_len",
                        "total_seeds",
                        "covered_seeds",
                        "final seeds",
                        "members_dist_min",
                        "members_dist_max",
                        "members_dist_avg",
                        "members_dist_std",
                        # "radius_ref_out",
                        # "radius_ref_in",
                        "diameter_out",
                        "diameter_in",
                    ]
                )
        solution = [ind for ind in self.archive]
        n = len(solution)

        # Obtain misclassified member of an individual on the frontier.
        outer_frontier = []
        # Obtain correctly classified member of an individual on the frontier.
        inner_frontier = []
        for ind in solution:
            if ind.m1.predicted_label != ind.m1.expected_label:
                misclassified_member = ind.m1
                correct_member = ind.m2
            else:
                misclassified_member = ind.m2
                correct_member = ind.m1
            outer_frontier.append(misclassified_member)
            inner_frontier.append(correct_member)

        out_radius = None
        in_radius = None
        out_radius_ref = None
        in_radius_ref = None
        out_diameter = None
        in_diameter = None
        stats = [None] * 4
        final_seeds = []
        if len(solution) > 0:
            # reference_filename = "ref_digit/cinque_rp.npy"
            # reference = np.load(reference_filename)
            out_diameter = get_diameter(outer_frontier)
            in_diameter = get_diameter(inner_frontier)
            final_seeds = self.get_seeds()
            stats = self.get_dist_members()
            # TODO: what for?
            # out_radius_ref = get_radius_reference(outer_frontier, reference)
            # in_radius_ref = get_radius_reference(inner_frontier, reference)

        if STOP_CONDITION == "iter":
            budget = NGEN
        elif STOP_CONDITION == "time":
            budget = RUNTIME
        else:
            budget = "no budget"
        # TODO: add stuff
        config = {
            "popsize": str(POPSIZE),
            "budget": str(budget),
            "budget_type": str(STOP_CONDITION),
            "archive tshd": str(ARCHIVE_THRESHOLD),
            "reseed": str(RESEEDUPPERBOUND),
            "K": str(K_SD),
            "model": str(CLASSIFIER_WEIGHTS_PATH),
        }

        dst = join(Folder.DST, "config.json")

        # dst = RESULTS_PATH + '/config.json'
        with open(dst, "w") as f:
            (json.dump(config, f, sort_keys=True, indent=4))

        dst = join(Folder.DST, REPORT_NAME)
        with open(dst, mode="a") as report_file:
            report_writer = csv.writer(
                report_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            timestamp, elapsed_time = Timer.get_timestamps()
            report_writer.writerow(
                [
                    Folder.run_id,
                    str(generation),
                    str(elapsed_time),
                    str(n),
                    str(len(labels)),
                    str(len(self.archived_labels)),
                    str(len(final_seeds)),
                    str(stats[0]),
                    str(stats[1]),
                    str(stats[2]),
                    str(stats[3]),
                    # str(out_radius_ref),
                    # str(in_radius_ref),
                    # str(out_radius),
                    # str(in_radius),
                    str(out_diameter),
                    str(in_diameter),
                ]
            )
        # Save logbook as CSV table
        if logbook is not None:
            logbook_dst = join(Folder.DST, "logbook.csv")
            # Always write all records (overwrites each time with full history)
            with open(logbook_dst, mode="w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "gen",
                        "evals",
                        "min_obj1",
                        "min_obj2",
                        "max_obj1",
                        "max_obj2",
                        "avg_obj1",
                        "avg_obj2",
                    ]
                )
                for record in logbook:
                    writer.writerow(
                        [
                            record["gen"],
                            record["evals"],
                            record["min"][0],
                            record["min"][1],
                            record["max"][0],
                            record["max"][1],
                            record["avg"][0],
                            record["avg"][1],
                        ]
                    )

    def get_seeds(self):
        seeds = set()
        for ind in self.get_archive():
            seeds.add(ind.prompt)
        return seeds

    def get_dist_members(self):
        distances = list()
        stats = [None] * 4
        for ind in self.get_archive():
            distances.append(getattr(ind, self.distance_input))

        stats[0] = np.min(distances)
        stats[1] = np.max(distances)
        stats[2] = np.mean(distances)
        stats[3] = np.std(distances)
        return stats
