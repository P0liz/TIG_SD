# %%writefile main.py
import random
import numpy as np
import torch
from deap import base, creator, tools
from deap.tools.emo import selNSGA2

from mnist_member import MnistMember
from digit_mutator import DigitMutator
from predictor import Predictor
from timer import Timer
import archive_manager
import utils
from individual import Individual
from diffusion import get_pipeline
from config import (
    NGEN,
    POPSIZE,
    RESEEDUPPERBOUND,
    STOP_CONDITION,
    STEPSIZE,
    DJ_DEBUG,
    DEVICE,
    HEIGHT,
    WIDTH,
    DTYPE,
    RESEED_INTERVAL,
    PROMPTS,
    INITIALPOP,
    ARCHIVE_TYPE,
    CONF_CHANGE,
)


pipe = get_pipeline()

# DEAP framework setup.
# Maximize aggregate_ff, minimize misclass
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", Individual, fitness=creator.FitnessMulti)


class GeneticAlgorithm:

    def __init__(self):
        self.archive = archive_manager.Archive()

    # ========================================================================
    # Generation
    # ========================================================================

    def generate_member(
        self, prompt, expected_label, guidance_scale=3.5, max_attempts=10
    ):
        """
        Generate a member with Stable Diffusion and Validate it if possible
        Starting with higher guidance_scale to get more realistic outputs
        Args:
            prompt (_type_): _description_
            expected_label (_type_): _description_
            latent (_type_): _description_
            guidance_scale (float, optional): set how strongly prompt is followed. Defaults to 3.5
            max_attempts (int, optional): _description_. Defaults to 10.

        Returns:
            member ore None if failure
        """
        for i in range(max_attempts):
            # Generate random latent at every attempt (sometimes too much noise)
            latent = torch.randn(
                (1, pipe.unet.config.in_channels, HEIGHT // 8, WIDTH // 8),
                device=DEVICE,
                dtype=DTYPE,
            )
            # Generate member and classify it
            member = MnistMember(latent, expected_label)
            DigitMutator(member).generate(prompt, guidance_scale=guidance_scale)
            prediction, confidence = Predictor.predict_single(member, expected_label)

            # Validation
            if prediction == expected_label:
                member.predicted_label = prediction
                member.confidence = confidence
                member.confidence_history.append(confidence)
                member.correctly_classified = True
                return member
            else:
                continue

        print(f"Failed to generate valid member for label {expected_label}")
        return None

    def create_individual(self, label=None):
        """
        Args:
            label (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            new basic individual, composed a couple of members
        """

        if label is None:
            label = random.randint(0, 9)
        prompt = PROMPTS[label]

        # Generate members
        m1 = self.generate_member(prompt, label)
        if m1 is None:
            # Riprova con guidance più alta
            m1 = self.generate_member(prompt, label, guidance_scale=4.5)
            if m1 is None:
                raise ValueError(f"Cannot create individual for label {label}")

        m2 = m1.clone()

        # Create individual
        individual = creator.Individual(m1, m2, prompt, m1.latent.clone())
        Individual.USED_LABELS.add(label)

        return individual

    def create_population(self, size):
        """
        Args:
            size (_type_): _description_

        Returns:
            create initial population of individuals
        """
        population = []

        print(f"Creating initial population of {size} individuals...")

        for i in range(size):
            try:
                # Use different labels
                if INITIALPOP == "random":
                    label = random.randint(0, len(PROMPTS) - 1)
                elif INITIALPOP == "sequence":
                    label = i % len(PROMPTS)
                else:
                    break
                ind = self.create_individual(label=label)
                population.append(ind)

                print(
                    f"  [{i+1}/{size}] label={ind.m1.predicted_label}, "
                    f"conf={ind.m1.confidence:.3f}"
                )

            except ValueError as e:
                print(f"Failed to create individual {i+1}: {e}")
                # Need to fill the population anyway to avoid errors later
                population.append(self.create_individual())  # create with random label
                continue

        if len(population) != POPSIZE:
            raise ValueError(f"Invalid INITIALPOP value: {INITIALPOP}")
        return population

    # ========================================================================
    # Mutation
    # ========================================================================
    def mutate_individual(self, individual: "Individual"):
        """
        Mutate a single digit chosen randomly and keep the mutation either way it goes
        Keep track of standing steps to increase mutation size if needed
        Args:
            individual (_type_): _description_
        """
        # Chose random member
        if random.getrandbits(1):
            member_to_mutate = individual.m1
            other_member = individual.m2
        else:
            member_to_mutate = individual.m2
            other_member = individual.m1

        # Mutate and predict
        prompt = PROMPTS[member_to_mutate.expected_label]
        DigitMutator(member_to_mutate).mutate(prompt)
        individual.reset()  # reset fitness-related fields

        # Update distances
        individual.members_distance = utils.get_distance(
            member_to_mutate, other_member, "latent_euclidean"
        )
        individual.members_img_euc_dist = utils.get_distance(
            member_to_mutate, other_member, "image_euclidean"
        )
        individual.members_latent_cos_sim = utils.get_distance(
            member_to_mutate, other_member, "latent_cosine"
        )

    # ========================================================================
    # Selection
    # ========================================================================
    # TODO: probably does not work >>> not enough inds of a label to reach archived status
    # Should try with a bigger population
    def select_non_dominant_population(self, individuals: list["Individual"]):
        MAX_IND_PER_LABEL = POPSIZE // 3  # Max one third of the population
        kept = []
        overflown = []
        sorted_pops = selNSGA2(individuals, len(individuals))

        # Selecting only a limited number of individuals per label
        # selNSGA2 orders by best fronts so the first elements are the best
        label_count = [0] * len(PROMPTS)
        for ind in sorted_pops:
            label = ind.m1.expected_label
            label_count[label] += 1
            if label_count[label] <= MAX_IND_PER_LABEL:
                kept.append(ind)
            else:
                overflown.append(ind)

        # Assuring the population to return matches POPSIZE
        if len(kept) < POPSIZE:
            n_missing = POPSIZE - len(kept)
            if len(overflown) >= n_missing:
                # Fill with individuals from overflown
                kept.extend(overflown[:n_missing])
            else:
                # Not enough overflown, fill with new individuals
                n_missing = POPSIZE - len(kept)
                for i in range(n_missing):
                    new_ind = self.create_individual()
                    kept.append(new_ind)
        elif len(kept) > POPSIZE:
            # Trim to exactly POPSIZE (keep best ones, already sorted)
            kept = kept[:POPSIZE]
        return kept

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate_batch(self, individuals: list["Individual"]):
        """
        Evaluate a batch of individuals.
        Args:
            individuals (_type_): _description_
        """
        members_to_predict: list[MnistMember] = []

        for ind in individuals:
            if ind.m1.predicted_label is None:
                members_to_predict.append(ind.m1)
            if ind.m2.predicted_label is None:
                members_to_predict.append(ind.m2)
        if len(members_to_predict) == 0:
            return
        batch_labels = [m.expected_label for m in members_to_predict]

        predictions, confidences = Predictor.predict(members_to_predict, batch_labels)

        # Assign results
        for member, pred, conf in zip(members_to_predict, predictions, confidences):
            member.predicted_label = pred
            member.confidence = conf
            if member.expected_label == pred:
                member.correctly_classified = True
            else:
                member.correctly_classified = False
            print(
                f"exp: {member.expected_label} -> pred: {pred} (confidence: {conf:.3f})"
            )

            # If confidence diff is too low and the new confidence is higher then...
            if (
                abs(conf - member.confidence) <= CONF_CHANGE
                and conf >= member.confidence
            ):
                member.standing_steps += 1
            else:
                member.standing_steps = 0

    def evaluate_fitness(self, individuals: list["Individual"], gen=0):
        for ind in individuals:
            ind.evaluate(self.archive.get_archive(), gen)
            ind.fitness.values = (ind.aggregate_ff, ind.misclass)

        """ # Normalize fitness values
        # Get min/max dynamically
        agg_ff_values = [ind.aggregate_ff for ind in individuals]
        misclass_values = [ind.misclass for ind in individuals]
        min_agg = min(agg_ff_values)
        max_agg = max(agg_ff_values)
        min_mis = min(misclass_values)
        max_mis = max(misclass_values)

        # Normalize and assign 
        for ind in individuals:
            norm_agg_ff = (
                (ind.aggregate_ff - min_agg) / (max_agg - min_agg)
                if max_agg > min_agg
                else 0
            )
            norm_misclass = (
                (ind.misclass - min_mis) / (max_mis - min_mis)
                if max_mis > min_mis
                else 0
            )
            ind.fitness.values = (norm_agg_ff, norm_misclass)
        """

    def clone_individual(self, individual: "Individual"):
        # Clone an individual (deep copy)
        # Use the new constructor to include the fitness field
        new_ind: Individual = creator.Individual(
            individual.m1.clone(),
            individual.m2.clone(),
            individual.prompt,
            individual.original_noise.clone(),
        )
        new_ind.members_distance = individual.members_distance
        new_ind.members_img_euc_dist = individual.members_img_euc_dist
        new_ind.members_latent_cos_sim = individual.members_latent_cos_sim
        new_ind.prompt = individual.prompt
        # for plotting
        new_ind.members_distances = list(individual.members_distances)
        new_ind.members_img_euc_dists = list(individual.members_img_euc_dists)
        new_ind.members_latent_cos_sims = list(individual.members_latent_cos_sims)
        return new_ind

    # TODO: test different strategy >> prompte prompts and latents that are already in the archive
    # which means less diversity but higher chance of acceptance
    def reseed_population(self, population, n_reseed):
        """
        Reseed n_reseed individuals in the population to promote diversity
        Delete the worst individuals and replace them with new ones
        """
        if n_reseed == 0:
            return population

        # Find labels which are not in the archive
        all_labels = set(range(10))
        unused_labels = all_labels - self.archive.archived_labels

        # Substitute the last n_reseed (worst after selection)
        for i in range(n_reseed):
            idx = len(population) - 1 - i

            # Promote diversity: use unused labels first
            if len(unused_labels) > 0:
                new_label = random.choice(list(unused_labels))
                unused_labels.remove(new_label)
            else:
                new_label = random.randint(0, 9)

            try:
                population[idx] = self.create_individual(label=new_label)
                print(
                    f"Reseeding: Added new individual with label {population[idx].m1.expected_label}"
                )
            except ValueError:
                print(
                    f"Failed to reseed individual {population[idx].id}, keeping the old one"
                )

        # Population cut
        count = [0] * len(PROMPTS)
        for pop in population:
            idx = pop.m1.expected_label
            count[idx] += 1
            # if a label is equal or over 80% of population remove all those individuals
            if count[idx] >= len(population) * 0.8:
                # remove all individuals with that label
                population = [ind for ind in population if ind.m1.expected_label != idx]
                print(f"Reseeding: Cut population to remove label {idx}")
                # add new ones
                added = POPSIZE - len(population)
                try:
                    for i in range(added):
                        unused_labels.remove(idx)
                        if len(unused_labels) > 0:
                            new_label = random.choice(list(unused_labels))
                            unused_labels.remove(new_label)
                        else:
                            new_label = random.randint(0, 9)
                        new_ind = self.create_individual(label=new_label)
                        population.append(new_ind)
                except ValueError:
                    print(f"Failed to create new individual after population cut")
                finally:
                    print(f"Reseeding: Added {added} individuals after population cut")
                    break

        return population

    def update_archive(self, individuals: list["Individual"]):
        print("Updating archive...")
        for ind in individuals:
            if ind.archive_candidate:
                if ARCHIVE_TYPE == "size":
                    self.archive.update_size_based_archive(ind)
                elif ARCHIVE_TYPE == "dist":
                    self.archive.update_dist_based_archive(ind)
                elif ARCHIVE_TYPE == "bucket":
                    self.archive.update_bucket_archive(ind)
                else:
                    raise ValueError(f"Invalid ARCHIVE_TYPE value: {ARCHIVE_TYPE}")

    # Called at each gen, even if data is not modified
    def update_data_to_plot(self, individuals: list["Individual"]):
        for ind in individuals:
            ind.m1.confidence_history.append(ind.m1.confidence)
            ind.m2.confidence_history.append(ind.m2.confidence)
            ind.members_distances.append(ind.members_distance)
            ind.members_img_euc_dists.append(ind.members_img_euc_dist)
            ind.members_latent_cos_sims.append(ind.members_latent_cos_sim)

    # ========================================================================
    # Run
    # ========================================================================

    def run(self):
        # Stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max", "avg", "std"

        print("Generating initial population")
        population: list["Individual"] = self.create_population(POPSIZE)
        if len(population) < POPSIZE:
            print(f"Warning - only {len(population)}/{POPSIZE} individuals created")

        # Initial evaluation
        self.evaluate_batch(population)
        self.evaluate_fitness(population)
        self.update_archive(population)
        # This is just to assign the crowding distance to the individuals (no actual selection is done)
        population = selNSGA2(population, len(population))

        record = stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)
        print(f"Gen 0: {logbook.stream}")
        # to plot label distribution
        labels_history = {0: [ind.m1.expected_label for ind in population]}

        if DJ_DEBUG:
            self.archive.create_report(
                Individual.USED_LABELS, generation=0, logbook=logbook
            )

        # Begin the generational process
        gen = 1
        while gen <= NGEN:
            print(f"### GENERATION {gen}")

            # 1. Select future parents
            offspring = tools.selTournamentDCD(population, POPSIZE)
            offspring = [self.clone_individual(ind) for ind in offspring]

            # 2. Reseeding
            if len(self.archive.get_archive()) > 0 and gen % RESEED_INTERVAL == 0:
                n_reseed = random.randint(1, RESEEDUPPERBOUND)
                population = self.reseed_population(population, n_reseed)

            # 3. Mutation
            print(f"Mutating {len(offspring)} offspring...")
            for i, ind in enumerate(offspring):
                print(f"[{i+1}/{len(offspring)}]")
                self.mutate_individual(ind)
                del ind.fitness.values

            # 4. Evaluation
            all_individuals = population + offspring
            self.evaluate_batch(all_individuals)
            self.evaluate_fitness(all_individuals, gen)
            self.update_archive(all_individuals)
            print(f"Archive size: {len(self.archive.get_archive())}")

            # 5. Survival selection (NSGA-II)
            population = selNSGA2(all_individuals, POPSIZE)
            # population = self.select_non_dominant_population(all_individuals)
            print(f"Survived: {len(population)} individuals")
            print(" ".join(str(ind.m1.expected_label) for ind in population))

            # 6. Stats
            self.update_data_to_plot(population)  # for plotting dists and confs
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(all_individuals), **record)
            print(f"Gen {gen}: {logbook.stream}")
            # to plot label distribution
            labels_history[gen] = [ind.m1.expected_label for ind in population]

            # 7. Debug report
            if DJ_DEBUG and gen % STEPSIZE == 0:
                self.archive.create_report(Individual.USED_LABELS, gen, logbook)

            gen += 1

            # Stop condition
            if STOP_CONDITION == "time" and not Timer.has_budget():
                print("Time budget exhausted")
                break

        # Ending process
        self.archive.create_report(
            Individual.USED_LABELS, "final", logbook, labels_history
        )

        print(f"Final statistics:")
        print(f"Individuals created: {Individual.COUNT}")
        print(f"Labels explored: {Individual.USED_LABELS}")
        print(f"Archive size: {len(self.archive.get_archive())}")
        print(f"Final population: {len(population)}")

        return population, self.archive


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    from folder import Folder
    from utils import print_archive_experiment

    Folder.initialize()

    ga = GeneticAlgorithm()
    population, archive = ga.run()

    print("\n### FINAL ARCHIVE")
    print_archive_experiment(archive.get_archive())
    print("GAME OVER")
