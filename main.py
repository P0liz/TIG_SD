import random
import numpy as np
import torch
from deap import base, creator, tools
from deap.tools.emo import selNSGA2

# NSGA-II fa:
# 1. Non-dominated sorting (crea fronti di Pareto)
# 2. Crowding distance (misura diversità)
# 3. Seleziona: fronti migliori + individui più isolati

from mnist_member import MnistMember
from digit_mutator import DigitMutator
from predictor import Predictor
from timer import Timer
import archive_manager
import utils
from individual import Individual
from mutation_manager import get_pipeline
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
)

PROMPTS = [
    "A photo of Z0ero Number0",
    "A photo of one1 Number1",
    "A photo of two2 Number2",
    "A photo of three3 Number3",
    "A photo of Four4 Number4",
    "A photo of Five5 Number5",
    "A photo of Six6 Number6",
    "A photo of Seven7 Number7",
    "A photo of Eight8 Number8",
    "A photo of Nine9 Number9",
]

pipe = get_pipeline()

# DEAP framework setup.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", Individual, fitness=creator.FitnessMulti)

# Global individual variables
Individual.COUNT = 0
Individual.USED_LABELS = set()


class GeneticAlgorithm:

    def __init__(self, rand_seed=None):
        self.archive = archive_manager.Archive()

        # Keep deterministic outputs
        if rand_seed is not None:
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)

    # ========================================================================
    # Generation
    # ========================================================================

    def generate_member(
        self, prompt, expected_label, latent, guidance_scale=3.5, max_attempts=10
    ):
        """
        Generate a member with Stable Diffusion and Validate it if possible
        Args:
            prompt (_type_): _description_
            expected_label (_type_): _description_
            latent (_type_): _description_
            guidance_scale (float, optional): _description_. Defaults to 3.5.
            max_attempts (int, optional): _description_. Defaults to 10.

        Returns:
            member ore None if failure
        """
        for i in range(max_attempts):
            # Generate member and classify it
            member = MnistMember(latent, expected_label)
            DigitMutator(member).generate(prompt, guidance_scale=guidance_scale)
            prediction, confidence = Predictor.predict_single(member, expected_label)

            # Validation
            if prediction == expected_label:
                member.predicted_label = prediction
                member.confidence = confidence
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
        Individual.COUNT += 1

        if label is None:
            label = random.randint(0, 9)
        prompt = PROMPTS[label]

        latent = torch.randn(
            (1, pipe.unet.config.in_channels, HEIGHT // 8, WIDTH // 8),
            device=DEVICE,
            dtype=DTYPE,
        )

        # Generate members
        m1 = self.generate_member(prompt, label, latent)
        if m1 is None:
            # Riprova con guidance più alta
            m1 = self.generate_member(prompt, label, latent, guidance_scale=5.0)
            if m1 is None:
                raise ValueError(f"Cannot create individual for label {label}")

        m2 = m1.clone()

        # Create individual
        individual = creator.Individual(m1, m2, prompt, latent)
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
                label = i % 10
                ind = self.create_individual(label=label)
                population.append(ind)

                print(
                    f"  [{i+1}/{size}] label={ind.m1.expected_label}, "
                    f"conf={ind.m1.confidence:.3f}"
                )

            except ValueError as e:
                print(f"  ✗ Failed to create individual {i+1}: {e}")
                continue

        return population

    # ========================================================================
    # Mutation
    # ========================================================================
    def mutate_individual(self, individual):
        """
        Mutate a single digit chosen randomly and keep the mutation either way it goes
        Keep track of standing steps to increase mutation size if needed
        Save confidence history and distances for plotting
        Args:
            individual (_type_): _description_
        """
        # Chose random member
        if random.getrandbits(1):
            member_to_mutate = individual.m1
            other_member = individual.m2
            ism1 = True
        else:
            member_to_mutate = individual.m2
            other_member = individual.m1
            ism1 = False

        if ism1:
            individual.m1.confidence_history.append(member_to_mutate.confidence)
        else:
            individual.m2.confidence_history.append(member_to_mutate.confidence)

        # Mutate and predict
        prompt = PROMPTS[member_to_mutate.expected_label]
        DigitMutator(member_to_mutate).mutate(prompt)
        individual.reset()

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
        individual.members_distances.append(individual.members_distance)
        individual.members_img_euc_dists.append(individual.members_img_euc_dist)
        individual.members_latent_cos_sims.append(individual.members_latent_cos_sim)

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate_batch(self, individuals):
        """
        Evaluate a batch of individuals.
        Args:
            individuals (_type_): _description_
        """
        members_to_predict = []

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
            # If confidence diff is too low and the new confidence is higher then...
            if abs(conf - member.confidence) <= 0.01 and conf >= member.confidence:
                member.standing_steps += 1
            else:
                member.standing_steps = 0

            member.predicted_label = pred
            member.confidence = conf
            if member.expected_label == pred:
                member.correctly_classified = True
            else:
                member.correctly_classified = False

    def evaluate_fitness(self, individuals):
        for ind in individuals:
            ind.evaluate(self.archive.get_archive())
            ind.fitness.values = (ind.aggregate_ff, ind.misclass)

    def clone_individual(self, individual):
        # Clone an individual (deep copy)
        # Use the new constructor to include the fitness field
        new_ind = creator.Individual(
            individual.m1.clone(),
            individual.m2.clone(),
            individual.prompt,
            individual.original_noise.clone(),
        )
        new_ind.members_distance = individual.members_distance
        new_ind.members_img_euc_dist = individual.members_img_euc_dist
        new_ind.members_latent_cos_sim = individual.members_latent_cos_sim
        new_ind.members_distances = list(individual.members_distances)
        new_ind.members_img_euc_dists = list(individual.members_img_euc_dists)
        new_ind.members_latent_cos_sims = list(individual.members_latent_cos_sims)
        new_ind.prompt = individual.prompt
        return new_ind

    # TODO: See later
    def reseed_population(self, population, n_reseed):
        """
        Sostituisce gli individui peggiori con nuovi individui
        usando label non ancora esplorate.
        """
        if n_reseed == 0:
            return population

        # Trova label non usate
        all_labels = set(range(10))
        unused_labels = all_labels - Individual.USED_LABELS

        # Sostituisci gli ultimi n_reseed (i peggiori dopo selezione)
        for i in range(n_reseed):
            idx = len(population) - 1 - i

            if len(unused_labels) > 0:
                new_label = random.choice(list(unused_labels))
                unused_labels.remove(new_label)
            else:
                new_label = random.randint(0, 9)

            try:
                population[idx] = self.create_individual(label=new_label)
                print(f"Reseeded individual {idx} with label {new_label}")
            except ValueError:
                print(f"Failed to reseed individual {idx}")

        return population

    def update_archive(self, individuals):
        for ind in individuals:
            if ind.archive_candidate:
                self.archive.update_archive(ind)

    # ========================================================================
    # Run
    # ========================================================================

    def run(self):
        # Stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("avg", np.mean, axis=0)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max", "avg"

        print("Generating initial population")
        population = self.create_population(POPSIZE)
        if len(population) < POPSIZE:
            print(f"Warning: only {len(population)}/{POPSIZE} individuals created")

        # Initial evaluation
        self.evaluate_batch(population)
        self.evaluate_fitness(population)
        self.update_archive(population)
        # This is just to assign the crowding distance to the individuals (no actual selection is done)
        population = selNSGA2(population, len(population))

        record = stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)
        print(f"Gen 0: {logbook.stream}")

        # Begin the generational process
        gen = 1
        while gen <= NGEN:
            print(f"### GENERATION {gen}")

            # 1. Select future parents
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.clone_individual(ind) for ind in offspring]

            # 2. Reseeding (optional)
            # See later cause I have to understand if it makes sense,
            # and if it does, is it better to keep or delete duplicates?
            """
            if len(self.archive.get_archive()) > 0 and gen % 10 == 0:
                n_reseed = random.randint(
                    1, min(RESEEDUPPERBOUND, len(population) // 4)
                )
                population = self.reseed_population(population, n_reseed)
            """

            # 3. Mutation
            print(f"Mutating {len(offspring)} offspring...")
            for ind in offspring:
                self.mutate_individual(ind)
                del ind.fitness.values

            # 4. Evaluation
            all_individuals = population + offspring
            self.evaluate_batch(all_individuals)
            self.evaluate_fitness(all_individuals)
            self.update_archive(all_individuals)
            print(f"Archive size: {len(self.archive.get_archive())}")

            # 5. Survival selection (NSGA-II)
            # Using len(population) because it could differ from POPSIZE (in case of reseeding)
            population = selNSGA2(all_individuals, len(population))
            print(f"Survived: {len(population)} individuals")

            # 6. Statistiche
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(all_individuals), **record)
            print(f"Gen {gen}: {logbook.stream}")

            # 7. Debug report
            if DJ_DEBUG and gen % STEPSIZE == 0:
                self.archive.create_report(Individual.USED_LABELS, gen)

            gen += 1

            # Stop condition
            if STOP_CONDITION == "time" and not Timer.has_budget():
                print("Time budget exhausted")
                break

        # Ending process
        self.archive.create_report(Individual.USED_LABELS, "final")

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

    Folder.initialize()

    # Crea e esegui GA
    ga = GeneticAlgorithm(rand_seed=7)
    population, archive = ga.run()

    # Report finale
    print("\n### FINAL ARCHIVE")
    from utils import print_archive, print_archive_experiment

    # TODO: change how an ind is saved (atm doubled info)
    print_archive_experiment(archive.get_archive())
    # TODO: why both functions?
    # print_archive(archive.get_archive())

    print("GAME OVER")
