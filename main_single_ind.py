# %%writefile main_single_ind.py
import torch
import random
import re

from timer import Timer
from folder import Folder
from individual import Individual
from predictor import Predictor
from member_mutator import MemberMutator
from member import Member
from diffusion import pipeline_manager
from data_visualization import export_as_gif, plot_distance
from config import *
import utils

# Local configuration
MODE = "standard"  # Or "custom"
SEED = random.randint(0, 2**32 - 1)


def mutate_rand_member(m1: "MemberMutator", m2: "MemberMutator", images1, images2, generator=None):
    # Scegli membro casuale
    if random.getrandbits(1):
        selected_m = m1
        other_m = m2
        is1 = True
    else:
        selected_m = m2
        other_m = m1
        is1 = False

    if MODE == "custom":
        selected_m.denoise_and_decode(isMutating=True, generator=generator)
    elif MODE == "standard":
        selected_m.initial_mutation(generator)
        selected_m.generate(generator=generator)
    else:
        raise ValueError("Unknown mode")

    if is1:
        images1.append(selected_m.members[0].image_tensor)
    else:
        images2.append(selected_m.members[0].image_tensor)

    # Predici
    prediction, confidence = Predictor.predict_single(selected_m.members[0], selected_m.members[0].expected_label)
    selected_m.members[0].confidence_history.append(confidence)

    # If confidence diff is too low and the or confidence is higher then...
    if (
        abs(confidence - selected_m.members[0].confidence) <= CONF_CHANGE
        or confidence >= selected_m.members[0].confidence
    ):
        selected_m.members[0].standing_steps += 1
    else:
        selected_m.members[0].standing_steps = 0

    return prediction, confidence, selected_m.members[0], other_m.members[0]


def main(prompt, expected_label, max_steps=NGEN):
    # Set all random seeds
    torch.manual_seed(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # Initialize pipeline if not already done
    if not pipeline_manager._initialized:
        pipeline_manager.initialize(mode=MODE)

    if MODE == "custom":
        unet_channels = pipeline_manager.unet.config.in_channels
    elif MODE == "standard":
        unet_channels = pipeline_manager.pipe.unet.config.in_channels
    else:
        raise ValueError("Unknown mode")

    # Force finding a valid initial latent
    tries = 0
    while True:
        generator = torch.Generator(device=DEVICE).manual_seed(tries)
        # Starting from a random latent noise vector
        initial_latent = torch.randn(
            (1, unet_channels, HEIGHT // 8, WIDTH // 8), device=DEVICE, dtype=DTYPE, generator=generator
        )
        member1 = Member(initial_latent, expected_label)

        # Initial generation and validation
        # Higher guidance_scale to assure the prompt is followed correctly
        mutator1 = MemberMutator(prompt, member1)
        if MODE == "custom":
            mutator1.denoise_and_decode(isMutating=False, guidance_scale=3.5, generator=generator)
        elif MODE == "standard":
            mutator1.generate(guidance_scale=3.5, generator=generator)
        else:
            raise ValueError("Unknown mode")
        prediction, confidence = Predictor.predict_single(member1, expected_label)

        # Initial assignment
        member1.predicted_label = prediction
        member1.confidence = confidence
        member1.confidence_history.append(member1.confidence)
        if member1.expected_label == member1.predicted_label:
            member1.correctly_classified = True
        else:
            member1.correctly_classified = False

        if not member1.correctly_classified:
            print(prompt, " - exp: ", expected_label)
            print(f"pred={member1.predicted_label} " f"exp={member1.expected_label}")
            print("Initial latent does not satisfy the label")
            tries += 1
        else:
            break

    # second member of an Individual
    mutator2 = mutator1.clone()
    member2 = mutator2.members[0]

    print(f"[000] " f"exp={member1.expected_label} " f"conf={member1.confidence:.3f}")

    images1 = [member1.image_tensor]
    images2 = [member2.image_tensor]
    euc_img_dists = []
    euc_img_dists.append(0)
    euc_dists = []
    euc_dists.append(0)
    latent_cos_sims = []
    latent_cos_sims.append(0)

    # Iterative mutation process
    iteration_times = []
    for step in range(1, max_steps + 1):

        # Mutation
        prediction, confidence, selected_digit, other_digit = mutate_rand_member(
            mutator1, mutator2, images1, images2, generator
        )

        selected_digit.predicted_label = prediction
        selected_digit.confidence = confidence
        if selected_digit.expected_label == selected_digit.predicted_label:
            selected_digit.correctly_classified = True
        else:
            selected_digit.correctly_classified = False

        cos_sim = utils.get_distance(selected_digit, other_digit, "latent_cosine")
        latent_cos_sims.append(cos_sim)
        euc_dist = utils.get_distance(selected_digit, other_digit, "latent_euclidean")
        euc_dists.append(euc_dist)
        img_euc_dist = utils.get_distance(selected_digit, other_digit, "image_euclidean")
        euc_img_dists.append(img_euc_dist)

        elapsed = Timer.get_elapsed_time()
        elapsed_seconds = elapsed.total_seconds()
        iteration_times.append(elapsed_seconds)
        print(
            f"[{step:03d}] "
            f"pred={selected_digit.predicted_label} "
            f"conf={selected_digit.confidence:.3f} "
            f"cos_sim={cos_sim:.3f} "
            f"euc_dist={euc_dist:.3f} "
            f"euc_img_dist={img_euc_dist:.3f} "
            f"time={elapsed_seconds}s"
        )

        if not selected_digit.correctly_classified:
            print(
                f" Label flipped at step {step} "
                f"pred={selected_digit.predicted_label} "
                f"exp={selected_digit.expected_label}"
            )
            break

    # Timing stats
    print(f"\nTiming stats:")
    print(f"  Total time: {sum(iteration_times)}s")
    print(f"  Avg per iteration: {sum(iteration_times)/len(iteration_times)}s")
    print(f"  Min: {min(iteration_times)}s, Max: {max(iteration_times)}s")

    ind = Individual(member1, member2, prompt, None)  # Create final individual to see results
    ind.misstep = step
    ind.bad_prediction = selected_digit.predicted_label
    ind.members_distance = euc_dist
    ind.members_img_euc_dist = img_euc_dist
    ind.members_latent_cos_sim = cos_sim
    ind.export()
    base_path = f"{Folder.DST}"
    export_as_gif(f"{base_path}/individual_{Folder.run_id}_1.gif", images1, rubber_band=True)
    export_as_gif(f"{base_path}/individual_{Folder.run_id}_2.gif", images2, rubber_band=True)
    plot_distance(euc_dists, f"{base_path}/euclidean_distance_{Folder.run_id}.png", "Euclidean distance")
    plot_distance(latent_cos_sims, f"{base_path}/cosine_similarity_{Folder.run_id}.png", "Cosine similarity")
    plot_distance(
        euc_img_dists, f"{base_path}/euclidean_image_distance_{Folder.run_id}.png", "Euclidean image distance"
    )
    plot_distance(iteration_times, f"{base_path}/iteration_times_{Folder.run_id}.png", "Iteration Time (seconds)")


if __name__ == "__main__":
    import config

    Folder.initialize()
    Timer.initialize()
    random.seed(SEED)
    randprompt = random.choice(PROMPTS)
    if DATASET == "mnist":
        expected_label = int(re.search(r"Number(\d+)", randprompt).group(1))
    elif DATASET == "imagenet":
        expected_label = IMAGENET_LABEL
    else:
        raise ValueError("Unsupported dataset specified in config")

    main(prompt=randprompt, expected_label=expected_label)
    print("GAME OVER")


# source ./venv/bin/activate
