# %%writefile main_single_ind.py
import torch
import random
import re

from folder import Folder
from individual import Individual
from predictor import Predictor
from digit_mutator import DigitMutator
from mnist_member import MnistMember
from diffusion import pipeline_manager
from data_visualization import export_as_gif, plot_confidence, plot_distance
from config import *
import utils


def mutate_rand_digit(m1: "DigitMutator", m2: "DigitMutator", images1, images2):
    # Scegli membro casuale
    if random.getrandbits(1):
        selected_m = m1
        other_m = m2
        is1 = True
    else:
        selected_m = m2
        other_m = m1
        is1 = False

    selected_m.denoise_and_decode(isMutating=True)

    if is1:
        images1.append(selected_m.digit.image_tensor)
    else:
        images2.append(selected_m.digit.image_tensor)

    # Predici
    prediction, confidence = Predictor.predict_single(selected_m.digit, selected_m.digit.expected_label)
    selected_m.digit.confidence_history.append(confidence)

    return prediction, confidence, selected_m.digit, other_m.digit


def main(prompt, expected_label, max_steps=NGEN):
    # Initialize pipeline if not already done
    if not pipeline_manager._initialized:
        pipeline_manager.initialize(mode="custom")

    assert pipeline_manager._mode == "custom", "Pipeline must be in custom mode"

    # Force finding a valid initial latent
    while True:
        # Starting from a random latent noise vector
        initial_latent = torch.randn(
            (1, pipeline_manager.unet.config.in_channels, HEIGHT // 8, WIDTH // 8), device=DEVICE, dtype=DTYPE
        )

        digit1 = MnistMember(initial_latent, expected_label)

        # Initial generation and validation
        # Higher guidance_scale to assure the prompt is followed correctly
        mutator1 = DigitMutator(digit1, initial_latent)
        mutator1.denoise_and_decode(isMutating=False, guidance_scale=3.5)
        prediction, confidence = Predictor.predict_single(digit1, expected_label)

        # Initial assignment
        digit1.predicted_label = prediction
        digit1.confidence = confidence
        if digit1.expected_label == digit1.predicted_label:
            digit1.correctly_classified = True
        else:
            digit1.correctly_classified = False

        if not digit1.correctly_classified:
            print(prompt, " - exp: ", expected_label)
            print(f"pred={digit1.predicted_label} " f"exp={digit1.expected_label}")
            print("Initial latent does not satisfy the label")
        else:
            break

    print(f"[000] " f"exp={digit1.expected_label} " f"conf={digit1.confidence:.3f}")

    # second member of an Individual
    mutator2 = mutator1.clone()
    digit2 = mutator2.digit

    images1 = []
    images2 = []
    euc_img_dists = []
    euc_img_dists.append(0)
    euc_dists = []
    euc_dists.append(0)
    latent_cos_sims = []
    latent_cos_sims.append(0)

    # Iterative mutation process
    for step in range(1, max_steps + 1):
        # Mutation
        prediction, confidence, selected_digit, other_digit = mutate_rand_digit(mutator1, mutator2, images1, images2)

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
        print(
            f"[{step:03d}] "
            f"pred={selected_digit.predicted_label} "
            f"conf={selected_digit.confidence:.3f} "
            f"cos_sim={cos_sim:.3f} "
            f"euc_dist={euc_dist:.3f} "
            f"euc_img_dist={img_euc_dist:.3f}"
        )

        if not selected_digit.correctly_classified:
            print(
                f" Label flipped at step {step} "
                f"pred={selected_digit.predicted_label} "
                f"exp={selected_digit.expected_label}"
            )
            break

    ind = Individual(digit1, digit2, prompt, None)  # Create final individual to see results
    ind.misstep = step
    ind.bad_prediction = selected_digit.predicted_label
    ind.members_distance = euc_dist
    ind.members_img_euc_dist = img_euc_dist
    ind.members_latent_cos_sim = cos_sim
    ind.export()
    base_path = f"{Folder.DST}"
    export_as_gif(f"{base_path}/individual_{Folder.run_id}_1.gif", images1, rubber_band=True)
    export_as_gif(f"{base_path}/individual_{Folder.run_id}_2.gif", images2, rubber_band=True)
    plot_confidence(digit1.confidence_history, f"{base_path}/confidence_{Folder.run_id}_1.png")
    plot_confidence(digit2.confidence_history, f"{base_path}/confidence_{Folder.run_id}_2.png")
    plot_distance(euc_dists, f"{base_path}/euclidean_distance_{Folder.run_id}.png", "Euclidean distance")
    plot_distance(latent_cos_sims, f"{base_path}/cosine_similarity_{Folder.run_id}.png", "Cosine similarity")
    plot_distance(
        euc_img_dists, f"{base_path}/euclidean_image_distance_{Folder.run_id}.png", "Euclidean image distance"
    )


if __name__ == "__main__":

    Folder.initialize()
    randprompt = random.choice(PROMPTS)
    expected_label = int(re.search(r"Number(\d+)", randprompt).group(1))
    main(prompt=randprompt, expected_label=expected_label)
    print("GAME OVER")


# source ./venv/bin/activate
