import torch
import random
import re

from folder import Folder
from individual import Individual
from predictor import Predictor
from digit_mutator import DigitMutator
from mnist_member import MnistMember
from mutation_manager import get_pipeline
from data_visualization import export_as_gif, plot_confidence, plot_distance
from config import DEVICE, HEIGHT, WIDTH, DTYPE, TRYNEW, STEPS


def main(prompt, expected_label, max_steps=STEPS):
    # Starting from a random latent noise vector
    latent = torch.randn(
        (1, get_pipeline().unet.config.in_channels, HEIGHT // 8, WIDTH // 8),
        device=DEVICE,
        dtype=DTYPE,
    )
    # Scala il latent secondo lo scheduler
    # latent = latent * get_pipeline().scheduler.init_noise_sigma

    digit1 = MnistMember(latent, expected_label)

    # Initial generation and validation
    # Higher guidance_scale to assure the prompt is followed correctly
    DigitMutator(digit1).generate(prompt, guidance_scale=3.5)
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
        ind = Individual(digit1, digit1)
        ind.export()
        print("Initial latent does not satisfy the label")
        return

    print(f"[000] " f"exp={digit1.expected_label} " f"conf={digit1.confidence:.3f}")
    digit2 = digit1.clone()  # second member of an Individual

    # Circular walk
    noise_x = torch.randn_like(latent)  # Direzione X (fissa)
    noise_y = torch.randn_like(latent)  # Direzione Y (fissa)

    images = []
    conficence_scores = []
    euc_img_dists = []
    euc_img_dists.append(0)
    latent_cos_sims = []
    latent_cos_sims.append(0)

    # Iterative mutation process
    for step in range(1, max_steps + 1):
        # Chose digit with lower confidence score (cause we want to change the digit's prediction)
        if digit1.confidence <= digit2.confidence:
            best_digit = digit1
            other_digit = digit2
        else:
            best_digit = digit2
            other_digit = digit1

        # After selecting best and before mutating, save values to draw/plot
        images.append(best_digit.image)
        conficence_scores.append(best_digit.confidence)

        DigitMutator(best_digit).mutate(prompt)
        # DigitMutator(digit).mutate(prompt, step, noise_x, noise_y)    # Circular walk
        prediction, confidence = Predictor.predict_single(best_digit, expected_label)

        best_digit.predicted_label = prediction
        best_digit.confidence = confidence
        if best_digit.expected_label == best_digit.predicted_label:
            best_digit.correctly_classified = True
        else:
            best_digit.correctly_classified = False

        cos_sim = best_digit.cosine_similarity(other_digit)
        latent_cos_sims.append(cos_sim)
        euc_dist = best_digit.image_distance(other_digit)
        euc_img_dists.append(euc_dist)
        print(
            f"[{step:03d}] "
            f"pred={best_digit.predicted_label} "
            f"conf={best_digit.confidence:.3f} "
            f"cos_sim={cos_sim:.3f} "
            f"euc_dist={euc_dist:.3f}"
        )

        if not best_digit.correctly_classified:
            print(
                f" Label flipped at step {step} "
                f"pred={best_digit.predicted_label} "
                f"exp={best_digit.expected_label}"
            )
            break

    ind = Individual(digit1, digit2)  # Create final individual to see results
    ind.misstep = step
    ind.misclass = best_digit.predicted_label
    ind.members_img_euc_dist = euc_dist
    ind.members_latent_cos_sim = cos_sim
    ind.export()
    base_path = f"{Folder.DST}"
    export_as_gif(
        f"{base_path}/individual_{Folder.run_id}.gif", images, rubber_band=True
    )
    plot_confidence(conficence_scores, f"{base_path}/confidence_{Folder.run_id}.png")
    plot_distance(
        euc_img_dists,
        f"{base_path}/euclidean_distance_{Folder.run_id}.png",
        "Euclidean distance",
    )
    plot_distance(
        latent_cos_sims,
        f"{base_path}/cosine_similarity_{Folder.run_id}.png",
        "Cosine similarity",
    )


if __name__ == "__main__":
    prompts = [
        "A photo of Z0ero Number0",
        "A photo of one1 Number1",
        "A photo of two2 Number2 ",
        "A photo of three3 Number3",
        "A photo of Four4 Number4",
        "A photo of Five5 Number5",
        "A photo of Six6 Number6",
        "A photo of Seven7 Number7 ",
        "A photo of Eight8 Number8",
        "A photo of Nine9 Number9",
    ]
    for i in range(0, 5):
        Folder.initialize()
        randprompt = random.choice(prompts)
        expected_label = int(re.search(r"Number(\d+)", randprompt).group(1))
        main(prompt=randprompt, expected_label=expected_label)
        print("GAME OVER")


# source ./venv/bin/activate
