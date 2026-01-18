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
from config import DEVICE, HEIGHT, WIDTH, DTYPE, TRYNEW, STEPS, MUTATION_TYPE


# Mutate only the one with the lower confidence and keep the mutation only if it is better (lower)
def single_mutation_conf(
    prompt, digit1, digit2, expected_label, images, confidence_scores
):
    # Chose digit with lower confidence score (cause we want to change the digit's prediction)
    if digit1.confidence <= digit2.confidence:
        selected_digit = digit1
        other_digit = digit2
    else:
        selected_digit = digit2
        other_digit = digit1

    # Save data pre mutation
    images.append(selected_digit.image_tensor)
    confidence_scores.append(selected_digit.confidence)
    pre_mutation_digit = selected_digit.clone()

    DigitMutator(selected_digit).mutate(prompt)
    # DigitMutator(digit).mutate(prompt, step, noise_x, noise_y)    # Circular walk
    prediction, confidence = Predictor.predict_single(selected_digit, expected_label)

    # Revert mutation in case confidence gets higher
    if confidence > pre_mutation_digit.confidence:
        if selected_digit is digit1:
            digit1 = pre_mutation_digit
        else:
            digit2 = pre_mutation_digit
        return None

    return prediction, confidence, selected_digit, other_digit


def single_mutation_random(
    prompt, digit1, digit2, images1, conf_scores1, images2, conf_scores2
):
    """
    Mutate a single digit chosen randomly and keep the mutation either way it goes
    Args:
        prompt (_type_): _description_
        digit1 (_type_): _description_
        digit2 (_type_): _description_
        images1 (_type_): _description_
        conf_scores1 (_type_): _description_
        images2 (_type_): _description_
        conf_scores2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Chose random member
    flag = random.uniform(0, 1)
    if flag < 0.5:
        member_to_mutate = digit1
        other_member = digit2
        ism1 = True
    else:
        member_to_mutate = digit2
        other_member = digit1
        ism1 = False
    print(f"Mutation on member: {1 if ism1 else 2}")

    # Mutate and predict
    DigitMutator(member_to_mutate).mutate(prompt)
    prediction, new_confidence = Predictor.predict_single(
        member_to_mutate, member_to_mutate.expected_label
    )

    # If confidence diff is too low and the new confidence is higher then...
    if (
        abs(new_confidence - member_to_mutate.confidence) <= 0.01
        and new_confidence >= member_to_mutate.confidence
    ):
        member_to_mutate.standing_steps += 1
    else:
        member_to_mutate.standing_steps = 0

    if ism1:
        images1.append(member_to_mutate.image_tensor)
        conf_scores1.append(new_confidence)
    else:
        images2.append(member_to_mutate.image_tensor)
        conf_scores2.append(new_confidence)

    return prediction, new_confidence, member_to_mutate, other_member


# Mutate both members, see if the confidence of both is lower
# At the end chose the one with the lower confidence
def dual_mutation(
    prompt, digit1, digit2, expected_label, images1, conf_scores1, images2, conf_scores2
):
    # Save data pre mutation
    pre_mutation_digit1 = digit1.clone()
    pre_mutation_digit2 = digit2.clone()

    # Mutate both digits
    DigitMutator(digit1).mutate(prompt)
    DigitMutator(digit2).mutate(prompt)

    # Predict both digits
    prediction1, confidence1 = Predictor.predict_single(digit1, expected_label)
    prediction2, confidence2 = Predictor.predict_single(digit2, expected_label)

    # Keep only best mutation
    # TODO: It might make sens to have 2 different gifs and graphs for each member in this case
    # Revert if confidence increased
    if confidence1 > pre_mutation_digit1.confidence:
        digit1 = pre_mutation_digit1
        confidence1 = digit1.confidence
        prediction1 = digit1.predicted_label

    if confidence2 > pre_mutation_digit2.confidence:
        digit2 = pre_mutation_digit2
        confidence2 = digit2.confidence
        prediction2 = digit2.predicted_label

    # Keep digit with lower confidence
    if confidence1 < confidence2:
        images1.append(digit1.image_tensor)
        conf_scores1.append(confidence1)
        return prediction1, confidence1, digit1, digit2
    else:
        images2.append(digit2.image_tensor)
        conf_scores2.append(confidence2)
        return prediction2, confidence2, digit2, digit1


def main(prompt, expected_label, max_steps=STEPS):
    # Force finding a valid initial latent
    while True:
        # Starting from a random latent noise vector
        latent = torch.randn(
            (1, get_pipeline().unet.config.in_channels, HEIGHT // 8, WIDTH // 8),
            device=DEVICE,
            dtype=DTYPE,
        )

        print(f"Mutation type: {MUTATION_TYPE}")
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
            print("Initial latent does not satisfy the label")
        else:
            break

    print(f"[000] " f"exp={digit1.expected_label} " f"conf={digit1.confidence:.3f}")
    digit2 = digit1.clone()  # second member of an Individual

    # Circular walk
    noise_x = torch.randn_like(latent)  # Direzione X (fissa)
    noise_y = torch.randn_like(latent)  # Direzione Y (fissa)

    images1 = []
    conf_scores1 = []
    images2 = []
    conf_scores2 = []
    euc_img_dists = []
    euc_img_dists.append(0)
    euc_dists = []
    euc_dists.append(0)
    latent_cos_sims = []
    latent_cos_sims.append(1)

    # Iterative mutation process
    for step in range(1, max_steps + 1):

        if MUTATION_TYPE == "dual":
            prediction, confidence, selected_digit, other_digit = dual_mutation(
                prompt,
                digit1,
                digit2,
                expected_label,
                images1,
                conf_scores1,
                images2,
                conf_scores2,
            )
        elif MUTATION_TYPE == "single_conf":
            result = single_mutation_conf(
                prompt, digit1, digit2, expected_label, images1, conf_scores1
            )
            if result is None:
                continue
            prediction, confidence, selected_digit, other_digit = result
        else:
            result = single_mutation_random(
                prompt, digit1, digit2, images1, conf_scores1, images2, conf_scores2
            )
            prediction, confidence, selected_digit, other_digit = result

        selected_digit.predicted_label = prediction
        selected_digit.confidence = confidence
        if selected_digit.expected_label == selected_digit.predicted_label:
            selected_digit.correctly_classified = True
        else:
            selected_digit.correctly_classified = False

        cos_sim = selected_digit.cosine_similarity(other_digit)
        latent_cos_sims.append(cos_sim)
        euc_dist = selected_digit.distance(other_digit)
        euc_dists.append(euc_dist)
        img_euc_dist = selected_digit.image_distance(other_digit)
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

    ind = Individual(
        digit1, digit2, prompt, None
    )  # Create final individual to see results
    ind.misstep = step
    ind.bad_prediction = selected_digit.predicted_label
    ind.members_distance = euc_dist
    ind.members_img_euc_dist = img_euc_dist
    ind.members_latent_cos_sim = cos_sim
    ind.export()
    base_path = f"{Folder.DST}"
    export_as_gif(
        f"{base_path}/individual_{Folder.run_id}_1.gif", images1, rubber_band=True
    )
    export_as_gif(
        f"{base_path}/individual_{Folder.run_id}_2.gif", images2, rubber_band=True
    )
    plot_confidence(conf_scores1, f"{base_path}/confidence_{Folder.run_id}_1.png")
    plot_confidence(conf_scores2, f"{base_path}/confidence_{Folder.run_id}_2.png")
    plot_distance(
        euc_dists,
        f"{base_path}/euclidean_distance_{Folder.run_id}.png",
        "Euclidean distance",
    )
    plot_distance(
        latent_cos_sims,
        f"{base_path}/cosine_similarity_{Folder.run_id}.png",
        "Cosine similarity",
    )
    plot_distance(
        euc_img_dists,
        f"{base_path}/euclidean_image_distance_{Folder.run_id}.png",
        "Euclidean image distance",
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
