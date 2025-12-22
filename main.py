import torch
import numpy as np

from individual import Individual
from predictor import Predictor
from digit_mutator import DigitMutator
from mnist_member import MnistMember

def main(prompt, expected_label, max_steps=50):

    # Starting from a random latent noise vector
    latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    digit = MnistMember(latent, expected_label)

    # Initial generation and validation
    img = DigitMutator(digit).generate(prompt)
    prediction, confidence = Predictor.predict_single(img.image, img.image)
    
    # Initial assignment
    digit.predicted_label = prediction
    digit.confidence = confidence
    if digit.expected_label == digit.predicted_label:
        digit.correctly_classified = True
    else:
        digit.correctly_classified = False

    if not digit.correctly_classified:
        print(digit.predicted_label, digit.expected_label)
        ind = Individual(digit, digit)
        ind.export()
        raise RuntimeError("Initial latent does not satisfy the label")
    reference = digit   # reference digit for distance calculations from original

    print(f" Step 0 | conf={digit.confidence:.3f}")

    # Iterative mutation process
    for step in range(1, max_steps + 1):
        DigitMutator(digit).mutate(prompt)
        prediction, confidence = Predictor.predict_single(reference.image, digit.image)
        
        digit.predicted_label = prediction
        digit.confidence = confidence
        if digit.expected_label == digit.predicted_label:
            digit.correctly_classified = True
        else:
            digit.correctly_classified = False

        print(
            f"[{step:03d}] "
            f"pred={digit.predicted_label} "
            f"conf={digit.confidence:.3f}"
        )

        if not digit.correctly_classified:
            print(f" Label flipped at step {step}")
            break

    ind = Individual(reference, digit)  # Create final individual to see results
    #ind.members_distance = reference.image_distance(digit)
    #ind.members_distance = cosine_similarity(reference.latent, digit.latent)
    ind.export()

# TODO: Allow also different digits to be generated
if __name__ == "__main__":
    main(prompt="A photo of Five5 Number5", expected_label=5)
    print("GAME OVER")

# source ./venv/bin/activate