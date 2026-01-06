import torch
import random
import re

from folder import Folder
from individual import Individual
from predictor import Predictor
from digit_mutator import DigitMutator
from mnist_member import MnistMember
from mutation_manager import get_pipeline
from config import DEVICE, HEIGHT, WIDTH , DTYPE, TRYNEW

def main(prompt, expected_label, max_steps=100):

    # Starting from a random latent noise vector
    latent = torch.randn((1, get_pipeline().unet.config.in_channels, HEIGHT // 8, WIDTH // 8), 
                        device=DEVICE, 
                        dtype=DTYPE)
    if not TRYNEW: 
        # Scala il latent secondo lo scheduler
        latent = latent * get_pipeline().scheduler.init_noise_sigma

    digit = MnistMember(latent, expected_label)

    # Initial generation and validation
    DigitMutator(digit).generate(prompt)
    prediction, confidence = Predictor.predict_single(digit, digit)
    
    # Initial assignment
    digit.predicted_label = prediction
    digit.confidence = confidence
    if digit.expected_label == digit.predicted_label:
        digit.correctly_classified = True
    else:
        digit.correctly_classified = False

    if not digit.correctly_classified:
        print(prompt, " - ", expected_label)
        print(
            f"pred={digit.predicted_label} "
            f"exp={digit.expected_label}"
        )
        ind = Individual(digit, digit)
        ind.export()
        raise RuntimeError("Initial latent does not satisfy the label")
    reference = digit.clone()   # reference digit for distance calculations from original

    print(
        f"[Step 0] "
        f"exp={digit.expected_label} "
        f"conf={digit.confidence:.3f}"
    )

    # Iterative mutation process
    for step in range(1, max_steps + 1):
        DigitMutator(digit).mutate(prompt)
        prediction, confidence = Predictor.predict_single(reference, digit)
        
        digit.predicted_label = prediction
        digit.confidence = confidence
        if digit.expected_label == digit.predicted_label:
            digit.correctly_classified = True
        else:
            digit.correctly_classified = False

        print(
            f"[{step:03d}] "
            f"pred={digit.predicted_label} "
            f"dist={digit.confidence:.3f}"
        )

        if not digit.correctly_classified:
            print(digit.predicted_label, digit.expected_label)
            print(f" Label flipped at step {step}")
            break

    ind = Individual(reference, digit)  # Create final individual to see results
    ind.misstep = step
    ind.export()


if __name__ == "__main__":
    prompts =[ "A photo of Z0ero Number0","A photo of one1 Number1","A photo of two2 Number2 ","A photo of three3 Number3","A photo of Four4 Number4","A photo of Five5 Number5","A photo of Six6 Number6","A photo of Seven7 Number7 ","A photo of Eight8 Number8","A photo of Nine9 Number9"]
    for i in range(0,2):
        Folder.initialize() 
        randprompt = random.choice(prompts)
        expected_label = int(re.search(r"Number(\d+)", randprompt).group(1)) 
        main(prompt=randprompt, expected_label=expected_label)
        print("GAME OVER")
        TRYNEW = not TRYNEW

# source ./venv/bin/activate