import numpy as np
import torch

from mnist_classifier.model_mnist import MnistClassifier
from config import DEVICE, IMG_SIZE, CLASSIFIER_WEIGHTS_PATH


class Predictor:

    # Load the pre-trained model.
    classifier = MnistClassifier(img_size=IMG_SIZE).to(DEVICE)
    # Load pretrained model
    classifier.load_state_dict(
        torch.load(
            CLASSIFIER_WEIGHTS_PATH,
            map_location=DEVICE,
        )
    )
    classifier.eval()
    print("Loaded classifier")

    def predict_single(dig, exp_label):
        new_logits = (
            Predictor.classifier(dig.image_tensor).squeeze().detach().cpu().numpy()
        )

        # Margin between top logits as confidence score
        # label = np.argmax(new_logits)
        # confidence = simple_confidence_margin(new_logits)
        confidence, label = confidence_margin(new_logits, exp_label)

        return label, confidence


def confidence_margin(logits, exp_label):
    # Convert numpy array to torch tensor
    logits_tensor = torch.from_numpy(logits).float()
    # Apply softmax to normalize (since I do not want row logits)
    softmax_probs = torch.softmax(logits_tensor, dim=0).numpy()
    # print(f"softmax_probs: {softmax_probs}")

    expected_prob = softmax_probs[exp_label]
    # Select the two best indices
    best_indices = np.argsort(-softmax_probs)[:2]
    best_index1, best_index2 = best_indices

    if best_index1 == exp_label:
        best_but_not_expected = best_index2
    else:
        best_but_not_expected = best_index1
    best_prob = softmax_probs[best_but_not_expected]
    # Calculate margin between expected and top normalized logits
    margin = expected_prob - best_prob
    new_label = np.argmax(logits)
    return margin, new_label


def simple_confidence_margin(logits):
    logits_tensor = torch.from_numpy(logits).float()
    softmax_probs = torch.softmax(logits_tensor, dim=0).numpy()  # Normalize logits
    print(f"softmax_probs: {softmax_probs}")
    # Get margin between top 2 normalized logits
    sorted_probs = np.sort(softmax_probs)[::-1]
    return sorted_probs[0] - sorted_probs[1]
