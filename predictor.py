# %%writefile predictor.py
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

    def predict_single(member, exp_label):
        with torch.no_grad():
            logits = Predictor.classifier(member.image_tensor).squeeze().cpu().numpy()

        # Margin between the 2 top logits as confidence score
        # confidence, label = simple_confidence_margin(logits)

        # margin between expected and top logits
        confidence, label = confidence_margin(logits, exp_label)

        return label, confidence

    def predict(members, exp_labels):
        batch_tensors = torch.stack(
            [member.image_tensor.squeeze(0) for member in members]
        ).to(DEVICE)

        with torch.no_grad():
            batch_logits = Predictor.classifier(batch_tensors).cpu().numpy()

        predictions = []
        confidences = []
        for logits, exp_label in zip(batch_logits, exp_labels):
            confidence, label = confidence_margin(logits, exp_label)
            predictions.append(label)
            confidences.append(confidence)

        return predictions, confidences


def confidence_margin(logits, exp_label):
    # Convert numpy array to torch tensor
    logits_tensor = torch.from_numpy(logits).float()
    # Apply softmax to normalize (since I do not want row logits)
    softmax_probs = torch.softmax(logits_tensor, dim=0).numpy()
    # print(f"softmax_probs: {softmax_probs}")

    expected_prob = softmax_probs[exp_label]
    # Select the two best indices
    best_index1, best_index2 = np.argsort(-softmax_probs)[:2]

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
    new_label = np.argmax(logits)
    return sorted_probs[0] - sorted_probs[1], new_label
