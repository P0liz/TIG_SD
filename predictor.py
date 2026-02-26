# %%writefile predictor.py
import numpy as np
import torch

from mnist_classifier.model_mnist import MnistClassifier
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from config import DATASET, DEVICE, IMG_SIZE, CLASSIFIER_WEIGHTS_PATH


class Predictor:

    if DATASET == "mnist":
        # MNIST classifier
        classifier = MnistClassifier(img_size=IMG_SIZE).to(DEVICE)
        classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=DEVICE))
    elif DATASET == "imagenet":
        # VGG19 pretrained on ImageNet
        weights = VGG19_BN_Weights.DEFAULT
        classifier = vgg19_bn(weights=weights).to(DEVICE)
    else:
        raise ValueError("Unsupported dataset specified in config")

    classifier.eval()
    print("Loaded classifier")

    def predict_single(member, exp_label):
        with torch.no_grad():
            # unsqueeze to add batch dimension needed by classifier
            logits = Predictor.classifier(member.image_tensor.unsqueeze(0)).squeeze().cpu().numpy()

        # Margin between the 2 top logits as confidence score
        # label, confidence = simple_confidence_margin(logits)

        # margin between expected and top logits
        label, confidence = confidence_margin(logits, exp_label)

        return label, confidence

    def predict(members, exp_labels):
        # Creating batch tensor for all members (batch size, channels, height, width)
        batch_tensors = torch.stack([member.image_tensor for member in members]).to(DEVICE)

        with torch.no_grad():
            batch_logits = Predictor.classifier(batch_tensors).cpu().numpy()

        predictions = []
        confidences = []
        for logits, exp_label in zip(batch_logits, exp_labels):
            label, confidence = confidence_margin(logits, exp_label)
            # label, confidence = comparison_confidence_helper(logits, exp_label)
            predictions.append(label)
            confidences.append(confidence)

        return predictions, confidences


def comparison_confidence_helper(logits, exp_label):
    import config

    if config.TRYNEW:
        label, confidence = normalized_confidence_margin(logits, exp_label)
    else:
        label, confidence = confidence_margin(logits, exp_label)

    return label, confidence


def normalized_confidence_margin(logits, exp_label):
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
    return new_label, margin


def confidence_margin(logits, exp_label):
    # Convert numpy array to torch tensor
    logits_tensor = torch.from_numpy(logits).float()
    # print(f"logits_tensor: {logits_tensor}")

    expected_logit = logits_tensor[exp_label]
    # Select the two best indices
    best_index1, best_index2 = np.argsort(-logits_tensor)[:2]

    if best_index1 == exp_label:
        best_but_not_expected = best_index2
    else:
        best_but_not_expected = best_index1
    best_logit = logits_tensor[best_but_not_expected]
    # Calculate margin between expected and top normalized logits
    margin = expected_logit - best_logit
    new_label = np.argmax(logits)
    return new_label, margin


def simple_confidence_margin(logits):
    logits_tensor = torch.from_numpy(logits).float()
    softmax_probs = torch.softmax(logits_tensor, dim=0).numpy()  # Normalize logits
    print(f"softmax_probs: {softmax_probs}")
    # Get margin between top 2 normalized logits
    sorted_probs = np.sort(softmax_probs)[::-1]
    new_label = np.argmax(logits)
    return new_label, sorted_probs[0] - sorted_probs[1]
