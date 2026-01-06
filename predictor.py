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
    
    def predict_single(ref, dig):
        #img_detached = img.detach()
        new_logits = Predictor.classifier(dig.image).squeeze().detach().cpu().numpy()
        
        # 1) was just applying softmax to the logits and getting the higher one

        # 2) margin between top logits as confidence score
        confidence, label = Predictor.confidence_margin(new_logits, ref.expected_label)

        # 3) Using distance in latent space as confidence measure
        #confidence = torch.norm(dig.latent - ref.latent).item()
        #normalized_conf = torch.sigmoid(-torch.tensor(confidence)).item()
        #label = np.argmax(new_logits)
        return label, confidence

    def confidence_margin(logits, exp_label):
        expected_logit = logits[exp_label]
        # Select the two best indices [perturbed logit]
        best_indices = np.argsort(-logits)[:2]
        best_index1, best_index2 = best_indices
        if best_index1 == exp_label:
            best_but_not_expected = best_index2
        else:
            best_but_not_expected = best_index1
        # Get the best prediction
        new_logit = logits[best_but_not_expected]
        # Calculate margin between the expected and the best logits
        margin = expected_logit - new_logit

        new_label = np.argmax(logits)
        return margin, new_label