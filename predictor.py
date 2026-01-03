import numpy as np
import torch
import torch.nn.functional as F

from mnist_classifier.model_mnist import MnistClassifier
from config import num_classes, DEVICE, IMG_SIZE, CLASSIFIER_WEIGHTS_PATH


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
        original_logit = Predictor.classifier(dig.image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit).item()
        
        # 1) not really a confidence score, just appling softmax to logits
        # but this does not account for cases when all logits are similar (model is unsure)
        #probs = F.softmax(torch.tensor(original_logit), dim=0)
        #confidence = probs[original_label].item()

        # 2) margin between top-2 logits as confidence score
        #confidence = Predictor.confidence_margin(torch.tensor(original_logit))

        # 3) Using distance in latent space as confidence measure
        confidence = torch.norm(dig.latent - ref.latent).item()
        return original_label, confidence

    def confidence_margin(logits):
        sorted_logits, _ = torch.sort(logits, descending=True)
        margin = sorted_logits[0] - sorted_logits[1]
        return margin.item()

    """ TODO: understand if needed
    @staticmethod
    def predict(img, label):
        
        # Predictions vector
        predictions = Predictor.model.predict(img)

        predictions1 = list()
        confidences = list()
        for i in range(len(predictions)):
            preds = predictions[i]
            explabel = label[i]
            prediction1, prediction2 = np.argsort(-preds)[:2]

            # Activation level corresponding to the expected class
            confidence_expclass = preds[explabel]

            if prediction1 != explabel:
                confidence_notclass = preds[prediction1]
            else:
                confidence_notclass = preds[prediction2]

            confidence = confidence_expclass - confidence_notclass
            predictions1.append(prediction1)
            confidences.append(confidence)

        return predictions1, confidences

    @staticmethod
    def predict_single(img, label):
        explabel = (np.expand_dims(label, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, num_classes)
        explabel = np.argmax(explabel.squeeze())

        # Predictions vector
        predictions = Predictor.model.predict(img)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != label:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        confidence = confidence_expclass - confidence_notclass

        return prediction1, confidence
    """
