import torch

STEPS = 100

# Classifier
CLASSIFIER_WEIGHTS_PATH = './mnist_classifier/weight/MNIST_conv_classifier.pth'

# Stable Diffusion 
MODEL_ID_PATH = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./SD_weights"
LORA_WEIGHTS = "Mnist_Lora_sdv1.5-000005.safetensors"
#DELTA = 0.05   # works when calculating delta with distance between latent vectors
DELTA = 0.2     # need something bigger when calculating delta with confidence scores

# Torch settings
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    VARIANT = "fp16"
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    VARIANT = None

# Standard image dimensions for 1.5 Stable Diffusion
HEIGHT = 512
WIDTH = 512

# Dev testing
TRYNEW = False