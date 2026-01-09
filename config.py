import torch

# Timer
RUNTIME = 3600

# Evaluator
K_SD = 0.1
K = 1

IMG_SIZE = 28

STEPS = 100

# Classifier
CLASSIFIER_WEIGHTS_PATH = "./mnist_classifier/weight/MNIST_conv_classifier.pth"

# Stable Diffusion
MODEL_ID_PATH = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./SD_weights"
LORA_WEIGHTS = "Mnist_Lora_sdv1.5-000005.safetensors"
DELTA = 0.02  # works when calculating delta with distance between latent vectors
NOISE_SCALE = 0.025  # Circular walk

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
