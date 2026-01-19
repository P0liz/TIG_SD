import torch
import numpy as np

DJ_DEBUG = 1

# Popolazione e generazioni
POPSIZE = 12  # Must be divisible by 4
NGEN = 50
INITIALPOP = "random"  # TODO: implement different initialization methods
STEPSIZE = 10
RESEEDUPPERBOUND = 3  # Reseeding moderato

# Archive configuration
ARCHIVE_THRESHOLD = np.inf  # Disabilitato per ora
REPORT_NAME = "stats.csv"
STOP_CONDITION = "iter"  # Or 'time'
DISTANCE_METRIC = "latent_euclidean"  # Or 'image_euclidean' Or 'latent_cosine'

# Timer
RUNTIME = 3600

# Evaluator
K_SD = 0.1
K = 1

# Iterations
STEPS = 100

# Classifier
CLASSIFIER_WEIGHTS_PATH = "./mnist_classifier/weight/MNIST_conv_classifier.pth"
IMG_SIZE = 28

# Stable Diffusion
MODEL_ID_PATH = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./SD_weights"
LORA_WEIGHTS = "Mnist_Lora_sdv1.5-000005.safetensors"
DELTA = 0.04  # Old value 0.025
STANDING_STEP_LIMIT = 3
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
MUTATION_TYPE = "single_random"  # Or "single_conf" # Or 'dual'
