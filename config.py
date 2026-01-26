# %%writefile config.py
import torch

# Dev testing
TRYNEW = True
MUTATION_TYPE = "single_random"  # Or "single_conf" # Or 'dual'
DJ_DEBUG = True  # If True, creates detailed debug reports
SHORT_GEN = True

# Popolazione e generazioni
if SHORT_GEN:
    POPSIZE = 8  # Must be divisible by 4
    NGEN = 50  # Number of generations
    RESEED_INTERVAL = 3
else:
    POPSIZE = 20  # Must be divisible by 4
    NGEN = 100  # Number of generations
    RESEED_INTERVAL = 5
INITIALPOP = "random"  # Or "sequence"
STEPSIZE = 10
RESEEDUPPERBOUND = POPSIZE // 4  # Max number of reseed individuals

# Archive configuration
# Minimum distance between two individuals to be considered different enough
ARCHIVE_TYPE = "size"  # Or "dist"
ARCHIVE_THRESHOLD = 12
REPORT_NAME = "stats.csv"
STOP_CONDITION = "iter"  # Or 'time'
TARGET_SIZE = (3 * POPSIZE) // 4  # Ideal number of archived individuals

# Timer
RUNTIME = 3600

# Evaluator
K_SD = 0.1
K = 1

# Classifier
CLASSIFIER_WEIGHTS_PATH = "./mnist_classifier/weight/MNIST_conv_classifier.pth"
IMG_SIZE = 28

# Stable Diffusion
MODEL_ID_PATH = "runwayml/stable-diffusion-v1-5"
# LORA_PATH = "./SD_weights"
LORA_PATH = "/kaggle/input/mnist-lora-sd-weights"  # Path for Kaggle
LORA_WEIGHTS = "Mnist_Lora_sdv1.5-000005.safetensors"
if SHORT_GEN:
    DELTA = 0.02  # affects perturbation size for mutation
    STANDING_STEP_LIMIT = 2
else:
    DELTA = 0.01  # Old method value 0.025
    STANDING_STEP_LIMIT = 3
DISTANCE_METRIC = "latent_euclidean"  # Or 'image_euclidean' Or 'latent_cosine'
CONF_CHANGE = 1.0  # ATTENTION: this changes with the chosen metric # Or 0.5 Or 0.02
# Circular walk
NOISE_SCALE = 0.025
CIRC_STEPS = 100

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

PROMPTS = [
    "A photo of Z0ero Number0",
    "A photo of one1 Number1",
    "A photo of two2 Number2",
    "A photo of three3 Number3",
    "A photo of Four4 Number4",
    "A photo of Five5 Number5",
    "A photo of Six6 Number6",
    "A photo of Seven7 Number7",
    "A photo of Eight8 Number8",
    "A photo of Nine9 Number9",
]
