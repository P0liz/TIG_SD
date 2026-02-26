# %%writefile config.py
import torch
from pathlib import Path

# Dev testing
TRYNEW = True
DJ_DEBUG = True  # If True, creates detailed debug reports
SHORT_GEN = True

# Dataset
DATASET = "imagenet"  # Or "mnist"

# Popolazione e generazioni
if SHORT_GEN:
    POPSIZE = 12  # Must be divisible by 4
    NGEN = 50  # Number of generations
else:
    POPSIZE = 20
    NGEN = 100
RESEED_INTERVAL = 5
INITIALPOP = "sequence"  # Or random"
STEPSIZE = 10
RESEEDUPPERBOUND = POPSIZE // 6  # Max number of reseed individuals

# Archive configuration
ARCHIVE_TYPE = "bucket"  # Or "size" # Or "dist"
REPORT_NAME = "stats.csv"
STOP_CONDITION = "iter"  # Or 'time'
TARGET_SIZE = 30  # Ideal number of archived individuals
MAX_BUCKET_SIZE = 3
BUCKET_CONFIG = "size"  # Or "dist"
# Minimum distance between two individuals to be considered different enough (the higher the less inds archived)
# ATTENTION: this changes with the chosen metric
DIST_THRESHOLD = 20  # img_euc: 8 # lat_cos: 0.5 # lat_euc: 20

# Timer
RUNTIME = 3600

# Evaluator
K_SD = 0.1
K = 1

# Classifier
if DATASET == "mnist":
    CLASSIFIER_WEIGHTS_PATH = "./mnist_classifier/weight/MNIST_conv_classifier.pth"
    IMG_SIZE = 28
elif DATASET == "imagenet":
    CLASSIFIER_WEIGHTS_PATH = None  # Pretrained VGG19 weights are used, so no path needed
    IMG_SIZE = None

# Stable Diffusion
MODEL_ID_PATH = "runwayml/stable-diffusion-v1-5"
# Standard image dimensions for 1.5 Stable Diffusion
HEIGHT = 512
WIDTH = 512
LORA_PATH = "./SD_weights"
# LORA_PATH = "/kaggle/input/mnist-lora-sd-weights"  # Path for Kaggle
if DATASET == "mnist":
    LORA_WEIGHTS = "Mnist_Lora_sdv1.5-000005.safetensors"
    NUM_INFERENCE_STEPS = 15
elif DATASET == "imagenet":
    LORA_WEIGHTS = "pizza_imgnet-000005.safetensors"
    NUM_INFERENCE_STEPS = 25
DELTA = 0.025  # affects perturbation size for mutation
STANDING_STEP_LIMIT = 2
DISTANCE_METRIC = "latent_euclidean"  # Or 'image_euclidean' Or 'latent_cosine'
# ATTENTION: this changes with the chosen metric
CONF_CHANGE = 1.0  # img_euc: 0.5 # lat_cos: 0.02 # lat_euc: 1.0

# Torch settings
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    VARIANT = "fp16"
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    VARIANT = None

if DATASET == "mnist":
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
elif DATASET == "imagenet":
    PROMPTS = ["A photo of 1pizza pizza_slice"]
    # PROMPTS = ["A photo of 1pizza"]
    IMAGENET_LABEL = 963  # OR change 850 for teddy class

# Diversity analysis
ANALYSIS_CONFIG = "single_run"  # Or "archives"
DIVERSITY_OUTPUT_FOLDER = "diversity_analysis_results"
Path(DIVERSITY_OUTPUT_FOLDER).mkdir(exist_ok=True)
FOCUS_NAME = "focus"
OTHERS_NAME = "others"
