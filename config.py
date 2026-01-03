import torch

DJ_DEBUG = 1

# GA Setup
POPSIZE = 100

STOP_CONDITION = "iter"
#STOP_CONDITION = "time"

NGEN = 100
RUNTIME = 3600
STEPSIZE = 10
# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10

K_SD = 0.1

# K-nearest
K = 1

# Archive configuration
ARCHIVE_THRESHOLD = 4.0

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28 * 28
num_classes = 10


INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False


RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'


# Classifier
CLASSIFIER_WEIGHTS_PATH = './mnist_classifier/weight/MNIST_conv_classifier.pth'

# Stable Diffusion 
MODEL_ID_PATH = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./SD_weights"
LORA_WEIGHTS = "Mnist_Lora_sdv1.5-000005.safetensors"
DELTA = 0.05

# Torch settings
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == "cuda:0" else torch.float32
VARIANT = "fp16" if DTYPE == torch.float16 else None

HEIGHT = 224
WIDTH = 224