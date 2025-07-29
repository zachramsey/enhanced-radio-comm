import torch
import os

# Training settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK = False
USE_ONNXRT = False
USE_CUDA = False

# Data Parameters
PLOT_DIR = "src/training/plots"
DATASET_DIR = "src/training/datasets"
DATASET = "sun397"
TRAIN_PCT = 0.99
VAL_PCT = 0.01
TEST_PCT = 0.0
NUM_EXAMPLES = 5

# Training parameters
MAX_EPOCHS = 1000
BATCH_SIZE = 24
NETWORK_CHANNELS = 32
COMPRESS_CHANNELS = 64
LEARNING_RATE = 1e-4
DISTORTION_LAMBDA = 0.002
EVAL_FREQ = 100

# Model checkpoint settings
load_model = False
model_path = "picks/VideoModel-128_192-psnr_29_392147.pth"
MODEL_DIR = "src/training/models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, model_path) if load_model else None

# Model export settings
EXPORT_DIR = "src/exports"
CONTROL_DIR = "pte/control"
REMOTE_DIR = "pte/remote"
BACKEND = "xnnpack"
QUANTIZE = False
EXPORT_FREQ = 500
