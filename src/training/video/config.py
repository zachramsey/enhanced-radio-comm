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
BATCH_SIZE = 6
NETWORK_CHANNELS = 128
COMPRESS_CHANNELS = 192
LEARNING_RATE = 1e-4
DISTORTION_LAMBDA = 0.002
EVAL_FREQ = 5

# Model checkpoint settings
load_model = False
model_path = "250428_141734/VideoModel_1050.pth"
MODEL_DIR = "src/training/models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, model_path) if load_model else None

# Model export settings
EXPORT_DIR = "src/exports"
CONTROL_DIR = "src/control"
REMOTE_DIR = "src/remote"
BACKEND = "xnnpack"
EXPORT_FREQ = 10
