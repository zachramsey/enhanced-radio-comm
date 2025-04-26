import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "src/training/datasets"
PLOT_DIR = "src/training/plots"
MODEL_DIR = "src/models"
EXPORT_DIR = "src/exports"

DATASET = "sun397"
TRAIN_PCT = 0.99
VAL_PCT = 0.01
TEST_PCT = 0.0
NUM_EXAMPLES = 5

BATCH_SIZE = 4
NETWORK_CHANNELS = 128
COMPRESS_CHANNELS = 192
LEARNING_RATE = 1e-4
DISTORTION_LAMBDA = 0.001

EVAL_FREQ = 100
SAVE_MODEL_FREQ = 100
