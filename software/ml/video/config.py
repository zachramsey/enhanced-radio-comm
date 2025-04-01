import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "./software/ml/datasets/"
PLOT_DIR = "./software/ml/plots/"
MODEL_DIR = "./software/ml/models/"
EXPORT_DIR = "./software/ml/exports/"

DATASET = "sun397"
TRAIN_PCT = 0.95
VAL_PCT = 0.025
TEST_PCT = 0.025
NUM_EXAMPLES = 5

BATCH_SIZE = 16
NETWORK_CHANNELS = 128
COMPRESS_CHANNELS = 192
LEARNING_RATE = 1e-4
DISTORTION_LAMBDA = 0.01

EVAL_FREQ = 100
SAVE_MODEL_FREQ = 500
