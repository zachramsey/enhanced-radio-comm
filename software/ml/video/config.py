import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "places365"
DATASET_DIR = "./software/ml/datasets/"
PLOT_DIR = "./software/ml/plots/"
MODEL_DIR = "./software/ml/models/"
EXPORT_DIR = "./software/ml/exports/"

BATCH_SIZE = 16
NETWORK_CHANNELS = 128
COMPRESS_CHANNELS = 192
LEARNING_RATE = 1e-4
DISTORTION_LAMBDA = 0.01

EVAL_FREQ = 100
TRAIN_PRINT_FREQ = 1
SAVE_MODEL_FREQ = 500
