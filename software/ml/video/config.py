import torch

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 1
NETWORK_CHANNELS = 128
COMPRESS_CHANNELS = 192
DISTORTION_LAMBDA = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "./data"
DATASET = "places365"
SAVE_MODEL_PATH = "./video_model.pth"