import os
from datetime import datetime
import warnings
import torch

from video.config import *
from video.loader import ImageDataLoader
from video.trainer import VideoModelTrainer

if __name__ == "__main__":
    print()
    torch.manual_seed(42)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Ensure the directories exist
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(EXPORT_DIR): os.makedirs(EXPORT_DIR)

    # Create subdirectories for current run
    dt = datetime.now().strftime("%y%m%d_%H%M%S")
    PLOT_DIR = os.path.join(PLOT_DIR, dt)
    MODEL_DIR = os.path.join(MODEL_DIR, dt)
    EXPORT_DIR = os.path.join(EXPORT_DIR, dt)
    os.makedirs(PLOT_DIR)
    os.makedirs(MODEL_DIR)
    os.makedirs(EXPORT_DIR)

    # Load the dataset
    data_loader = ImageDataLoader(
        DATASET_DIR,
        DATASET,
        BATCH_SIZE,
        TRAIN_PCT,
        VAL_PCT,
        TEST_PCT,
        NUM_EXAMPLES,
        DEVICE
    )

    # Train the model
    trainer = VideoModelTrainer(data_loader)
    trainer.train()
