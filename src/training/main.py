import os
from datetime import datetime
import warnings
import torch

from video.config import *
from video.loader import ImageDataLoader
from video.trainer import VideoModelTrainer

if __name__ == "__main__":
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(42)
    # torch.backends.cudnn.deterministic = True

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(EXPORT_DIR): os.makedirs(EXPORT_DIR)

    dt = datetime.now().strftime("%y%m%d_%H%M%S")
    PLOT_DIR = os.path.join(PLOT_DIR, dt)
    MODEL_DIR = os.path.join(MODEL_DIR, dt)
    EXPORT_DIR = os.path.join(EXPORT_DIR, dt)
    os.makedirs(PLOT_DIR)
    os.makedirs(MODEL_DIR)
    os.makedirs(EXPORT_DIR)

    # if DEVICE == "cuda":
    #     torch.backends.cudnn.benchmark = True

    print()
    dataset = ImageDataLoader(
        DATASET_DIR, 
        DATASET, 
        BATCH_SIZE,
        train_pct=TRAIN_PCT,
        val_pct=VAL_PCT,
        test_pct=TEST_PCT,
        num_examples=NUM_EXAMPLES
    )

    trainer = VideoModelTrainer(
        dataset,
        NETWORK_CHANNELS,
        COMPRESS_CHANNELS,
        BATCH_SIZE,
        LEARNING_RATE,
        DISTORTION_LAMBDA,
        DEVICE,
        SAVE_MODEL_FREQ,
        MODEL_DIR,
        PLOT_DIR,
        EXPORT_DIR,
        # model_dir="video_model_125.pth"
    )

    print()
    for epoch in range(20):
        trainer.train()
