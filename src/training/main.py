import os
import sys
from datetime import datetime
import warnings
import torch

from video.config import *
from video.loader import ImageDataLoader
from video.trainer import VideoModelTrainer

from video.encoder import VideoEncoder
from video.decoder import VideoDecoder
from video.compiler import ExecutorchModel

if __name__ == "__main__":
    print()
    n = len(sys.argv)
    
    # Train the model
    if n == 1:
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
        plot_dir = os.path.join(PLOT_DIR, dt)
        model_dir = os.path.join(MODEL_DIR, dt)
        export_dir = os.path.join(EXPORT_DIR, dt)
        os.makedirs(plot_dir)
        os.makedirs(model_dir)
        os.makedirs(export_dir)

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
        trainer = VideoModelTrainer(
            data_loader,
            plot_dir,
            model_dir,
            export_dir
        )
        trainer.train()

    # Load an existing model and create the XNNPack executables
    else:
        # Validate arguments
        if n != 7:
            print(f"Usage: python main.py <model_path> <c_network> <c_compress> <control_dir> <remote_dir> <quantize>")
            sys.exit(1)

        model_path = sys.argv[1]
        c_network = int(sys.argv[2])
        c_compress = int(sys.argv[3])
        control_dir = sys.argv[4]
        remote_dir = sys.argv[5]
        quantize = sys.argv[6].lower() == "true" or sys.argv[6].lower() == "t" or sys.argv[6].lower() == "yes" or sys.argv[6].lower() == "y" or sys.argv[6] == "1"

        # Create the encoder and decoder models
        encoder = VideoEncoder(c_network, c_compress)
        decoder = VideoDecoder(c_network, c_compress)

        # Load the model weights
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            # Make sure encoder and decoder are properly loaded and not None
            encoder.load(model_path)
            decoder.load(model_path)
            print(f"Model loaded successfully")
        else:
            print(f"Model path {model_path} does not exist")
            sys.exit(1)

        # Validate the control and remote directories
        if not os.path.exists(control_dir):
            print(f"Control directory {control_dir} does not exist, creating it")
            os.makedirs(control_dir)
        if not os.path.exists(remote_dir):
            print(f"Remote directory {remote_dir} does not exist, creating it")
            os.makedirs(remote_dir)

        # Create the XNNPack model
        print("Creating XNNPackModel")
        xnnpack_model = ExecutorchModel(
            encoder=encoder,
            decoder=decoder,
            c_network=c_network,
            c_compress=c_compress,
            export_dir=None,
            control_dir=control_dir,
            remote_dir=remote_dir,
            quantize=quantize
        )
        print("XNNPackModel successfully created")

        enc_name = f"img_enc_xnnpack_{'q8' if quantize else 'fp32'}.pte"
        print(f"Encoder model saved at: {os.path.join(remote_dir, enc_name)}")

        dec_name = f"img_dec_xnnpack_{'q8' if quantize else 'fp32'}.pte"
        print(f"Decoder model saved at: {os.path.join(control_dir, dec_name)}")
