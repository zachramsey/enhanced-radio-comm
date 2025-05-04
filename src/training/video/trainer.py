import math
import time
import torch
from torcheval.metrics.image.psnr import PeakSignalNoiseRatio
import matplotlib.pyplot as plt

from .config import *
from .utils import print_inline_every
from .simulate import simulate_transmission

from .loader import ImageDataLoader
from .model import VideoModel
from .encoder import VideoEncoder
from .decoder import VideoDecoder
from .compiler import XNNPackModel

class VideoModelTrainer:
    '''
    Train the video model on a dataset of images.

    Parameters
    ----------
    data : ImageDataLoader
        The dataset to train on.
    model_path : str[optional]
        The path to a pre-trained model to load.

    Methods
    -------
    train() -> tuple
        Train the model on the dataset.
    evaluate() -> tuple
        Evaluate the model on the validation set.
    test_model(step, loader)
        Test the model on a given DataLoader.
    plot_losses(total_losses, rate_losses, distortion_losses)
        Plot the training losses.
    '''
    
    def __init__(self, data: ImageDataLoader):
        self.train_dl = data.train_dl
        self.len_train_dl = len(data.train_dl)
        self.len_train_data = len(data.train_dl.dataset)

        self.val_dl = data.val_dl
        self.len_val_dl = len(data.val_dl)
        self.len_val_data = len(data.val_dl.dataset)

        self.example_dl = data.example_dl

        # Initialize the training model
        self.model = VideoModel(NETWORK_CHANNELS, COMPRESS_CHANNELS).to(DEVICE)

        # Initialize inference models
        self.encoder = VideoEncoder(NETWORK_CHANNELS, COMPRESS_CHANNELS, DEVICE)
        self.decoder = VideoDecoder(NETWORK_CHANNELS, COMPRESS_CHANNELS, DEVICE)

        # Initialize ExecuTorch models
        self.exec_encoder = None
        self.exec_decoder = None

        # Load existing model if provided
        self.model_path = CHECKPOINT_PATH
        self.last_batch = 0
        if self.model_path is not None:
            print(f">>> Loading model from {self.model_path}")
            self.model.load(self.model_path)
            self.last_batch = int(self.model_path.split("_")[-1].split(".")[0])

        # Set benchmark mode for CUDA
        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = BENCHMARK

        # Dynamically choose compiler backend if available
        if torch.onnx.is_onnxrt_backend_supported() and USE_ONNXRT:
            print(">>> Using ONNXRT backend for compilation")
            self.model = torch.compile(self.model, backend="onnxrt")
        elif torch.cuda.is_available() and USE_CUDA:
            if torch.cuda.get_device_capability()[0] > 6:
                print(">>> Using TorchInductor backend for compilation")
                self.model = torch.compile(self.model, backend="inductor", mode="reduce-overhead")
            else:
                print(">>> Using CudaGraphs backend for compilation")
                self.model = torch.compile(self.model, backend="cudagraphs")

        # Model Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Training losses
        self.rate_losses: list[float] = []
        self.distortion_losses: list[float] = []
        self.losses: list[float] = []

        # Model statistics
        self.psnr = PeakSignalNoiseRatio(device=DEVICE)
        self.rate_loss = 0.0
        self.distortion_loss = 0.0
        self.loss = 0.0
        self.time = 0.0


    def train(self):
        for e in range(MAX_EPOCHS):
            print(f"\n========================\n"
                    f"Epoch {e+1}/{MAX_EPOCHS}\n"
                    f"========================\n")
            
            # Train the model
            for i, (data, _) in enumerate(self.train_dl):
                # Skip to the last batch if resuming training
                if i <= self.last_batch and self.last_batch > 0:
                    print_inline_every(i, 50, self.last_batch, f">>> Skipping batch {i} of {self.last_batch}")
                    continue

                # Training pass
                self.train_step(i, data)

                # Flags for saving, exporting, and evaluating
                do_export = i % EXPORT_FREQ == 0
                do_eval = i % EVAL_FREQ == 0

                # Save a model checkpoint
                if (do_export or do_eval) and i > 0:
                    # Save the model
                    self.model_path = f"{MODEL_DIR}/VideoModel_{i}.pth"
                    self.model.save(self.model_path)

                    # Update bottlenecks and set to eval mode
                    self.model.hyper_bottleneck.update()
                    self.model.image_bottleneck.update()
                    self.model.eval()

                    # Update the encoder model
                    self.encoder.load(self.model_path)
                    encoder_path = f"{MODEL_DIR}/EncoderModel_{i*BATCH_SIZE}.pth"
                    self.encoder.save(encoder_path)

                    # Update the decoder model
                    self.decoder.load(self.model_path)
                    decoder_path = f"{MODEL_DIR}/DecoderModel_{i*BATCH_SIZE}.pth"
                    self.decoder.save(decoder_path)

                    # Periodically export the model
                    if do_export:
                        self.export()
                    
                    # Periodically evaluate the model
                    if do_eval:
                        self.reset_stats()  # Reset statistics for evaluation
                        self.plot_losses()  # Plot training losses
                        self.simulate(i)    # Simulate transmission

                        # Evaluate the model
                        for i, (data, _) in enumerate(self.val_dl):
                            with torch.no_grad():
                                self.eval_step(i, data)

                        self.reset_stats()  # Reset statistics for training
                        print("\n")

                    self.model.train()  # Set the model back to training mode

    def train_step(self, step: int, data: torch.Tensor):
        start = time.time()     # Start time for this iteration
        data = data.to(DEVICE)  # Move data to the device

        # Delayed bottleneck update
        if step % 5 == 0:
            self.model.hyper_bottleneck.update()
            self.model.image_bottleneck.update()

        # Training pass
        self.optimizer.zero_grad()
        noise = lambda x: x + torch.empty_like(x).uniform_(-10, 10)
        reconstruction, y_likelihoods, z_likelihoods = self.model(data, noise_func=noise)
        rate_loss, distortion_loss, loss = self.rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data)
        loss.backward()
        self.optimizer.step()

        # Update training statistics
        self.rate_losses.append(rate_loss.item())
        self.distortion_losses.append(distortion_loss.item())
        self.losses.append(loss.item())

        self.psnr = self.psnr.update(data, reconstruction)
        self.rate_loss += rate_loss.item()
        self.distortion_loss += distortion_loss.item()
        self.loss += loss.item()
        self.time += (time.time() - start)

        # Print training statistics
        train_iter = ((step-1) % EVAL_FREQ) + 1
        print_inline_every(step, 1, self.len_train_dl, (
            f"Train Step [{step*BATCH_SIZE:>7}/{self.len_train_data:<7}] ({(self.time / train_iter):.2f} s/it) | "
            f"Compress: {(self.rate_loss / train_iter):.6f} | "
            f"Distort: {(self.distortion_loss / train_iter):.6f} | "
            f"Total: {(self.loss / train_iter):.6f} | "
            f"PSNR: {self.psnr.compute().item():.6f}"
        ))

    def eval_step(self, step: int, data: torch.Tensor):
        start = time.time()     # Start time for this iteration
        data = data.to(DEVICE)  # Move data to the device

        # Evaluation pass
        reconstruction, y_likelihoods, z_likelihoods = self.model(data)
        rate_loss, distortion_loss, loss = self.rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data)
        
        # Update evaluation statistics
        self.rate_loss += rate_loss.item()
        self.distortion_loss += distortion_loss.item()
        self.loss += loss.item()
        self.psnr = self.psnr.update(data, reconstruction)
        self.time += time.time() - start

        # Print evaluation statistics
        print_inline_every(step, 1, self.len_val_dl, (
            f" Eval Step [{step*BATCH_SIZE:>7}/{self.len_val_data:<7}] ({(self.time / (step+1)):.2f} s/it) | "
            f"Compress: {(self.rate_loss / (step+1)):.6f} | "
            f"Distort: {(self.distortion_loss / (step+1)):.6f} | "
            f"Total: {(self.loss / (step+1)):.6f} | "
            f"PSNR: {self.psnr.compute().item():.6f}"
        ))

    def rate_distortion_loss(self, reconstruction: torch.Tensor, latent_likelihoods: torch.Tensor, hyper_latent_likelihoods: torch.Tensor, original: torch.Tensor):
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width
        bits = (latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()) / -math.log(2)
        bpp_loss = bits / num_pixels
        distortion_loss = torch.nn.functional.mse_loss(reconstruction, original)
        combined_loss = DISTORTION_LAMBDA * 255**2 * distortion_loss + bpp_loss
        return bpp_loss, distortion_loss, combined_loss

    def export(self):
        if BACKEND is None: return
        elif BACKEND == "xnnpack":
            xnnpack_model = XNNPackModel(
                self.encoder,
                self.decoder,
                NETWORK_CHANNELS,
                COMPRESS_CHANNELS,
                EXPORT_DIR,
                CONTROL_DIR,
                REMOTE_DIR
            )
            xnnpack_model.load_methods()
            self.exec_encoder = xnnpack_model.encoder
            self.exec_decoder = xnnpack_model.decoder
        # more backends can be added here

    def simulate(self, step:int):
        simulate_transmission(
            step,
            self.example_dl,
            self.model,
            self.encoder,
            self.decoder,
            self.exec_encoder,
            self.exec_decoder,
            PLOT_DIR,
            device=DEVICE,
        )

    def reset_stats(self):
        self.psnr.reset()
        self.rate_loss = 0
        self.distortion_loss = 0
        self.loss = 0
        self.time = 0

    def plot_losses(self):
        plt.plot(self.losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.savefig(f"{PLOT_DIR}/losses.png")
        plt.close()
