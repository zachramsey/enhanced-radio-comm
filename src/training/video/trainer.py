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
from .compiler import ExecutorchModel

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
    
    def __init__(
        self,
        data: ImageDataLoader,
        plot_dir: str,
        model_dir: str,
        export_dir: str
    ):
        self.train_dl = data.train_dl
        self.len_train_dl = len(data.train_dl)
        self.len_train_data = len(data.train_dl.dataset)

        self.val_dl = data.val_dl
        self.len_val_dl = len(data.val_dl)
        self.len_val_data = len(data.val_dl.dataset)

        self.example_dl = data.example_dl

        self.plot_dir = plot_dir
        self.model_dir = model_dir
        self.export_dir = export_dir

        # Initialize the training model
        self.model = VideoModel(NETWORK_CHANNELS, COMPRESS_CHANNELS).to(DEVICE)

        # Load existing model if provided
        self.model_path = CHECKPOINT_PATH
        self.last_batch = 0
        if self.model_path is not None:
            print(f">>> Loading model from {self.model_path}")
            self.model.load(self.model_path)
            self.last_batch = int(self.model_path.split("_")[-1].split(".")[0])

        # Initialize the encoder and decoder models
        self.encoder = VideoEncoder(NETWORK_CHANNELS, COMPRESS_CHANNELS, self.model)
        self.decoder = VideoDecoder(NETWORK_CHANNELS, COMPRESS_CHANNELS, self.model)

        # Initialize the Executorch model
        self.et_model = None

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
                # if i <= self.last_batch and self.last_batch > 0:
                #     print_inline_every(i, 50, self.last_batch, f">>> Skipping batch {i} of {self.last_batch}")
                #     continue

                # Training pass
                self.train_step(i, data)
                
                # Plot training losses
                self.plot_losses()

                # Flags for saving, exporting, and evaluating
                do_export = i % EXPORT_FREQ == 0
                do_eval = i % EVAL_FREQ == 0

                # Save a model checkpoint
                if (do_export or do_eval) and i > 0:
                    # Save the model
                    self.model_path = f"{self.model_dir}/VideoModel_{i*BATCH_SIZE}.pth"
                    self.model.save(self.model_path)

                    # Set the model to evaluation mode
                    self.model.hyper_bottleneck.update()
                    self.model.image_bottleneck.update()
                    self.model.eval()
                    self.encoder.eval()
                    self.decoder.eval()

                    # # Update the encoder model
                    # self.encoder.load(self.model_path)
                    # encoder_path = f"{MODEL_DIR}/EncoderModel_{i*BATCH_SIZE}.pth"
                    # self.encoder.save(encoder_path)

                    # # Update the decoder model
                    # self.decoder.load(self.model_path)
                    # decoder_path = f"{MODEL_DIR}/DecoderModel_{i*BATCH_SIZE}.pth"
                    # self.decoder.save(decoder_path)
                    
                    # Periodically evaluate the model
                    if do_eval:
                        # Reset statistics for evaluation
                        self.reset_stats()

                        # Evaluate the model
                        with torch.no_grad():
                            for j, (data, _) in enumerate(self.val_dl):
                                self.eval_step(j, data)

                        # Reset statistics for training
                        self.reset_stats()
                        print("\n")

                    # Periodically export the model
                    if do_export:
                        self.export(i*BATCH_SIZE)      # Export the model
                        self.simulate(i*BATCH_SIZE)    # Simulate transmission

                    # Set the model back to training mode
                    self.model.train()

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

    def export(self, step: int):
        if BACKEND is None: return
        elif BACKEND == "xnnpack":
            self.et_model = ExecutorchModel(
                step,
                self.encoder,
                self.decoder,
                NETWORK_CHANNELS,
                COMPRESS_CHANNELS,
                self.export_dir,
                CONTROL_DIR,
                REMOTE_DIR,
                quantize=QUANTIZE,
                device=DEVICE
            )
        # more backends can be added here

    def simulate(self, step: int):
        simulate_transmission(
            step,
            self.example_dl,
            self.model,
            self.encoder,
            self.decoder,
            self.et_model.encoder,
            self.et_model.decoder,
            self.plot_dir,
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
        plt.savefig(f"{self.plot_dir}/losses.png")
        plt.close()
