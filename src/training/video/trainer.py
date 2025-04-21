import math
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.image.psnr import PeakSignalNoiseRatio
from torcheval.metrics.image.ssim import StructuralSimilarity
import matplotlib.pyplot as plt
import numpy as np

from .config import *
from .utils import print_inline_every
from .simulate import simulate_transmission

from .loader import ImageDataLoader
from .model import VideoModel
from .encoder import VideoEncoder
from .decoder import VideoDecoder
from .compiler import XNNPackModel

# Theory: research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf
# Implementation: https://github.com/psyrocloud/MS-SSIM_L1_LOSS
from .ms_ssim_l1_loss import MS_SSIM_L1_Loss

class VideoModelTrainer:
    '''
    Train the video model on a dataset of images.

    Parameters
    ----------
    dataset : ImageDataLoader
        The dataset to train on.
    ch_network : int
        The number of channels in the network.
    ch_compress : int
        The number of channels in the compressed representation.
    batch_size : int
        The batch size to use for training.
    learning_rate : float
        The learning rate to use for training.
    distortion_lambda : float
        The lambda value to use for the distortion loss.
    device : str
        The device to use for training (e.g. "cuda" or "cpu").
    save_freq : int
        The number of steps between saving the model.
    model_dir : str
        The directory to save models in.
    plot_dir : str
        The directory to save plots in.
    export_dir : str
        The directory to save exported models in.
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
        dataset: ImageDataLoader,
        ch_network: int,
        ch_compress: int,
        batch_size: int,
        learning_rate: float,
        distortion_lambda: float,
        device: str,
        save_freq: int,
        model_dir: str,
        plot_dir: str,
        export_dir: str,
        model_path: str = None
    ):
        self.data = dataset
        self.ch_network = ch_network
        self.ch_compress = ch_compress
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.distortion_lambda = distortion_lambda
        self.device = device
        self.save_freq = save_freq
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.export_dir = export_dir
        self.model_path = model_path

        self.model = VideoModel(ch_network, ch_compress).to(device)

        # Dynamically choose compiler backend if available
        if torch.onnx.is_onnxrt_backend_supported():
            self.model = torch.compile(self.model, backend="onnxrt")
        elif torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] > 6:
                self.model = torch.compile(self.model, backend="inductor", mode="reduce-overhead")
            else:
                self.model = torch.compile(self.model, backend="cudagraphs")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.mix_crit = MS_SSIM_L1_Loss(alpha=0.85)

        self.scaler = torch.amp.GradScaler(enabled=DEVICE=="cuda")

        if self.model_path is not None:
            print(f"Loading model from {self.model_path}")
            self.model.load(self.model_path)


    def train(self):
        self.model.train()
        rate_losses = []
        distortion_losses = []
        total_losses = []
        eval_loss = np.inf

        len_dl = len(self.data.train_dl)
        len_data = len(self.data.train_dl.dataset)

        for i, (data, _) in enumerate(self.data.train_dl):
            data = data.to(self.device)

            self.model.hyper_bottleneck.update()
            self.model.image_bottleneck.update()

            self.optimizer.zero_grad()
            reconstruction, y_likelihoods, z_likelihoods = self.model(data)
            rate_loss, distortion_loss, loss = self.rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            rate_losses.append(rate_loss.item())
            distortion_losses.append(distortion_loss.item())
            total_losses.append(loss.item())

            msg = (f"Training Step [{i*self.batch_size}/{len_data}] | "
                   f"Rate Loss: {rate_loss.item():.6f} | "
                   f"Distortion Loss: {distortion_loss.item():.6f} | "
                   f"Total Loss: {loss.item():.6f}")
            print_inline_every(i, 1, len_dl, msg)

            if i % self.save_freq == 0:
                self.model_path = f"{self.model_dir}/VideoModel_{i}.pth"
                self.model.save(self.model_path)

            if i % EVAL_FREQ == 0 and i > 0:
                # self.model.save(model_dir + f"video_model_{i}.pth")
                self.plot_losses(total_losses, rate_losses, distortion_losses)
                self.test_model(i, self.data.example_data)
                avg_loss, avg_rate_loss, avg_distortion_loss = self.evaluate()
                if abs(avg_loss - eval_loss) < 0.0001:
                    print("Early stopping")
                    break
                eval_loss = avg_loss

        return total_losses, rate_losses, distortion_losses


    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_distortion_loss = 0
        total_rate_loss = 0
        total_psnr = 0
        # total_ssim = 0

        len_dl = len(self.data.val_dl)
        len_data = len(self.data.val_dl.dataset)

        with torch.no_grad():
            for i, (data, _) in enumerate(self.data.val_dl):
                data = data.to(self.device)

                reconstruction, y_likelihoods, z_likelihoods = self.model(data)

                rate_loss, distortion_loss, loss = self.rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data)
                total_loss += loss.item()
                total_distortion_loss += distortion_loss.item()
                total_rate_loss += rate_loss.item()

                _reconstruction = reconstruction.permute(0, 3, 1, 2) / 255.0
                _data = data.permute(0, 3, 1, 2) / 255.0
                total_psnr += PeakSignalNoiseRatio().update(_data, _reconstruction).compute().item()
                # total_ssim += StructuralSimilarity().update(_data, _reconstruction).compute().item()

                msg = (f"Eval Step [{i*self.batch_size}/{len_data}] | "
                       f"Rate Loss: {rate_loss.item():.6f} | "
                       f"Distortion Loss: {distortion_loss.item():.6f} | "
                       f"Total Loss: {loss.item():.6f}")
                print_inline_every(i, 1, len_dl, msg)

        avg_loss = total_loss / len_dl
        avg_distortion_loss = total_distortion_loss / len_dl
        avg_rate_loss = total_rate_loss / len_dl
        print(f"Eval Avgs | "
              f"Compression: {avg_rate_loss:.6f} | "
              f"Distortion: {avg_distortion_loss:.6f} | "
              f"Total Loss: {avg_loss:.6f} | "
            #   f"SSIM: {total_ssim / len_dl:.6f} | "
              f"PSNR: {total_psnr / len_dl:.6f}\n\n")
        self.model.train()
        return avg_loss, avg_rate_loss, avg_distortion_loss
    

    def test_model(self, step:int, loader:DataLoader):
        # Set up and save the encoder model
        encoder = VideoEncoder(self.ch_network, self.ch_compress).to(self.device)
        encoder.load(self.model_path)
        encoder.to('cpu')
        encoder_path = f"{self.model_dir}/EncoderModel_{step*self.batch_size}.pth"
        encoder.save(encoder_path)

        # Set up and save the decoder model
        decoder = VideoDecoder(self.ch_network, self.ch_compress).to(self.device)
        decoder.load(self.model_path)
        decoder.to('cpu')
        decoder_path = f"{self.model_dir}/DecoderModel_{step*self.batch_size}.pth"
        decoder.save(decoder_path)

        # Simulate transmission with the original models
        with torch.no_grad():
            simulate_transmission(loader, encoder, decoder, self.plot_dir, type="errors")
            

    def rate_distortion_loss(self, reconstruction: torch.Tensor, latent_likelihoods: torch.Tensor, hyper_latent_likelihoods: torch.Tensor, original: torch.Tensor):
        reconstruction = reconstruction.permute(0, 3, 1, 2) / 255.0
        original = original.permute(0, 3, 1, 2) / 255.0
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width
        bits = (latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()) / -math.log(2)
        bpp_loss = bits / num_pixels
        # distortion_loss = self.mix_crit(reconstruction, original)
        distortion_loss = torch.nn.functional.mse_loss(reconstruction, original)
        combined_loss = self.distortion_lambda * 255**2 * distortion_loss + bpp_loss
        return bpp_loss, distortion_loss, combined_loss


    def plot_losses(self, total_losses, rate_losses, distortion_losses):
        plt.plot(total_losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.savefig(f"{self.plot_dir}/losses.png")
        plt.close()
