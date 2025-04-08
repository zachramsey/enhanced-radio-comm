import os
import warnings
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics.image.psnr import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
import numpy as np

from config import *
from data_loader import ImageDataLoader
from model import VideoModel
from encoder import VideoModelEncoder
from decoder import VideoModelDecoder
from entropy_models import simulate_impairments, simulate_errors
from utils import print_inline_every, tensor_to_image

# Theory: research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf
# Implementation: https://github.com/psyrocloud/MS-SSIM_L1_LOSS
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS as MixCrit

class VideoModelTrainer:
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

        self.model = VideoModel(ch_network, ch_compress, batch_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.mix_crit = MixCrit(alpha=0.85)

        if self.model_path is not None:
            print(f"Loading model from {self.model_path}")
            self.model.load(self.model_dir + self.model_path)

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

            self.optimizer.zero_grad()
            reconstruction, y_likelihoods, z_likelihoods = self.model(data)
            rate_loss, distortion_loss, loss = self.rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                self.model.save(self.model_dir + f"video_model_{i}.pth")
                self.model_path = f"video_model_{i}.pth"

            if i % EVAL_FREQ == 0 and i > 0:
                # self.model.save(model_dir + f"video_model_{i}.pth")
                self.plot_losses(total_losses, rate_losses, distortion_losses)
                self.simulate(i, self.data.example_data)
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
                total_psnr += PeakSignalNoiseRatio().update(data, reconstruction).compute().item()

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
              f"PSNR: {total_psnr / len_dl:.6f}\n\n")
        self.model.train()
        return avg_loss, avg_rate_loss, avg_distortion_loss
    
    def simulate(self, step, data_loader: DataLoader):
        encoder = VideoModelEncoder(self.ch_network, self.ch_compress, self.batch_size).to(self.device)
        decoder = VideoModelDecoder(self.ch_network, self.ch_compress, self.batch_size).to(self.device)
        encoder.load(self.model_dir + self.model_path)
        decoder.load(self.model_dir + self.model_path)

        with torch.no_grad():
            original_imgs_np = []
            reconstructed_imgs_np = []
            for data, _ in data_loader:
                data = data.to(self.device)

                z_strings = encoder.encode_hyper(data)
                # z_strings = [simulate_impairments(z_strings[0], interference_prob=1.0)]
                z_strings = [simulate_errors(z_strings[0], p=0.92, r=0.08)]
                decoder.decode_hyper(z_strings)
                
                y_strings = encoder.encode_image()
                # y_strings = [simulate_impairments(y_strings[0], interference_prob=1.0)]
                z_strings = [simulate_errors(z_strings[0], p=0.92, r=0.08)]
                reconstruction = decoder.decode_image(y_strings)

                # Visualize the original and reconstructed image
                original_imgs_np.append(tensor_to_image(data))
                reconstructed_imgs_np.append(tensor_to_image(reconstruction))

            num_examples = len(original_imgs_np)
            fig, axs = plt.subplots(2, num_examples, figsize=(20, 5))
            for i in range(num_examples):
                axs[0, i].imshow(original_imgs_np[i])
                axs[0, i].axis("off")
                axs[1, i].imshow(reconstructed_imgs_np[i])
                axs[1, i].axis("off")
            plt.savefig(self.plot_dir + f"simulation_{step*self.batch_size}.png")
            plt.close(fig)
            
    def rate_distortion_loss(self, reconstruction: torch.Tensor, latent_likelihoods: torch.Tensor, hyper_latent_likelihoods: torch.Tensor, original: torch.Tensor):
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width
        bits = (latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()) / -math.log(2)
        bpp_loss = bits / num_pixels
        # distortion_loss = self.mix_crit(reconstruction, original)
        distortion_loss = torch.nn.functional.mse_loss(reconstruction, original)
        combined_loss = self.distortion_lambda * 255 ** 2 * distortion_loss + bpp_loss
        return bpp_loss, distortion_loss, combined_loss

    def plot_losses(self, total_losses, rate_losses, distortion_losses):
        plt.plot(total_losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.savefig(self.plot_dir + "losses.png")
        plt.close()


if __name__ == "__main__":
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(42)
    # torch.backends.cudnn.deterministic = True

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    if not os.path.exists(EXPORT_DIR): os.makedirs(EXPORT_DIR)

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
