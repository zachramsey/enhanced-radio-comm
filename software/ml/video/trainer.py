import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from config import *
from data_loader import ImageDataLoader
from model import VideoModel
from encoder import VideoModelEncoder
from decoder import VideoModelDecoder
from utils import print_inline_every, tensor_to_image

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
        load_model: str = None
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
        self.load_model = load_model

        self.model = VideoModel(ch_network, ch_compress, batch_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def rate_distortion_loss(self, reconstruction, latent_likelihoods, hyper_latent_likelihoods, original):
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width
        bits = (latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()) / -math.log(2)
        bpp_loss = bits / num_pixels
        distortion_loss = F.mse_loss(reconstruction, original)
        combined_loss = self.distortion_lambda * 255 ** 2 * distortion_loss + bpp_loss
        return bpp_loss, distortion_loss, combined_loss

    def train(self):
        self.model.train()
        rate_losses = []
        distortion_losses = []
        total_losses = []
        eval_loss = np.inf

        if self.load_model is not None:
            print(f"Loading model from {self.load_model}")
            self.model.load(self.model_dir + self.load_model)

        for i, (data, _) in enumerate(self.data.train_dl):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            reconstruction, y_likelihoods, z_likelihoods = self.model(data)
            rate_loss, distortion_loss, loss = self.rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data, DISTORTION_LAMBDA)
            loss.backward()
            self.optimizer.step()
            self.model.bottleneck_update()

            rate_losses.append(rate_loss.item())
            distortion_losses.append(distortion_loss.item())
            total_losses.append(loss.item())

            train_len = len(self.data.train_dl.dataset)
            num_steps = train_len // self.batch_size
            msg = (f"Training Step [Batch {i}/{num_steps}] | "
                   f"Rate Loss: {rate_loss.item():.6f} | "
                   f"Distortion Loss: {distortion_loss.item():.6f} | "
                   f"Total Loss: {loss.item():.6f}")
            print_inline_every(i, TRAIN_PRINT_FREQ, num_steps, msg)

            if i % self.save_freq == 0:
                self.model.save(self.model_dir + f"video_model_{i}.pth")
                self.load_model = f"video_model_{i}.pth"

            if i % EVAL_FREQ == 0 and i > 0:
                # self.model.save(model_dir + f"video_model_{i}.pth")
                self.plot_losses(total_losses, rate_losses, distortion_losses)
                avg_loss, avg_rate_loss, avg_distortion_loss = self.eval()
                if abs(avg_loss - eval_loss) < 0.0001:
                    print("Early stopping")
                    break
                eval_loss = avg_loss

        return total_losses, rate_losses, distortion_losses

    def eval(self):
        self.model.eval()
        total_loss = 0
        total_distortion_loss = 0
        total_rate_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.data.val_dl):
                data = data.to(self.device)

                reconstruction, y_likelihoods, z_likelihoods = self.model(data)
                rate_loss, distortion_loss, loss = self.rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data, DISTORTION_LAMBDA)

                total_loss += loss.item()
                total_distortion_loss += distortion_loss.item()
                total_rate_loss += rate_loss.item()

        len_eval = len(self.data.val_dl.dataset)
        avg_loss = total_loss / len_eval
        avg_distortion_loss = total_distortion_loss / len_eval
        avg_rate_loss = total_rate_loss / len_eval
        print(f"\nEvaluation Step | "
              f"Avg Rate Loss: {avg_rate_loss:.4f} | "
              f"Avg Distortion Loss: {avg_distortion_loss:.4f} | "
              f"Avg Total Loss: {avg_loss:.4f}\n\n")
        self.model.train()
        return avg_loss, avg_rate_loss, avg_distortion_loss
    
    def simulate(self):
        state_dict = torch.load(self.model_dir + "video_model.pth")
        encoder = VideoModelEncoder(self.ch_network, self.ch_compress, self.batch_size, state_dict)
        decoder = VideoModelDecoder(self.ch_network, self.ch_compress, self.batch_size, state_dict)

        with torch.no_grad():
            original_imgs_np = []
            reconstructed_imgs_np = []
            for data, _ in self.data.example_dl:
                data = data.to(self.device)
                reconstruction, _, _ = self.model(data)

                z_strings = encoder.encode_hyper(data)
                # z_strings_noisy = z_strings     # TODO: Add noise and drop data to simulate transmission
                self.model.decode_hyper(z_strings)
                
                y_strings = encoder.encode_image()
                # y_strings_noisy = y_strings     # TODO: Add noise and drop data to simulate transmission
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
            plt.savefig("simulation.png")            

    def plot_losses(self, total_losses, rate_losses, distortion_losses):
        plt.plot(total_losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.savefig(self.plot_dir + "losses.png")


if __name__ == "__main__":
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(42)
    # torch.backends.cudnn.deterministic = True

    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    if not os.path.exists(EXPORT_DIR): os.makedirs(EXPORT_DIR)

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
        load_model="video_model_1200.pth"
    )

    print("\n")
    trainer.train()
