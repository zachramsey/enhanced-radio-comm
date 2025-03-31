
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
    def __init__(self, dataset: ImageDataLoader):
        self.data = dataset
        self.model = VideoModel(NETWORK_CHANNELS, COMPRESS_CHANNELS, BATCH_SIZE).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    @staticmethod
    def _rate_distortion_loss(reconstruction, latent_likelihoods, hyper_latent_likelihoods, original, distortion_lambda):
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width
        bits = (latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()) / -math.log(2)
        bpp_loss = bits / num_pixels
        distortion_loss = F.mse_loss(reconstruction, original)
        combined_loss = distortion_lambda * 255 ** 2 * distortion_loss + bpp_loss
        return bpp_loss, distortion_loss, combined_loss

    def train(self):
        self.model.train()
        rate_losses = []
        distortion_losses = []
        total_losses = []
        eval_losss = np.inf
        for i, (data, _) in enumerate(self.data.train):
            data = data.to(DEVICE)

            self.optimizer.zero_grad()
            reconstruction, y_likelihoods, z_likelihoods = self.model(data)
            rate_loss, distortion_loss, loss = self._rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data, DISTORTION_LAMBDA)
            loss.backward()
            self.optimizer.step()
            self.model.bottleneck_update()

            rate_losses.append(rate_loss.item())
            distortion_losses.append(distortion_loss.item())
            total_losses.append(loss.item())

            train_len = len(self.data.train.dataset)
            num_steps = train_len // BATCH_SIZE
            msg = (f"Train Step: {i * len(data)}/{train_len} | "
                   f"Rate Loss: {rate_loss.item():.6f} | "
                   f"Distortion Loss: {distortion_loss.item():.6f} | "
                   f"Total Loss: {loss.item():.6f}")
            print_inline_every(i, TRAIN_PRINT_FREQ, num_steps, msg)
            print("test")

            # if i % SAVE_MODEL_FREQ == 0:
            #     self.model.save(MODEL_DIR + f"video_model.pth")

            if i % EVAL_FREQ == 0 and i > 0:
                self.model.save(MODEL_DIR + f"video_model.pth")
                self.plot_losses(total_losses, rate_losses, distortion_losses)
                avg_loss, avg_rate_loss, avg_distortion_loss = self.eval()
                if abs(avg_loss - eval_losss) < 0.001:
                    print("Early stopping")
                    break
                eval_losss = avg_loss

        return total_losses, rate_losses, distortion_losses

    def eval(self):
        self.model.eval()
        total_loss = 0
        total_distortion_loss = 0
        total_rate_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.data.val):
                data = data.to(DEVICE)

                reconstruction, y_likelihoods, z_likelihoods = self.model(data)
                rate_loss, distortion_loss, loss = self._rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data, DISTORTION_LAMBDA)

                total_loss += loss.item()
                total_distortion_loss += distortion_loss.item()
                total_rate_loss += rate_loss.item()

        len_test = len(self.data.test.dataset)
        avg_loss = total_loss / len_test
        avg_distortion_loss = total_distortion_loss / len_test
        avg_rate_loss = total_rate_loss / len_test
        print(f"Eval Step | "
              f"Avg Rate Loss: {avg_rate_loss:.4f} | "
              f"Avg Distortion Loss: {avg_distortion_loss:.4f} | "
              f"Avg Total Loss: {avg_loss:.4f}")
        self.model.train()
        return avg_loss, avg_rate_loss, avg_distortion_loss
    
    def simulate(self):
        if self.data.example is None:
            print("No example dataset available")
            return
        
        state_dict = torch.load(MODEL_DIR + "video_model.pth")
        encoder = VideoModelEncoder(NETWORK_CHANNELS, COMPRESS_CHANNELS, BATCH_SIZE, state_dict)
        decoder = VideoModelDecoder(NETWORK_CHANNELS, COMPRESS_CHANNELS, BATCH_SIZE, state_dict)

        with torch.no_grad():
            original_imgs_np = []
            reconstructed_imgs_np = []
            for data, _ in self.data.example:
                data = data.to(DEVICE)

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
        plt.plot(total_losses)#, label="Total Loss")
        # plt.plot(rate_losses, label="Rate Loss")
        # plt.plot(distortion_losses, label="Distortion Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.savefig(PLOT_DIR + "losses.png")

        
if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    if not os.path.exists(EXPORT_DIR): os.makedirs(EXPORT_DIR)

    dataset = ImageDataLoader(DATASET_DIR, DATASET, BATCH_SIZE, val_pct=0.05, num_examples=5)
    trainer = VideoModelTrainer(dataset)
    trainer.train()
