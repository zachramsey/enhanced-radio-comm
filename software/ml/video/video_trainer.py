import sys
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_loader import ImageDataLoader
from video_model import VideoModel
from config import *


class VideoModelTrainer:
    def __init__(
            self,
            dataset: ImageDataLoader,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            network_channels: int,
            compress_channels: int,
            distortion_lambda: float,
            save_model_path: str,
            device: str
        ):
        self.epochs = epochs
        self.batch_size = batch_size

        self.network_channels = network_channels
        self.compress_channels = compress_channels

        self.learning_rate = learning_rate
        self.distortion_lambda = distortion_lambda

        self.save_model_path = save_model_path
        self.device = device

        self.data = dataset
        self.model = VideoModel(self.network_channels, self.compress_channels).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @staticmethod
    def _print_inline_every(iter, freq, term, msg):
        if iter % freq == 0 or iter == term - 1:
            if iter > 0: sys.stdout.write("\033[F\033[K")
            print(msg)

    @staticmethod
    def _tensor_to_image(tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = image.permute(1, 2, 0) # C x H x W  -> H x W x C
        image = image.numpy()
        image = (image * 255).astype('uint8') # Assuming images are normalized to [0, 1]
        return image

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
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            # self.test_epoch()
        self.simulate()
        torch.save(self.model.state_dict(), self.save_model_path)

    def train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, _) in enumerate(self.data.train):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            reconstruction, y_likelihoods, z_likelihoods = self.model(data)
            rate_loss, distortion_loss, loss = self._rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data, self.distortion_lambda)
            loss.backward()
            self.optimizer.step()

            train_len = len(self.data.train.dataset)
            num_steps = train_len // self.batch_size
            msg = (f"Train Epoch: {epoch} [{batch_idx * len(data)}/{train_len} "
                   f"({100. * batch_idx / train_len:.0f}%)]\t"
                   f"Loss: {loss.item():.6f} | "
                   f"Distortion Loss: {distortion_loss.item():.6f} | "
                   f"Rate Loss: {rate_loss.item():.6f}")
            self._print_inline_every(batch_idx, 1, num_steps, msg)

            if batch_idx == 50: break

    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        total_distortion_loss = 0
        total_rate_loss = 0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.data.test):
                data = data.to(self.device)

                reconstruction, y_likelihoods, z_likelihoods = self.model(data)
                rate_loss, distortion_loss, loss = self._rate_distortion_loss(reconstruction, y_likelihoods, z_likelihoods, data, self.distortion_lambda)

                total_loss += loss.item()
                total_distortion_loss += distortion_loss.item()
                total_rate_loss += rate_loss.item()

        len_test = len(self.data.test.dataset)
        avg_loss = total_loss / len_test
        avg_distortion_loss = total_distortion_loss / len_test
        avg_rate_loss = total_rate_loss / len_test
        print(f"====> Test set loss: {avg_loss:.4f} "
              f"Avg Distortion Loss: {avg_distortion_loss:.4f} "
              f"Avg Rate Loss: {avg_rate_loss:.4f}")
        return avg_loss, avg_distortion_loss, avg_rate_loss
    
    def simulate(self):
        if self.data.example is None:
            print("No example dataset available")
            return
        self.model.eval()
        with torch.no_grad():
            original_imgs_np = []
            reconstructed_imgs_np = []
            for data, _ in self.data.example:
                data = data.to(self.device)

                z_strings = self.model.encode_hyper(data)

                # TODO: Add noise and drop data to simulate transmission
                z_strings_noisy = z_strings

                self.model.decode_hyper(z_strings_noisy)
                
                y_strings = self.model.encode_image()

                # TODO: Add noise and drop data to simulate transmission
                y_strings_noisy = y_strings

                reconstruction = self.model.decode_image(y_strings_noisy)

                # Visualize the original and reconstructed image
                original_imgs_np.append(self._tensor_to_image(data))
                reconstructed_imgs_np.append(self._tensor_to_image(reconstruction))

            num_examples = len(original_imgs_np)
            plt.figure(figsize=(10, 5))
            for i in range(num_examples):
                plt.subplot(2, num_examples, i + 1)
                plt.imshow(original_imgs_np[i])
                plt.title("Original Image")
                plt.subplot(2, num_examples, num_examples + i + 1)
                plt.imshow(reconstructed_imgs_np[i])
                plt.title("Reconstructed Image")
            plt.show()


# --- Main Training and Testing ---
if __name__ == "__main__":
    dataset = ImageDataLoader(DATASET_DIR, DATASET, BATCH_SIZE, test_pct=0.05, num_examples=10)
    trainer = VideoModelTrainer(
        dataset,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        NETWORK_CHANNELS,
        COMPRESS_CHANNELS,
        DISTORTION_LAMBDA,
        SAVE_MODEL_PATH,
        DEVICE
    )
    trainer.train()
