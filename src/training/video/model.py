
import torch
import torch.nn as nn

from compressai.layers import GDN
from .bottlenecks import EntropyBottleneck, GaussianConditional

class VideoModel(nn.Module):
    def __init__(self, c_network: int, c_compress: int):
        super(VideoModel, self).__init__()
        self.c_network = c_network
        self.c_compress = c_compress

        # g_a
        # (3, 480, 640) -> (c_network, 240, 320)
        # (c_network, 240, 320) -> (c_network, 120, 160)
        # (c_network, 120, 160) -> (c_network, 60, 80)
        # (c_network, 60, 80) -> (c_compress, 30, 40)
        self.image_analysis = nn.Sequential(
            nn.Conv2d(3, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_compress, 5, stride=2, padding=2),
        )

        # h_a
        # (c_compress, 30, 40) -> (c_network, 30, 40)
        # (c_network, 30, 40) -> (c_network, 15, 20)
        # (c_network, 15, 20) -> (c_network, 8, 10)
        self.hyper_analysis = nn.Sequential(
            nn.Conv2d(c_compress, c_network, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
        )

        # Q, AE, AD
        self.hyper_bottleneck = EntropyBottleneck(channels=c_network)

        # h_s
        # (c_network, 8, 10) -> (c_network, 15, 20)
        # (c_network, 15, 20) -> (c_network, 30, 40)
        # (c_network, 30, 40) -> (c_compress, 30, 40)
        self.hyper_synthesis = nn.Sequential(
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=(0, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_compress, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Q, AE, AD
        self.image_bottleneck = GaussianConditional(scale_table=[0.11, 0.22, 0.44, 0.88, 1.76, 3.52, 7.04, 14.08])
        self.mean_params = nn.Sequential(
            nn.Conv2d(c_compress, c_compress, 3, padding=1)
        )
        self.scale_params = nn.Sequential(
            nn.Conv2d(c_compress, c_compress, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # g_s
        # (c_compress, 30, 40) -> (c_network, 60, 80)
        # (c_network, 60, 80) -> (c_network, 120, 160)
        # (c_network, 120, 160) -> (c_network, 240, 320)
        # (c_network, 240, 320) -> (3, 480, 640)
        self.image_synthesis = nn.Sequential(
            nn.ConvTranspose2d(c_compress, c_network, 5, stride=2, output_padding=1, padding=2),
            GDN(c_network, inverse=True),
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            GDN(c_network, inverse=True),
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            GDN(c_network, inverse=True),
            nn.ConvTranspose2d(c_network, 3, 5, stride=2, output_padding=1, padding=2),
            nn.Sigmoid()
        )


    def forward(self, x, noise_func = None, **kwargs):   # x: (batch_size, channels, height, width)
        '''
        Forward pass of the model

        This function performs the forward pass of the model, 
        which is used solely for training and sandbox testing.

        Parameters
        ----------
        x : Tensor
            Input image data | *(batch_size, channels, height, width)*

        Returns
        -------
        x_hat : Tensor
            Reconstructed image data | *(batch_size, channels, height, width)*
        y_likelihoods : Tensor
            Likelihood of the latent encoding
        z_likelihoods : Tensor
            Likelihood of the hyper latent encoding
        '''

        # Encode latent image from input image
        self.y = self.image_analysis(x)

        # Encode latent hyper-prior from latent image
        self.z = self.hyper_analysis(self.y)
        
        # Add noise to the latent hyper-prior (for training)
        self.z_hat, self.z_likelihoods = self.hyper_bottleneck(self.z, noise_func=noise_func, **kwargs)
        
        # Decode hyper-prior from compressed latent hyper-prior
        self.hyper_params = self.hyper_synthesis(self.z_hat)

        # Get the hyper-parameters (mean & std of a Gaussian distribution) from the hyper-prior
        self.means_hat = self.mean_params(self.hyper_params)
        self.scales_hat = self.scale_params(self.hyper_params)

        # Add noise to the latent image (for training)
        self.y_hat, self.y_likelihoods = self.image_bottleneck(self.y, self.scales_hat, means=self.means_hat, noise_func=noise_func, **kwargs)

        # Decode image from decompressed latent image
        self.x_hat = self.image_synthesis(self.y_hat)

        return self.x_hat, self.y_likelihoods, self.z_likelihoods
    

    def save(self, path):
        '''
        Save the model state dict

        Parameters
        ----------
        path : str
            Path to save the model state dict
        '''
        torch.save(self.state_dict(), path)


    def load(self, path):
        '''
        Load the model state dict
        
        Parameters
        ----------
        path : str
            Path to load the model state dict
        '''
        self.hyper_bottleneck.update()
        self.image_bottleneck.update()
        self.load_state_dict(torch.load(path))
