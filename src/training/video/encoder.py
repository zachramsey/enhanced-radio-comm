
# import torch
# from torch import ByteTensor
# from .model import VideoModel

# class VideoEncoder(VideoModel):
#     '''
#     Video Encoder Model

#     Parameters
#     ----------
#     network_channels : int
#         Number of channels in the network
#     compress_channels : int
#         Number of channels for compression
#     batch_size : int
#         Batch size for processing images

#     Attributes
#     ----------
#     image_analysis : torch.nn.Module
#         Image analysis module
#     hyper_analysis : torch.nn.Module
#         Hyper analysis module
#     hyper_bottleneck : torch.nn.Module
#         Hyper bottleneck module
#     hyper_synthesis : torch.nn.Module
#         Hyper synthesis module
#     entropy_parameters : torch.nn.Module
#         Entropy parameters module
#     image_bottleneck : torch.nn.Module
#         Image bottleneck module

#     Methods
#     -------
#     encode(x) -> tuple[ByteTensor, ByteTensor]
#         Compress the image from RGB888-encoded ByteTensor
#     '''
#     def __init__(self, network_channels: int, compress_channels: int, device: str = "cpu"):
#         super().__init__(network_channels, compress_channels)
#         self.device = device
#         self.to(device)

#     def forward(self, x: ByteTensor) -> tuple[torch.CharTensor, torch.CharTensor]:
#         '''
#         Compress the image from RGB888-encoded input

#         Parameters
#         ----------
#         x : Tensor
#             Input image data | *(height, width, channels)*

#         Returns
#         -------
#         y_string : ByteTensor
#             Quantized image latent
#         z_string : ByteTensor
#             Quantized hyper-prior latent
#         '''
#         # Conform raw image data to the expected format
#         x = x.reshape(1, *x.shape[-3:]).permute(0, 3, 1, 2)
#         x = x.float() / 255.0

#         # Encode latent image from input image
#         y = self.image_analysis(x)

#         # Encode latent hyper-prior from latent image
#         z = self.hyper_analysis(y)

#         # Quantize the latent hyper-prior
#         z_string = self.hyper_bottleneck.quantize(z)

#         # Decode hyper-prior from latent hyper-prior
#         hyper_params = self.hyper_synthesis(z)

#         # Get the hyper-parameters (mean & std of a Gaussian distribution) from the hyper-prior
#         means_hat = self.mean_params(hyper_params)

#         # Quantize the latent image with the scale indexes and hyper-prior means
#         y_string = self.image_bottleneck.quantize(y, means=means_hat)
        
#         return z_string, y_string

import torch
import torch.nn as nn

from compressai.layers import GDN
from .bottlenecks import EntropyBottleneck, GaussianConditional

class VideoEncoder(nn.Module):
    '''
    Video Encoder Model

    Parameters
    ----------
    network_channels : int
        Number of channels in the network
    compress_channels : int
        Number of channels for compression

    Methods
    -------
    encode(x) -> tuple[ByteTensor, ByteTensor]
        Compress the image from RGB888-encoded ByteTensor
    load(path: str)
        Load the model from a file
    save(path: str)
        Save the model to a file
    '''

    def __init__(self, c_network: int, c_compress: int, device: str = "cpu"):
        super(VideoEncoder, self).__init__()
        self.device = device

        self.image_analysis = nn.Sequential(
            nn.Conv2d(3, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_compress, 5, stride=2, padding=2),
        )

        self.hyper_analysis = nn.Sequential(
            nn.Conv2d(c_compress, c_network, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
        )

        self.hyper_bottleneck = EntropyBottleneck(channels=c_network)

        self.hyper_synthesis = nn.Sequential(
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=(0, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_compress, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.image_bottleneck = GaussianConditional(scale_table=[0.11, 0.22, 0.44, 0.88, 1.76, 3.52, 7.04, 14.08])
        self.mean_params = nn.Conv2d(c_compress, c_compress, 3, padding=1)

        self.to(device)
        self.eval()


    def forward(self, x: torch.ByteTensor) -> tuple[torch.CharTensor, torch.CharTensor]:
        '''
        Compress the image from RGB888-encoded input

        Parameters
        ----------
        x : Tensor
            Input image data | *(height, width, channels)*

        Returns
        -------
        y_string : ByteTensor
            Quantized image latent
        z_string : ByteTensor
            Quantized hyper-prior latent
        '''

        # Conform raw image data to the expected format
        x = x.reshape(1, *x.shape[-3:]).permute(0, 3, 1, 2)
        x = x.float() / 255.0

        # Encode latent image from input image
        y = self.image_analysis(x)

        # Encode latent hyper-prior from latent image
        z = self.hyper_analysis(y)

        # Quantize the latent hyper-prior
        z_string = self.hyper_bottleneck.quantize(z)

        # Decode hyper-prior from latent hyper-prior
        hyper_params = self.hyper_synthesis(z)

        # Get the hyper-parameters (mean & std of a Gaussian distribution) from the hyper-prior
        means_hat = self.mean_params(hyper_params)

        # Quantize the latent image with the scale indexes and hyper-prior means
        y_string = self.image_bottleneck.quantize(y, means=means_hat)
        
        return z_string, y_string


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
        self.load_state_dict(torch.load(path), strict=False)
