
# import torch
# from torch import ByteTensor
# from .model import VideoModel

# class VideoDecoder(VideoModel):
#     '''
#     Video Decoder Model

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
#     decode(z_string, y_string) -> ByteTensor
#         Decompress the hyper latent and image latent to RGB888-encoded ByteTensor
#     '''
#     def __init__(self, network_channels: int, compress_channels: int, device: str = "cpu"):
#         super().__init__(network_channels, compress_channels)
#         self.device = device
#         self.to(device)

#     def forward(self, z_string: torch.CharTensor, y_string: torch.CharTensor) -> ByteTensor:
#         '''
#         Decompress the data to RGB888-encoded output

#         Parameters
#         ----------
#         z_strings : list
#             Quantized hyper-prior latent
#         y_strings : list
#             Quantized image latent

#         Returns
#         -------
#         x_hat : ByteTensor
#             Reconstructed image data | *(height, width, channels)*
#         '''

#         z_string = z_string.float()
#         y_string = y_string.float()

#         # Dequantize the latent hyper-prior
#         z = self.hyper_bottleneck.dequantize(z_string, (self.c_network, 8, 10))

#         # Decode hyper-prior from latent hyper-prior
#         hyper_params = self.hyper_synthesis(z)

#         # Dequantize the latent image
#         means_hat = self.mean_params(hyper_params)

#         # Dequantize the latent image with the scale indexes and hyper-prior means
#         y = self.image_bottleneck.dequantize(y_string, (self.c_compress, 30, 40), means=means_hat)

#         # Decode image from dequantized latent image
#         x_hat = self.image_synthesis(y)

#         # Convert the tensor to the expected output format
#         x_hat = x_hat.squeeze(0)                            # 1 x C x H x W -> C x H x W
#         x_hat = x_hat.permute(1, 2, 0)                      #     C x H x W -> H x W x C
#         x_hat = (x_hat * 255).round().byte().contiguous()   #    [0.0, 1.0] -> [0, 255]

#         return x_hat

import torch
import torch.nn as nn

from compressai.layers import GDN
from .bottlenecks import EntropyBottleneck, GaussianConditional

class VideoDecoder(nn.Module):
    '''
    Video Decoder Model

    Parameters
    ----------
    network_channels : int
        Number of channels in the network
    compress_channels : int
        Number of channels for compression

    Methods
    -------
    decode(z_string, y_string) -> ByteTensor
        Decompress the hyper latent and image latent to RGB888-encoded ByteTensor
    load(path: str)
        Load the model from a file
    save(path: str)
        Save the model to a file
    '''
    
    def __init__(self, c_network: int, c_compress: int, device: str = "cpu"):
        super(VideoDecoder, self).__init__()
        self.device = device

        self.c_network = c_network
        self.c_compress = c_compress

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

        self.to(device)
        self.eval()

    def forward(self, z_string: torch.CharTensor, y_string: torch.CharTensor) -> torch.ByteTensor:
        '''
        Decompress the data to RGB888-encoded output

        Parameters
        ----------
        z_strings : list
            Quantized hyper-prior latent
        y_strings : list
            Quantized image latent

        Returns
        -------
        x_hat : ByteTensor
            Reconstructed image data | *(height, width, channels)*
        '''

        z_string = z_string.float()
        y_string = y_string.float()

        # Dequantize the latent hyper-prior
        z = self.hyper_bottleneck.dequantize(z_string, (self.c_network, 8, 10))

        # Decode hyper-prior from latent hyper-prior
        hyper_params = self.hyper_synthesis(z)

        # Dequantize the latent image
        means_hat = self.mean_params(hyper_params)

        # Dequantize the latent image with the scale indexes and hyper-prior means
        y = self.image_bottleneck.dequantize(y_string, (self.c_compress, 30, 40), means=means_hat)

        # Decode image from dequantized latent image
        x_hat = self.image_synthesis(y)

        # Convert the tensor to the expected output format
        x_hat = x_hat.squeeze(0)                            # 1 x C x H x W -> C x H x W
        x_hat = x_hat.permute(1, 2, 0)                      #     C x H x W -> H x W x C
        x_hat = (x_hat * 255).round().byte().contiguous()   #    [0.0, 1.0] -> [0, 255]

        return x_hat
    
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
