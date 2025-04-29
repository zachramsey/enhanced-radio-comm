
import torch
from torch import ByteTensor
from .model import VideoModel

class VideoDecoder(VideoModel):
    '''
    Video Decoder Model

    Parameters
    ----------
    network_channels : int
        Number of channels in the network
    compress_channels : int
        Number of channels for compression
    batch_size : int
        Batch size for processing images

    Attributes
    ----------
    image_analysis : torch.nn.Module
        Image analysis module
    hyper_analysis : torch.nn.Module
        Hyper analysis module
    hyper_bottleneck : torch.nn.Module
        Hyper bottleneck module
    hyper_synthesis : torch.nn.Module
        Hyper synthesis module
    entropy_parameters : torch.nn.Module
        Entropy parameters module
    image_bottleneck : torch.nn.Module
        Image bottleneck module

    Methods
    -------
    decode(z_string, y_string) -> ByteTensor
        Decompress the hyper latent and image latent to RGB888-encoded ByteTensor
    '''
    def __init__(self, network_channels: int, compress_channels: int, device: str = "cpu"):
        super().__init__(network_channels, compress_channels)
        self.device = device
        self.to(device)

    def forward(self, z_string: torch.CharTensor, y_string: torch.CharTensor) -> ByteTensor:
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
