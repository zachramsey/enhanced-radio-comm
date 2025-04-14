
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
    def __init__(self, network_channels: int, compress_channels: int):
        super().__init__(network_channels, compress_channels)
        self.eval()

    # def decode_hyper(self, z_string: ByteTensor) -> None:
    #     '''
    #     Decode the hyper latent

    #     Upon receiving the compressed hyper latent from the remote device,
    #     the control device will call this function to decompress the hyper latent.

    #     Parameters
    #     ----------
    #     z_strings : ByteTensor
    #         Quantized hyper-prior latent
    #     '''
    #     # Dequantize the latent hyper-prior
    #     z_hat = self.hyper_bottleneck.decompress([z_string], (8, 10))

    #     # Decode hyper-prior from latent hyper-prior
    #     self.hyper_params = self.hyper_synthesis(z_hat)


    # def decode_image(self, y_string: ByteTensor) -> ByteTensor:
    #     '''
    #     Decode the image latent

    #     After the compressed hyper latent is decompressed, the control device 
    #     will call this function to complete the reconstruction of the image data.

    #     Parameters
    #     ----------
    #     y_strings : ByteTensor
    #         Quantized image latent
        
    #     Returns
    #     -------
    #     x_hat : ByteTensor
    #         Reconstructed image data | *(height, width, channels)*
    #     '''
    #     # Dequantize the latent image
    #     sigma_hat, means_hat = self.entropy_parameters(self.hyper_params).chunk(2, 1)

    #     # Build the scale indexes for quantization
    #     indexes = self.image_bottleneck.build_indexes(sigma_hat)

    #     # Dequantize the latent image with the scale indexes and hyper-prior means
    #     y = self.image_bottleneck.decompress([y_string], indexes, means=means_hat)

    #     # Decode image from dequantized latent image
    #     x_hat = self.image_synthesis(y)

    #     # Convert the tensor to the expected output format
    #     x_hat = x_hat.squeeze(0).permute(1, 2, 0)   # 1 x C x H x W -> H x W x C
    #     min = x.min()
    #     x = (x - min) / (x.max() - min)             # Normalize to [0, 1]
    #     x = (x * 255).round().to(ByteTensor)        # [0.0, 1.0] -> [0, 255]

    #     return x_hat
    

    def forward(self, z_string: ByteTensor, y_string: ByteTensor) -> ByteTensor:
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
        # Dequantize the latent hyper-prior
        z_hat = self.hyper_bottleneck.decompress(z_string, (8, 10))

        # Decode hyper-prior from latent hyper-prior
        hyper_params = self.hyper_synthesis(z_hat)

        # Dequantize the latent image
        sigma_hat, means_hat = self.entropy_parameters(hyper_params).chunk(2, 1)

        # Build the scale indexes for quantization
        indexes = self.image_bottleneck.build_indexes(sigma_hat)

        # Dequantize the latent image with the scale indexes and hyper-prior means
        y = self.image_bottleneck.decompress(y_string, indexes, means=means_hat)

        # Decode image from dequantized latent image
        x_hat = self.image_synthesis(y)

        # Convert the tensor to the expected output format
        x_hat = x_hat.permute(0, 2, 3, 1)               # 1 x C x H x W -> 1 x H x W x C
        min = x_hat.min()
        x_hat = (x_hat - min) / (x_hat.max() - min)     # Normalize to [0, 1]
        x_hat = (x_hat * 255).round().byte()            # [0.0, 1.0] -> [0, 255]

        return x_hat
