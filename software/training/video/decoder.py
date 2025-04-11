
from torch import ByteTensor
from model import VideoModel

class VideoDecoder(VideoModel):
    def __init__(self, network_channels: int, compress_channels: int, batch_size: int):
        '''
        Initialize the decoder model

        Parameters
        ----------
        network_channels : int
            Number of channels in the network
        compress_channels : int
            Number of channels for compression
        batch_size : int
            Batch size for processing images
        '''
        super().__init__(network_channels, compress_channels, batch_size)
        self.eval()

    def decode_hyper(self, z_string: ByteTensor) -> None:
        '''
        Decode the hyper latent

        Upon receiving the compressed hyper latent from the remote device,
        the control device will call this function to decompress the hyper latent.

        Parameters
        ----------
        z_strings : ByteTensor
            Quantized hyper-prior latent
        '''
        # Dequantize the latent hyper-prior
        z_hat = self.hyper_bottleneck.decompress([z_string], (8, 10))

        # Decode hyper-prior from latent hyper-prior
        self.hyper_params = self.hyper_synthesis(z_hat)


    def decode_image(self, y_string: ByteTensor) -> ByteTensor:
        '''
        Decode the image latent

        After the compressed hyper latent is decompressed, the control device 
        will call this function to complete the reconstruction of the image data.

        Parameters
        ----------
        y_strings : ByteTensor
            Quantized image latent
        
        Returns
        -------
        x_hat : ByteTensor
            Reconstructed image data | *(height, width, channels)*
        '''
        # Dequantize the latent image
        sigma_hat, means_hat = self.entropy_parameters(self.hyper_params).chunk(2, 1)

        # Build the scale indexes for quantization
        indexes = self.image_bottleneck.build_indexes(sigma_hat)

        # Dequantize the latent image with the scale indexes and hyper-prior means
        y = self.image_bottleneck.decompress([y_string], indexes, means=means_hat)

        # Decode image from dequantized latent image
        x_hat = self.image_synthesis(y)

        # Convert the reconstructed image to a byte tensor
        x_hat = (x_hat.clamp(0, 1) * 255).round().to(ByteTensor)
        
        # Reshape the tensor to a flat tensor
        x_hat = x_hat.squeeze(0).permute(1, 2, 0).flatten()

        return x_hat
    

    def decode(self, z_string: ByteTensor, y_string: ByteTensor) -> ByteTensor:
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
        z_hat = self.hyper_bottleneck.decompress([z_string], (8, 10))

        # Decode hyper-prior from latent hyper-prior
        hyper_params = self.hyper_synthesis(z_hat)

        # Dequantize the latent image
        sigma_hat, means_hat = self.entropy_parameters(hyper_params).chunk(2, 1)

        # Build the scale indexes for quantization
        indexes = self.image_bottleneck.build_indexes(sigma_hat)

        # Dequantize the latent image with the scale indexes and hyper-prior means
        y = self.image_bottleneck.decompress([y_string], indexes, means=means_hat)

        # Decode image from dequantized latent image
        x_hat = self.image_synthesis(y)

        # Convert the reconstructed image to a byte tensor
        x_hat = (x_hat.clamp(0, 1) * 255.0).round().to(ByteTensor)
        
        # Reshape the tensor to a flat tensor
        x_hat = x_hat.squeeze(0).permute(1, 2, 0).flatten()

        return x_hat
