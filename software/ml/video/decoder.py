
from model import VideoModel

class VideoModelDecoder(VideoModel):
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

    def decode_hyper(self, z_strings):
        '''
        Decode the hyper latent

        Upon receiving the compressed hyper latent from the remote device,
        the control device will call this function to decompress the hyper latent.

        Parameters
        ----------
        z_strings : list
            Quantized hyper-prior latent

        Returns
        -------
        scales : Tensor
            Decompressed scales
        '''
        # Dequantize the latent hyper-prior
        self.z_hat = self.hyper_bottleneck.decompress(z_strings, (8, 10))

        # Decode hyper-prior from latent hyper-prior
        self.hyper_params = self.hyper_synthesis(self.z_hat)


    def decode_image(self, y_strings):
        '''
        Decode the image latent

        After the compressed hyper latent is decompressed, the control device 
        will call this function to complete the reconstruction of the image data.

        Parameters
        ----------
        y_strings : list
            Quantized image latent
        
        Returns
        -------
        x_hat : Tensor
            Reconstructed image data | *(batch_size, channels, height, width)*
        '''
        # Dequantize the latent image
        self.sigma_hat, self.means_hat = self.entropy_parameters(self.hyper_params).chunk(2, 1)

        # Build the scale indexes for quantization
        self.indexes = self.image_bottleneck.build_indexes(self.sigma_hat)

        # Dequantize the latent image with the scale indexes and hyper-prior means
        self.y = self.image_bottleneck.decompress(y_strings, self.indexes, means=self.means_hat)

        # Decode image from dequantized latent image
        self.x_hat = self.image_synthesis(self.y)

        return self.x_hat
