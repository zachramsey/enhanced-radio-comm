
from torch import ByteTensor
from . import VideoModel

class VideoEncoder(VideoModel):
    def __init__(self, network_channels: int, compress_channels: int, batch_size: int):
        '''
        Initialize the encoder model

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
        
    def encode_hyper(self, x: ByteTensor) -> ByteTensor:
        '''
        Encode the hyper latent

        The remote device will call this function first to get the quantized hyper-prior latent,
        which can be immediately sent to the control device before the image latent is compressed.

        Parameters
        ----------
        x : ByteTensor
            Input image data | *(height, width, channels)*

        Returns
        -------
        z_string : ByteTensor
            Quantized hyper-prior latent
        '''
        # Conform the raw image data to the expected input shape
        x = x.permute(2, 0, 1).unsqueeze(0)
        
        # Convert the image data to a float tensor and normalize it
        x = x.float() / 255.0

        # Encode latent image from input image
        self.y = self.image_analysis(x)

        # Encode latent hyper-prior from latent image
        self.z_hat = self.hyper_analysis(self.y)

        # Quantize the latent hyper-prior
        z_string = self.hyper_bottleneck.compress(self.z_hat)[0]

        return z_string
    
    def encode_image(self) -> ByteTensor:
        '''
        Encode the image latent

        After the compressed hyper latent is sent to the control device,
        the remote device will call this function to get the compressed image latent,
        which is then sent to the control device for the final reconstruction.

        Returns
        -------
        y_string : ByteTensor 
            Quantized image latent
        '''
        # Decode hyper-prior from latent hyper-prior
        hyper_params = self.hyper_synthesis(self.z_hat)

        # Get the hyper-parameters (mean & std of a Gaussian distribution) from the hyper-prior
        sigma_hat, means_hat = self.entropy_parameters(hyper_params).chunk(2, 1)
        
        # Build the scale indexes for quantization
        indexes = self.image_bottleneck.build_indexes(sigma_hat)

        # Quantize the latent image with the scale indexes and hyper-prior means
        y_string = self.image_bottleneck.compress(self.y, indexes, means_hat)[0]
        
        return y_string
    

    def encode(self, x: ByteTensor) -> tuple[ByteTensor, ByteTensor]:
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
        # Conform the raw image data to the expected input shape
        x = x.permute(2, 0, 1).unsqueeze(0)
        
        # Convert the image data to a float tensor and normalize it
        x = x.float() / 255.0

        # Encode latent image from input image
        y = self.image_analysis(x)

        # Encode latent hyper-prior from latent image
        z_hat = self.hyper_analysis(y)

        # Quantize the latent hyper-prior
        z_string = self.hyper_bottleneck.compress(z_hat)[0]

        # Decode hyper-prior from latent hyper-prior
        hyper_params = self.hyper_synthesis(z_hat)

        # Get the hyper-parameters (mean & std of a Gaussian distribution) from the hyper-prior
        sigma_hat, means_hat = self.entropy_parameters(hyper_params).chunk(2, 1)
        
        # Build the scale indexes for quantization
        indexes = self.image_bottleneck.build_indexes(sigma_hat)

        # Quantize the latent image with the scale indexes and hyper-prior means
        y_string = self.image_bottleneck.compress(y, indexes, means_hat)[0]
        
        return z_string, y_string
