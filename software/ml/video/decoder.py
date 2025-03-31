
from model import VideoModel

class VideoModelDecoder(VideoModel):
    def __init__(self, network_channels: int, compress_channels: int, batch_size: int, state_dict: dict):
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
        state_dict : dict
            Dictionary containing the state of the model
        '''
        self.model = VideoModel(network_channels, compress_channels, batch_size)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def encode_hyper(self, x):
        '''
        Encode the hyper latent

        The remote device will call this function first to get the quantized hyper-prior latent,
        which can be immediately sent to the control device before the image latent is compressed.

        Parameters
        ----------
        x : Tensor
            Input image data | *(batch_size, channels, height, width)*

        Returns
        -------
        z_strings : list
            Quantized hyper-prior latent
        '''
        # Encode latent image from input image
        self.y = self.image_analysis(x)

        # Encode latent hyper-prior from latent image
        self.z_hat = self.hyper_analysis(self.y)
        self.z_shape = self.z_hat.size()[-2:]

        # Quantize the latent hyper-prior
        self.z_strings = self.hyper_bottleneck.compress(self.z_hat)

        return self.z_strings
    
    def encode_image(self):
        '''
        Encode the image latent

        After the compressed hyper latent is sent to the control device,
        the remote device will call this function to get the compressed image latent,
        which is then sent to the control device for the final reconstruction.

        Returns
        -------
        y_strings : list
            Quantized image latent
        '''

        # Dequantize the latent hyper-prior
        self.z_hat = self.hyper_bottleneck.decompress(self.z_strings, self.z_shape)

        # Decode hyper-prior from latent hyper-prior
        self.hyper_params = self.hyper_synthesis(self.z_hat)

        # Get the hyper-parameters (mean & std of a Gaussian distribution) from the hyper-prior
        self.sigma_hat, self.means_hat = self.entropy_parameters(self.hyper_params).chunk(2, 1)
        
        # Build the scale indexes for quantization
        self.indexes = self.image_bottleneck.build_indexes(self.sigma_hat)

        # Quantize the latent image with the scale indexes and hyper-prior means
        self.y_strings = self.image_bottleneck.compress(self.y, self.indexes, self.means_hat)
        
        return self.y_strings
