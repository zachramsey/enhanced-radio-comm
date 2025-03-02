
import torch
import torch.nn as nn
from compressai.layers import GDN
from .entropy_models import EntropyBottleneck, GaussianConditional

class VideoEncoder(nn.Module):
    def __init__(self, c_network: int, c_compress: int):
        super(VideoEncoder, self).__init__()

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
        self.image_bottleneck = GaussianConditional(scale_table=None)

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

        self.register_buffer('y', torch.zeros(1))
        self.register_buffer('z_quantized', torch.zeros(1))

        self.register_buffer('scales_infer', torch.zeros(1))


    def forward(self, x):   # x: (batch_size, channels, height, width)
        ''' Forward pass of the model

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

        # >>> y <- g_a(x)
        self.y = self.image_analysis(x)

        # >>> z <- h_a(y)
        self.z = self.hyper_analysis(self.y)
        
        # >>> z_hat, _ <- Q(z)
        self.z_noisy, self.z_likelihoods = self.hyper_bottleneck(self.z, training=True)

        # >>> sigma_hat <- h_s(z_hat)
        self.scales = self.hyper_synthesis(self.z_noisy)

        # >>> y_hat, _ <- Q(y, sigma_hat)
        self.y_noisy, self.y_likelihoods = self.image_bottleneck(self.y, self.scales, training=True)
        
        # >>> x_hat <- g_s(y_hat)
        self.reconstruction = self.image_synthesis(self.y_noisy)

        return self.reconstruction, self.y_likelihoods, self.z_likelihoods


    def encode_hyper(self, x):
        ''' Encode the hyper latent

        The remote device will call this function first to get the compressed hyper latent,
        which can be immediately sent to the control device before the image latent is compressed.

        Parameters
        ----------
        x : Tensor
            Input image data | *(batch_size, channels, height, width)*

        Returns
        -------
        z_quantized : Tensor
            Quantized hyper latent
        z_strings : list
            Compressed hyper latent
        '''

        # >>> y <- g_a(x)
        self.y = self.image_analysis(x)

        # >>> z <- h_a(y)
        self.z = self.hyper_analysis(self.y)

        # >>> z_strings <- Q(z)
        z_strings = self.hyper_bottleneck.compress(self.z)

        # >>> z_hat, _ <- Q(z)
        self.z_quantized, self.z_likelihoods = self.hyper_bottleneck(self.z, training=False)

        return z_strings
    

    def encode_image(self):
        ''' Encode the image latent

        After the compressed hyper latent is sent to the control device,
        the remote device will call this function to get the compressed image latent,
        which is then sent to the control device for the final reconstruction.

        Returns
        -------
        y_strings : list
            Compressed image latent
        '''

        # >>> sigma_hat <- h_s(z_hat)
        self.scales = self.hyper_synthesis(self.z_quantized)

        # >>> y_hat, _ <- Q(y, sigma_hat)
        y_strings = self.image_bottleneck.compress(self.y, self.scales)

        return y_strings
    

    def decode_hyper(self, z_strings):
        ''' Decode the hyper latent

        Upon receiving the compressed hyper latent from the remote device,
        the control device will call this function to decompress the hyper latent.

        Parameters
        ----------
        z_strings : list
            Compressed hyper latent
        '''

        # >>> z_hat <- Q(z_strings)
        self.z_quantized = self.hyper_bottleneck.decompress(z_strings)

        # >>> sigma_hat <- h_s(z_hat)
        self.scales_infer = self.hyper_synthesis(self.z_quantized)


    def decode_image(self, y_strings):
        ''' Decode the image latent

        After the compressed hyper latent is decompressed, the control device 
        will call this function to complete the reconstruction of the image data.

        Parameters
        ----------
        y_strings : list
            Compressed image latent
        
        Returns
        -------
        x_hat : Tensor
            Reconstructed image data | *(batch_size, channels, height, width)*
        '''

        # >>> y_hat <- Q(y_strings, sigma_hat)
        self.y_noisy = self.image_bottleneck.decompress(y_strings, self.scales_infer)

        # >>> x_hat <- g_s(y_hat)
        self.reconstruction = self.image_synthesis(self.y_noisy)

        return self.reconstruction