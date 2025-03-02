
import torch.nn as nn
from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class VideoEncoder(nn.Module):
    def __init__(self, c_network: int, c_compress: int):
        super(VideoEncoder, self).__init__()

        # g_a
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
        self.hyper_synthesis = nn.Sequential(
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_compress, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Q, AE, AD
        self.image_bottleneck = GaussianConditional(scale_table=None)

        # g_s
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


    def forward(self, x):   # x: (batch_size, channels, height, width)
        ''' Forward pass of the model

        Parameters
        ----------
        x : Tensor
            Input image data | *(batch_size, channels, height, width)*

        Returns
        -------
        Tensor
            Reconstructed image data | *(batch_size, channels, height, width)*
        Tensor
            Latent encoding | *()*
        Tensor
            Hyper latent encoding | *()*
        '''

        # >>> y <- g_a(x)
        self.image_latent = self.image_analysis(x)

        # >>> z <- h_a(y)
        self.hyper_latent = self.hyper_analysis(self.image_latent)
        
        # z_hat <- Q(z)
        # z_enc <- AE(z_hat)
        # z_hat <- AD(z_enc)
        # >>> z_hat, z_enc <- AD(AE(Q(z))), AE(Q(z))
        self.hyper_latent_noisy, self.hyper_latent_likelihoods = self.hyper_bottleneck(self.hyper_latent, training=True)

        # >>> sigma_hat <- h_s(z_hat)
        self.scales = self.hyper_synthesis(self.hyper_latent_noisy)

        # y_hat <- Q(y)
        # y_enc <- AE(y_hat, sigma_hat)
        # y_hat <- AD(y_enc, sigma_hat)
        # >>> y_hat, y_enc <- AD(AE(Q(y), sigma_hat), sigma_hat), AE(Q(y), sigma_hat)
        self.image_latent_noisy, self.image_latent_likelihoods = self.image_bottleneck(self.image_latent, self.scales, training=True)
        
        # >>> x_hat <- g_s(y_hat)
        self.reconstruction = self.image_synthesis(self.image_latent_noisy)

        # >>> return x_hat, y_enc, z_enc
        return self.reconstruction, self.image_latent_likelihoods, self.hyper_latent_likelihoods
