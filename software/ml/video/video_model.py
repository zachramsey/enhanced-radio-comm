
import torch.nn as nn
from compressai.layers import GDN
# from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from entropy_models import EntropyBottleneck, GaussianConditional

class VideoModel(nn.Module):
    def __init__(self, c_network: int, c_compress: int, batch_size: int = 1):
        super(VideoModel, self).__init__()
        self.c_network = c_network
        self.c_compress = c_compress
        self.batch_size = batch_size

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
        self.image_bottleneck = GaussianConditional(scale_table=[0.11, 0.22, 0.44, 0.88, 1.76, 3.52, 7.04, 14.08])
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(c_compress, 2 * c_compress, 3, padding=1),
            nn.ReLU(inplace=True),
        )

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


    def forward(self, x):   # x: (batch_size, channels, height, width)
        ''' Forward pass of the model

        This function performs the forward pass of the model, 
        which is used solely for training and sandbox testing.

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
        self.z_hat, self.z_likelihoods = self.hyper_bottleneck(self.z, training=True)

        # >>> sigma_hat <- h_s(z_hat)
        self.hyper_params = self.hyper_synthesis(self.z_hat)
        self.scales_hat, self.means_hat = self.entropy_parameters(self.hyper_params).chunk(2, 1)

        # >>> y_hat, _ <- Q(y, sigma_hat)
        self.y_hat, self.y_likelihoods = self.image_bottleneck(self.y, self.scales_hat, means=self.means_hat, training=True)

        # >>> x_hat <- g_s(y_hat)
        self.reconstruction = self.image_synthesis(self.y_hat)

        return self.reconstruction, self.y_likelihoods, self.z_likelihoods
    

    def bottleneck_update(self):
        self.hyper_bottleneck.update()
        self.image_bottleneck.update()


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
        z_strings : list
            Compressed hyper latent
        '''

        # >>> y <- g_a(x)
        self.y = self.image_analysis(x)

        # >>> z <- h_a(y)
        self.z_hat = self.hyper_analysis(self.y)
        self.z_shape = self.z_hat.size()[-2:]

        # >>> z_hat, _ <- Q(z)
        # self.z_enc, _ = self.hyper_bottleneck(self.z_enc, training=False)

        # >>> z_strings <- Q(z)
        self.z_strings = self.hyper_bottleneck.compress(self.z_hat)

        return self.z_strings
    

    def decode_hyper(self, z_strings):
        ''' Decode the hyper latent

        Upon receiving the compressed hyper latent from the remote device,
        the control device will call this function to decompress the hyper latent.

        Parameters
        ----------
        z_strings : list
            Compressed hyper latent

        Returns
        -------
        scales : Tensor
            Decompressed scales
        '''

        # >>> z_hat <- Q(z_strings)
        self.z_hat = self.hyper_bottleneck.decompress(z_strings, (8, 10))

        # >>> sigma_hat <- h_s(z_hat)
        self.hyper_params_dec = self.hyper_synthesis(self.z_hat)
    

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

        self.z_hat = self.hyper_bottleneck.decompress(self.z_strings, self.z_shape)

        # >>> params <- h_s(z_hat)
        self.hyper_params = self.hyper_synthesis(self.z_hat)

        self.scales_hat, self.means_hat = self.entropy_parameters(self.hyper_params).chunk(2, 1)
        
        self.indexes = self.image_bottleneck.build_indexes(self.scales_hat)

        # >>> y_hat, _ <- Q(y, sigma_hat)
        return self.image_bottleneck.compress(self.y, self.indexes, self.means_hat)


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
        self.scales_hat, self.means_hat = self.entropy_parameters(self.hyper_params_dec).chunk(2, 1)

        self.indexes = self.image_bottleneck.build_indexes(self.scales_hat)

        # >>> y_hat <- Q(y_strings, sigma_hat)
        self.y = self.image_bottleneck.decompress(y_strings, self.indexes, means=self.means_hat)

        # >>> x_hat <- g_s(y_hat)
        return self.image_synthesis(self.y)
    