import torch
import torch.nn as nn
from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class VideoEncoder(nn.Module):
    def __init__(self, c_network: int, c_compress: int):
        super(VideoEncoder, self).__init__()

        self.image_analysis = nn.Sequential(
            nn.Conv2d(3, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            GDN(c_network),
            nn.Conv2d(c_network, c_compress, 5, stride=2, padding=2),
        )

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

        self.hyper_analysis = nn.Sequential(
            nn.Conv2d(c_compress, c_network, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_network, c_network, 5, stride=2, padding=2),
        )

        self.hyper_synthesis = nn.Sequential(
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_network, 5, stride=2, output_padding=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_network, c_compress, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):   # x: (batch_size, channels, height, width)
        # self.image_analysis(x)
        # self.image_synthesis(x)
        # self.hyper_analysis(torch.abs(x))
        # self.hyper_synthesis(x)
        pass