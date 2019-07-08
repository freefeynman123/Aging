import torch
from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(
            self,
            channels: int = 64
    ) -> None:
        super(Encoder, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.channels, kernel_size=3, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.channels, out_channels=2*self.channels, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=2*self.channels, out_channels=4*self.channels, kernel_size=3, stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4*self.channels, out_channels=8*self.channels, kernel_size=3, stride=2, padding=2)
        self.fc = nn.Linear(512*self.channels, 50)
    def forward(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass of encoder.
        :param x: Input image to be encoded.
        :return: Encoded image.
        """
        x = nn.ReLU(self.batch1(self.conv1(x)))
        x = nn.ReLU(self.batch2(self.conv2(x)))
        x = nn.ReLU(self.batch3(self.conv3(x)))
        x = nn.ReLU(self.conv4(x))
        x = x.view(1, -1)
        x = self.fc(x)
        return x

class Generator(nn.Module):
    def __init__(
            self,
            max_channels: int = 1024
    ) -> None:
        super(Generator, self).__init__()
        self.max_channels = max_channels
        self.fc = nn.Linear(50, 8*8*self.max_channels)
        self.conv1t = nn.ConvTranspose2d(in_channels=self.max_channels, out_channels=self.max_channels // 2,
                                         kernel_size=3, stride=2, padding=2)
        self.conv2t = nn.ConvTranspose2d(in_channels=self.max_channels // 2, out_channels=self.max_channels // 4,
                                         kernel_size=3, stride=2, padding=2)
        self.conv3t = nn.ConvTranspose2d(in_channels=self.max_channels // 4, out_channels= self.max_channels // 8,
                                         kernel_size=3, stride=2, padding=2)
        self.conv4t = nn.ConvTranspose2d(in_channels=self.max_channels // 8, out_channels= self.max_channels // 16,
                                         kernel_size=3, stride=2, padding=2)
        self.conv1 = nn.Conv2d(in_channels=self.max_channels // 16, out_channels=3, kernel_size=1)

    def forward(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through decoder
        :param x: Encoded image.
        :return: Decoded image.
        """
        x = self.fc(x)
        x = x.view(8, 8, self.max_channels)
        x = self.conv1t(x)
        x = self.conv2t(x)
        x = self.conv3t(x)
        x = self.conv4t(x)
        x = self.conv1(x)
        return x

class Dz(nn.Module):

    def __init__(
            self,
            sample_size: int = 30
    ) -> None:
        super(Dz, self).__init__()
        self.sample_size = sample_size
        self.prior = np.random.uniform(low=0, high=1, size=self.sample_size)
        self.fc1 = nn.Linear(self.sample_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, z, sample_size):
        z = np.random.choice(z, sample_size)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc4(z)
        return z

class Dimg(nn.Module):

    def __init__(self, channels: int, labels: int):
        #TODO think about how to add labels ('n' in the article)
        super(Dimg, self).__init__()
        self.channels = channels
        self.labels = labels
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d(self.labels+self.channels, 2*self.channels, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*self.channels, 4*self.channels, kernel_size=3, stride=2, padding=2)
        self.conv4 = nn.Conv2d(4*self.channels, 8*self.channels, kernel_size=3, stride=2, padding=2)
        self.fc1 = nn.Linear(8*8*8*self.channels, 1024)
        self.fc2 = nn.Linear(1024, 1)
    def forward(self, x):
        #TODO add batch normalization
        x =



