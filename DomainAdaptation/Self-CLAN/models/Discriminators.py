import torch
import torch.nn as nn
from models.spectral_norm import SpectralNorm


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1))
        self.conv3 = SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1))
        self.conv4 = SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1))
        self.classifier = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


class FCDiscriminator_Local(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator_Local, self).__init__()

        self.conv1 = nn.Conv2d(num_classes + 2048, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x


def discriminator(name='FC', num_classes=19, restore_from=None):

    # if name == 'FC':
    model_D = FCDiscriminator(num_classes=num_classes)
    if restore_from != 'None':
        saved_state_dict_D = torch.load(restore_from)
        model_D.load_state_dict(saved_state_dict_D)

    return model_D
