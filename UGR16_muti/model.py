import numpy as np
import torch.nn as nn


"""
The code is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_shape = opt.img_size
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 10, normalize=False),
            *block(10, 12),
            *block(12, 14),
            nn.Linear(14, self.img_shape),
            nn.Tanh()
            )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_shape = opt.img_size

        self.features = nn.Sequential(
            nn.Linear(self.img_shape, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 4),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.last_layer = nn.Sequential(
            nn.Linear(4, 1)
            )

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.last_layer(features)
        return validity

    def forward_features(self, img):
        features = self.features(img)
        return features


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_shape = opt.img_size

        self.model = nn.Sequential(
            nn.Linear(self.img_shape, 14),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(14, 12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(12, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, opt.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
