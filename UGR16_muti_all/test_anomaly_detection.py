import sys
sys.path.append("..")
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan_muti.test_anomaly_detection import test_anomaly_detection

from model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UGR16, NormalizeTransform


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, (x_test, y_test) = load_UGR16()
    mean = x_test.mean(axis=0)  # Mean of each feature
    std = x_test.std(axis=0)
    normalize = NormalizeTransform(mean, std)
    train_mnist = SimpleDataset(x_test, y_test,transform=normalize)
    test_dataloader = DataLoader(train_mnist, batch_size=1,
                                  shuffle=False)
    
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    test_anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)



"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=134,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    opt = parser.parse_args()

    main(opt)
