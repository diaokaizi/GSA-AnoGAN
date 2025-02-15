import sys
sys.path.append("..")
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan.train_encoder_izif import train_encoder_izif

from model import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UGR16


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), _ = load_UGR16()
    train_mnist = SimpleDataset(x_train, y_train,
                                transform=transforms.Compose(
                                    [transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])
                                )
    train_dataloader = DataLoader(train_mnist, batch_size=opt.batch_size,
                                  shuffle=True)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    train_encoder_izif(opt, generator, discriminator, encoder,
                       train_dataloader, device)



"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=4,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=3000,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
