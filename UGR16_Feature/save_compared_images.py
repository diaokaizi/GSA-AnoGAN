import sys
sys.path.append("..")
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fanogan.save_compared_images import save_compared_images

from model import Generator, Encoder
from tools import SimpleDataset, load_NF_UNSW_NB15


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, (x_test, y_test) = load_NF_UNSW_NB15()
    test_mnist = SimpleDataset(x_test, y_test,
                               transform=transforms.Compose(
                                   [transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
                               )
    test_dataloader = DataLoader(test_mnist, batch_size=opt.n_grid_lines,
                                 shuffle=True)

    generator = Generator(opt)
    encoder = Encoder(opt)

    save_compared_images(opt, generator, encoder, test_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grid_lines", type=int, default=10,
                        help="number of grid lines in the saved image")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=3,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_iters", type=int, default=None,
                        help="value of stopping iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
