import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from data_processing import denormalize

def show_imgs(imgs , save_name=""):
    sqrtn = int(np.ceil(np.sqrt(imgs.shape[0])))
    
    grid = torchvision.utils.make_grid(imgs , nrow=sqrtn , padding=2)
    fig = plt.figure(figsize=(sqrtn, sqrtn))

    ax = plt.subplot(1,1,1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(np.transpose(grid , (1,2,0)))
    plt.show()

    if save_name != "" and not test:
        plt.savefig(save_name)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
        
def sampler_noise(BATCH_SIZE , NOISE_DIM):
    return torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1)

class Generator(nn.Module):
    def __init__(self, nz , ngf , nc , extra_layers=0 , ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.02 , inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.02 , inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.02 , inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.02 , inplace=True),
            nn.Conv2d(ngf , ngf , 3 , 1 , 1 , bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.02 , inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )
        self.main = main

    def forward(self, x):
        output = self.main(x)
        return output
    
class Discriminator(nn.Module):
    def __init__(self , nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2,  inplace=True)
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)

