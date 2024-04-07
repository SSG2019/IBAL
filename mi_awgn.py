import math
import numpy as np

import torch
import torch.nn as nn
from modules import Modules
from torchvision.models import vgg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 功率约束
def PowerNormalize(z):
    z_square = torch.mul(z, z)
    power = torch.mean(z_square).sqrt()
    if power > 1:
        z = torch.div(z, power)
    return z

# SNR
def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


# 模型反演攻击模块
class MI(nn.Module):
    def __init__(self):
        super(MI, self).__init__()
        self.fc = nn.Linear(64, 784)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)

        return x


class Perceptual(nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        self.vggmodel = vgg.vgg16(pretrained=True).features[0:3]
        self.vggmodel.eval()
        self.vggmodel.to(device)
        self.lc = nn.MSELoss().to(device)

    def forward(self, x, y):
        out1 = self.vggmodel(x)
        out2 = self.vggmodel(y)
        loss = self.lc(out1, out2)

        return loss


class TMI(nn.Module):
    def __init__(self, args):
        super(TMI, self).__init__()
        self.model = Modules(args)
        self.model.load_state_dict(torch.load('./model_awgn/MNIST_SNR_19_epoch_106.pth')['model'])
        self.mi = MI()

    def forward(self, x, snr):
        z, _ =self.model.transmitter(x)
        z = PowerNormalize(z)
        channel_noise = SNR_to_noise(snr)
        z_hat = z + torch.normal(0, channel_noise, size=z.shape).to(device)
        z_hat = z_hat.detach()
        x_hat = self.mi(z_hat)

        return x_hat