import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x*self.weight

class Transmitter(nn.Module):
    def __init__(self, args):
        super(Transmitter, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(784, args.intermediate_dim)
        self.upper_tri_matrix = torch.triu(torch.ones((args.intermediate_dim, args.intermediate_dim))).to(device)

    def forward(self, x):
        x = x.view(-1, int(x.nelement() / x.shape[0]))

        weight = self.fc1.weight
        bias = self.fc1.bias
        l2_norm_squared = torch.sum(weight.pow(2), dim=1) + bias.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        fc1_weight = (weight.permute(1, 0) / l2_norm).permute(1, 0)
        fc1_bias = bias / l2_norm
        x = F.linear(x, fc1_weight, fc1_bias)

        noise = self.SNR_to_noise(self.args.snr)
        channel_noise = torch.FloatTensor([1]) * noise
        channel_noise = channel_noise.to(device)

        mu = nn.Parameter(torch.ones(self.args.intermediate_dim)).to(device)
        mu = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu, min=1e-4).to(device)
        x = torch.tanh(mu * x)
        KL = self.KL_log_uniform(channel_noise ** 2 / (x.pow(2) + 1e-4))

        return x, KL

    def KL_log_uniform(self, alpha_squared):
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695
        batch_size = alpha_squared.size(0)
        KL_term = k1 * F.sigmoid(k2 + k3 * torch.log(alpha_squared)) - 0.5 * F.softplus(
            -1 * torch.log(alpha_squared)) - k1

        return - torch.sum(KL_term) / batch_size

    def SNR_to_noise(self, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        return noise_std


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.hidden_channel = args.intermediate_dim

        self.fc2 = nn.Linear(args.intermediate_dim, 1024)
        self.fc2_2 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(1024 + 16, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x, KL):
        noise = self.SNR_to_noise(self.args.snr)
        channel_noise = torch.FloatTensor([1]) * noise
        channel_noise = channel_noise.to(device)
        noise_feature = self.fc2_2(channel_noise)
        noise_feature = noise_feature.expand(x.size()[0], 16)
        x = F.relu(self.fc2(x))
        x = torch.cat((x, noise_feature), dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1), KL * (0.1 / channel_noise)

    def SNR_to_noise(self, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        return noise_std

# decoder模块
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(64, 784)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)

        return x


class Modules(nn.Module):
    def __init__(self, args):
        super(Modules, self).__init__()
        self.transmitter = Transmitter(args)
        self.classifier = Classifier(args)
        self.decoder = Decoder()

