import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import numpy as np
import copy
import os
import math
from PIL import Image
from modules import Modules
from mi_awgn import TMI
from skimage.metrics import structural_similarity as ssim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--intermediate_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--snr', type=int, default=23)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=1e-3)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--decay_step', type=int, default=60)
args = parser.parse_args()

# Dataset
transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train = torchvision.datasets.MNIST(root = './data', train= True, download=False,transform=transf)
test = torchvision.datasets.MNIST(root = './data', train= False, download=False,transform=transf)
dataset_train1_spl, dataset_train2_spl, _ = torch.utils.data.random_split(train, [40000, 20000, len(train) - 60000])
dataset_test_spl, _ = torch.utils.data.random_split(test, [10000, len(test) - 10000])
test_data_loader = torch.utils.data.DataLoader(dataset_test_spl, batch_size=1000, shuffle=False, num_workers=2)  # 用于测试

# 初始化
def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# 固定随机种子
def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# SNR
def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

# 功率约束
def PowerNormalize(z):
    z_square = torch.mul(z, z)
    power = torch.mean(z_square).sqrt()
    if power > 1:
        z = torch.div(z, power)
    return z

# psnr计算
def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0**2 / mse)


def train_dec(model, dx_optimizer, dec_criterion, images):
    model.decoder.train()

    z, KL = model.transmitter(images)
    z = PowerNormalize(z)
    channel_noise = SNR_to_noise(args.snr)
    z_hat = z + torch.normal(0, channel_noise, size=z.shape).to(device)
    x_hat = model.decoder(z_hat)
    x_hat = x_hat.view(-1, 1, 28, 28)
    loss = dec_criterion(x_hat, images)

    dx_optimizer.zero_grad()
    loss.backward()
    dx_optimizer.step()

    return loss


def train_trx(model, tx_optimizer, cx_optimizer, dec_criterion, images, target, epoch):
    model.transmitter.train()
    model.classifier.train()

    z, KL = model.transmitter(images)
    z = PowerNormalize(z)
    channel_noise = SNR_to_noise(args.snr)
    z_hat = z + torch.normal(0, channel_noise, size=z.shape).to(device)
    y_hat, KL = model.classifier(z_hat, KL)
    x_hat = model.decoder(z_hat)
    x_hat = x_hat.view(-1, 1, 28, 28)

    if epoch <= 5:
        loss = F.nll_loss(y_hat, target)
    else:
        anneal_ratio = min(1, (epoch - 5) / 10)
        loss = F.nll_loss(y_hat, target) + args.beta * KL * anneal_ratio
    dec_loss = dec_criterion(x_hat, images)
    loss = loss - 1/(1+channel_noise)*dec_loss

    pred = y_hat.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred))

    tx_optimizer.zero_grad()
    cx_optimizer.zero_grad()
    loss.backward()
    tx_optimizer.step()
    cx_optimizer.step()

    return correct

def test_trx(model, test_data_loader):
    model.transmitter.eval()
    model.classifier.eval()
    correct = 0
    with torch.no_grad():
        for images, target in test_data_loader:
            images = images.to(device)
            target = target.to(device)
            z, KL = model.transmitter(images)
            z = PowerNormalize(z)
            channel_noise = SNR_to_noise(args.snr)
            z_hat = z + torch.normal(0, channel_noise, size=z.shape).to(device)
            y_hat, _ = model.classifier(z_hat, KL)
            pred = y_hat.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        return correct / len(test_data_loader.dataset)


# 面向任务通信网络训练
def main_train():
    print('面向任务通信网络训练')
    test_acc = 0
    kwargs = {'num_workers': 2, 'pin_memory': True}
    model = Modules(args).to(device)
    dec_criterion = nn.MSELoss()  # Decoder损失函数
    dec_criterion = dec_criterion.to(device)
    tx_optimizer = optim.Adam(model.transmitter.parameters(), lr=args.lr, weight_decay=0.0001)
    cx_optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=0.0001)
    dx_optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr, weight_decay=0.0001)

    tx_scheduler = StepLR(tx_optimizer, step_size=args.decay_step, gamma=args.gamma)
    cx_scheduler = StepLR(tx_optimizer, step_size=args.decay_step, gamma=args.gamma)
    dx_scheduler = StepLR(dx_optimizer, step_size=args.decay_step, gamma=args.gamma)

    initNetParams(model)
    for epoch in range(args.epochs):
        print('This is {}-th epoch'.format(epoch))
        print('模拟对手')
        train2_data_loader = torch.utils.data.DataLoader(dataset_train2_spl, batch_size=args.batch_size,
                                                         shuffle=True, **kwargs)
        for n, (images, _) in enumerate(train2_data_loader):
            images = images.to(device)
            for name, param in model.named_parameters():  # 冻结模型参数与更新的参数一块使用
                if "transmitter" in name:
                    param.requires_grad = False
                if "classifier" in name:
                    param.requires_grad = False
                if "decoder" in name:
                    param.requires_grad = True
            train_dec(model, dx_optimizer, dec_criterion, images)
        dx_scheduler.step()

        print('隐私通信')
        train1_data_loader = torch.utils.data.DataLoader(dataset_train1_spl, batch_size=args.batch_size,
                                                         shuffle=True, **kwargs)
        total_correct = 0
        for n, (images, target) in enumerate(train1_data_loader):
            images = images.to(device)
            target = target.to(device)
            for name, param in model.named_parameters(): # 冻结模型参数与更新的参数一块使用
                if "transmitter" in name:
                    param.requires_grad = True
                if "classifier" in name:
                    param.requires_grad = True
                if "decoder" in name:
                    param.requires_grad = False
            correct = train_trx(model, tx_optimizer, cx_optimizer, dec_criterion, images, target, epoch)
            total_correct += correct.sum().item()
        tx_scheduler.step()
        cx_scheduler.step()
        print("训练精度：", total_correct / len(train1_data_loader.dataset))

        if epoch > 100:
            acc = test_trx(model, test_data_loader)
            print('测试精度:', acc)
            if acc > test_acc:
                test_acc = acc
                saved_model = copy.deepcopy(model.state_dict())
                with open('./model_awgn/MNIST_SNR_{}_epoch_{}.pth'.format(args.snr, epoch),'wb') as f:
                    torch.save({'model': saved_model}, f)

# 测试分类精度
def main_test(): # 测试分类精度
    model = Modules(args).to(device)
    model.load_state_dict(torch.load('./model_awgn/MNIST_SNR_23_epoch_101.pth')['model'])
    accuracy = 0
    t = 10
    for i in range(t):
        acc = test_trx(model, test_data_loader)
        accuracy += acc
    print('测试精度:', accuracy / t)

# 模型反演攻击
def main_train_dec(snr): # 训练模型反演攻击时的信噪比
    kwargs = {'num_workers': 2, 'pin_memory': True}
    criterion1 = nn.MSELoss()
    criterion1 = criterion1.to(device)

    tmi = TMI(args).to(device)
    optimizer = optim.Adam(tmi.parameters(), lr=args.lr, weight_decay=0.001)
    print('模型反演攻击')
    for epoch in range(300):
        print('This is {}-th epoch'.format(epoch))
        train2_data_loader = torch.utils.data.DataLoader(dataset_train2_spl, batch_size=args.batch_size,
                                                         shuffle=True, **kwargs)
        for n, (images, _) in enumerate(train2_data_loader):
            images = images.to(device)
            x_hat = tmi(images, snr)
            x_hat = x_hat.view(-1, 1, 28, 28)
            loss = criterion1(x_hat, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    saved_model = copy.deepcopy(tmi.state_dict())
    with open('./model_awgn/MI_SNR_{}_epoch_{}.pth'.format(snr, epoch), 'wb') as f:
        torch.save({'tmi': saved_model}, f)

# 测试模型反演攻击
def main_test_dec(snr):
    tmi = TMI(args).to(device)
    tmi.load_state_dict(torch.load('./model_awgn/19MI_SNR_15_epoch_299.pth')['tmi'])
    test_data_loader = torch.utils.data.DataLoader(dataset_test_spl, batch_size=1, shuffle=False,
                                                   num_workers=2)  # 用于测试
    path1 = './fake_awgn/sampled-'
    path2 = './true_awgn/sampled-'
    list_psnr = []
    list_ssim = []
    tmi.eval()
    with torch.no_grad():
        for n, (images, _) in enumerate(test_data_loader):
            images = images.to(device)
            x_hat = tmi(images, snr)
            x_hat = x_hat.view(-1, 1, 28, 28)
            save_image(x_hat, os.path.join('fake_awgn', 'sampled-{}.png'.format(n)))
            save_image(images, os.path.join('true_awgn', 'sampled-{}.png'.format(n)))

            img_a = Image.open(path1 + str(n) + '.png')
            img_b = Image.open(path2 + str(n) + '.png')
            img_a = np.array(img_a)
            img_b = np.array(img_b)
            psnr_num = psnr(img_a, img_b)
            list_psnr.append(psnr_num)
            ssim_value = ssim(img_a, img_b, multichannel=True)
            list_ssim.append(ssim_value)
        print(np.mean(list_psnr))
        print(np.mean(list_ssim))
if __name__ == '__main__':
    seed_torch(0)
    # main_train() # 训练任务通信网络
    # main_test()  # 测试任务通信网络
    # main_train_dec(15) # 训练模型反演攻击网络
    main_test_dec(19) # 训练模型反演攻击网络


