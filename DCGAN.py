from __future__ import print_function
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

initial_vector_size = 100
n_channel = 1
image_size = 64
batch_size = 64
n_samples = batch_size * 10


dataset = dset.MNIST(root='data', train=False, download=True,#classes=['church_outdoor_val'],
                    transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            #transforms.Normalize((0.1307,), (0.3081,))
                        ]))

num_train = len(dataset)
indices = list(range(num_train))
train_idx = np.random.choice(indices, size=n_samples, replace=False)
train_sampler = SubsetRandomSampler(train_idx)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    '''
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    '''


class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x



def trainIters(generator, discriminator, batch_size, n_iters=50, lr=0.0001):
    start_time = time.time()
    criterion = nn.BCELoss()
    # setup optimizer
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr*2, betas=(0.5, 0.999))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    fixed_z_ = torch.randn((batch_size, 100)).to(device).view(-1, 100, 1, 1)

    for epoch in range(n_iters):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        for x_, _ in dataloader:
            # train discriminator D
            discriminator.zero_grad()

            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch).to(device)
            y_fake_ = torch.zeros(mini_batch).to(device)

            x_ = x_.to(device)
            D_result = discriminator(x_).squeeze()
            D_real_loss = criterion(D_result, y_real_)

            z_ = torch.randn((mini_batch, 100)).to(device).view(-1, 100, 1, 1)
            G_result = generator(z_)

            D_result = discriminator(G_result).squeeze()
            D_fake_loss = criterion(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            optimizerD.step()

            D_losses.append(D_train_loss.item())

            # train generator G
            generator.zero_grad()
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = z_.to(device)

            G_result = generator(z_)
            D_result = discriminator(G_result).squeeze()
            G_train_loss = criterion(D_result, y_real_)
            G_train_loss.backward()
            optimizerG.step()

            G_losses.append(G_train_loss.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), n_iters, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
            torch.mean(torch.FloatTensor(G_losses))))

        torch.save(generator, 'model/generator.pkl')
        torch.save(discriminator, 'model/discriminator.pkl')

        utils.save_image(x_, 'result/real_samples.png', normalize=True)
        fake = generator(fixed_z_)
        utils.save_image(fake.detach(), 'result/fake_samples_epoch_{}.png'.format(epoch + 1), normalize=True)

    end_time = time.time()
    total_ptime = end_time - start_time

    print("Training finish!... Total time: ", total_ptime)


if __name__ == '__main__':

    if os.path.exists('model/discriminator.pkl'):
        discriminator = torch.load('model/discriminator.pkl')
    else:
        discriminator = Discriminator().to(device)
        discriminator.apply(weights_init)
    if os.path.exists('model/generator.pkl'):
        generator = torch.load('model/generator.pkl')
    else:
        generator = Generator().to(device)
        generator.apply(weights_init)

    # Train the model
    trainIters(generator, discriminator, batch_size=batch_size, n_iters=10, lr=0.0001)