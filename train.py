import os
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from aging.dataloader import ImageTargetFolder
from aging.nets import Encoder, Generator, Dimg, Dz

# Setting environmental variable to resolve error:
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Dataset

DESTINATION_DIR = 'data'
image_size = 128
batch_size = 32
ngpu = 0
num_epochs = 10

dataset = ImageTargetFolder(root=DESTINATION_DIR,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

regex = '^.*?([0-9].*?)_([0-9])_.*$'
labels = [int(re.findall(regex, x[0])[0][0]) for x in dataset.imgs]
dataset.targets = labels

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))


# plt.figure(figsize=(8, 8))
# plt.axis('off')
# plt.title("Example images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Initialize neural networks
netG = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Initialize neural networks
netE = Encoder().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netE.apply(weights_init)

# Initialize neural networks
netDz = Dz().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netDz.apply(weights_init)

# Loss for Encoder-Generator training

L2loss = nn.MSELoss()
EGoptim = torch.optim.Adam(netG.parameters(), betas=(0.5, 0.5), lr=2e-4)

#TODO - inspect problem with dataloader.sample - different label order

for index, data in enumerate(dataloader):
    image, label = data
    image = torch.autograd.Variable(image).cpu()
    codes = netE(image)

    sample = torch.rand(codes.shape)
    z = torch.cat((sample, label[:, np.newaxis, np.newaxis].float()), dim=2)
    output = netG(z)
    loss = L2loss(output, image)
    EGoptim.zero_grad()
    loss.backward()
    EGoptim.step()

    if index % 50 == 0:
        print(f"Loss for {index} batch is equal to {loss}")
