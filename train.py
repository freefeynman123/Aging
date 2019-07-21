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
from aging.utils import convert_age, index_to_one_hot

# Setting environmental variable to resolve error:
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Dataset

DESTINATION_DIR = 'data'
image_size = 128
batch_size = 32
ngpu = 0
num_epochs = 10
num_classes = 11
lr = 2e-4
betas = (0.5, 0.5)
SEED = 42
torch.manual_seed(SEED)

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
netE = Encoder().to(device)
netG = Generator().to(device)
netDz = Dz().to(device)
netDimg = Dimg().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netE.apply(weights_init)
netG.apply(weights_init)
netDz.apply(weights_init)
netDimg.apply(weights_init)

# Loss for Encoder-Generator training

L2loss = nn.MSELoss()
CEloss = nn.CrossEntropyLoss()
optimizerE = torch.optim.Adam(netE.parameters(), betas=betas, lr=lr)
optimizerG = torch.optim.Adam(netG.parameters(), betas=betas, lr=lr)
optimizerDz = torch.optim.Adam(netDz.parameters(), betas=betas, lr=lr)
optimizerDimg = torch.optim.Adam(netDimg.parameters(), betas=betas, lr=lr)


for index, data in enumerate(dataloader):
    #Extracting data and converting labels to one hot encoded format
    image, label = data
    label = torch.tensor([convert_age(age) for age in label])
    label = index_to_one_hot(label, num_classes)
    image = torch.autograd.Variable(image).cpu()
    #Training encoder (using procedure for conditional variational autoencoder)
    z = netE(image)
    #Training Dz - keeping distribution of z close to uniform
    #Training on sample part
    netDz.zero_grad()
    sample = torch.autograd.Variable(torch.rand(z.shape))
    output_sample = netDz(sample)
    #Training on generated codes part
    output_z = netDz(z)
    #Calculating loss
    #TODO - think how to train Dz - probably give up cross entropy
    lossDzE = CEloss(output_z.view(-1, 1), output_sample.view(-1, 1))
    lossDzE.backward()
    #Optimizing networks
    optimizerE.step()
    optimizerDz.step()
    #Generator is going to be further optimized by Dimg
    z_label = torch.cat((z, label[:, np.newaxis].float()), dim=2)
    output = netG(z_label)
    lossG = L2loss(output, image)
    optimizerG.zero_grad()
    lossG.backward()
    optimizerG.step()
    #Training Dimg
    netDimg.zero_grad()
    #Reshape label to match image's shape after first convolution
    # label_reshaped =
    # Training on generated codes part
    output_z = netDz(z)
    # Calculating loss
    lossDzE = CEloss(output_sample, output_z)
    lossDzE.backward()




    if index % 50 == 0:
        print(f"Loss for {index} batch is equal to {loss}")