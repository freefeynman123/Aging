import os
import pickle
import re

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable

from aging.dataloader import ImageTargetFolder
from aging.nets import Encoder, Generator, Dimg, Dz
from aging.parameters import n_l, n_z, image_size, batch_size
from aging.utils import convert_age, index_to_one_hot, total_variation_loss

# Setting environmental variable to resolve error:
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Dataset

DESTINATION_DIR = 'data'
ngpu = 1
num_epochs = 10
num_classes = 10
lr = 2e-4
betas = (0.5, 0.999)
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
labels = [re.findall(regex, x[0])[0] for x in dataset.imgs]
labels = list(map(lambda x: (int(x[0]), int(x[1])), labels))
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

# Definition of optimizers
optimizerE = torch.optim.Adam(netE.parameters(), betas=betas, lr=lr)
optimizerG = torch.optim.Adam(netG.parameters(), betas=betas, lr=lr)
optimizerDz = torch.optim.Adam(netDz.parameters(), betas=betas, lr=lr)
optimizerDimg = torch.optim.Adam(netDimg.parameters(), betas=betas, lr=lr)

# Definition of losses
BCEloss = nn.BCELoss()
L1loss = nn.L1Loss()
L2loss = nn.MSELoss()
CEloss = nn.CrossEntropyLoss()

# Variable to progress/regress age

fixed_l = -torch.ones(80 * 10).view(80, 10).to(device)
for index, value in enumerate(fixed_l):
    value[index // 8] = 1

fixed_l_v = Variable(fixed_l)

# Output directory

output_directory = './results'

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Number of epochs

epochs = 50

for epoch in range(epochs):
    for index, data in enumerate(dataloader):
        # Extracting data and converting labels to one hot encoded format
        image, (label, gender) = data
        label = Variable(label).view(-1, 1)
        gender = Variable(gender.float())
        # Convert age to intervals
        label = torch.tensor([convert_age(age) for age in label])
        # Convert age to one-hot
        label = index_to_one_hot(label, num_classes)
        image = Variable(image.to(device))
        label = label.to(device)
        gender = gender.to(device)
        if epoch == 0 and index == 0:
            fixed_noise = image[:8].repeat(10, 1, 1, 1)
            fixed_g = gender[:8].view(-1, 1).repeat(10, 1)

            fixed_img_v = Variable(fixed_noise)
            fixed_g_v = Variable(fixed_g)

            with open("fixed_noise.pickle", "wb") as file:
                pickle.dump(fixed_noise, file)

        # prior distribution z_star, real_label, fake_label
        z_star = Variable(torch.FloatTensor(batch_size * n_z).uniform_(-1, 1).to(device)).view(batch_size, n_z)
        real_label = Variable(torch.ones(batch_size).fill_(1).to(device)).view(-1, 1)
        fake_label = Variable(torch.ones(batch_size).fill_(0).to(device)).view(-1, 1)

        ## train Encoder and Generator with reconstruction loss
        netE.zero_grad()
        netG.zero_grad()

        # EG_loss 1. L1 reconstruction loss
        z = netE(image)
        reconst = netG(z, label, gender)
        EG_L1_loss = L1loss(reconst, image)

        # EG_loss 2. GAN loss - image
        z = netE(image)
        reconst = netG(z, label, gender)
        D_reconst, _ = netDimg(reconst, label.view(batch_size, n_l, 1, 1), gender.view(batch_size, 1, 1, 1))
        G_img_loss = BCEloss(D_reconst, real_label)

        ## EG_loss 3. GAN loss - z
        Dz_prior = netDz(z_star)
        Dz = netDz(z)
        Ez_loss = BCEloss(Dz, real_label)

        ## EG_loss 4. TV loss - G
        reconst = netG(z.detach(), label, gender)
        G_tv_loss = total_variation_loss(reconst)

        EG_loss = EG_L1_loss + 0.0001 * G_img_loss + 0.01 * Ez_loss + G_tv_loss
        EG_loss.backward()

        optimizerE.step()
        optimizerG.step()

        ## train netDz with prior distribution U(-1,1)
        netDz.zero_grad()
        Dz_prior = netDz(z_star)
        Dz = netDz(z.detach())

        Dz_loss = BCEloss(Dz_prior, real_label) + BCEloss(Dz, fake_label)
        Dz_loss.backward()
        optimizerDz.step()

        ## train D_img with real images
        netDimg.zero_grad()
        D_img, D_clf = netDimg(image, label.view(batch_size, n_l, 1, 1), gender.view(batch_size, 1, 1, 1))
        D_reconst, _ = netDimg(reconst.detach(), label.view(batch_size, n_l, 1, 1), gender.view(batch_size, 1, 1, 1))

        D_loss = BCEloss(D_img, real_label) + BCEloss(D_reconst, fake_label)
        D_loss.backward()
        optimizerDimg.step()

    if epoch % 10 == 0:
        print("The current losses are: ", f"EG_L1_loss: {EG_L1_loss}", f"G_img_loss: {G_img_loss}",
              f"Ez_loss: {Ez_loss}", f"G_tv_loss: {G_tv_loss}", f"EG_loss: {EG_loss}", sep='\n')
        torch.save(netE.state_dict(), os.path.join(output_directory, f"netE_{epoch}.pickle"))
        torch.save(netG.state_dict(), os.path.join(output_directory, f"netG_{epoch}.pickle"))
        torch.save(netDz.state_dict(), os.path.join(output_directory, f"netDz_{epoch}.pickle"))
        torch.save(netDimg.state_dict(), os.path.join(output_directory, f"netDimg_{epoch}.pickle"))
