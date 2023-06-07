import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import multiprocessing
from torchvision import transforms
import torch.utils
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad as torch_grad
import pkbar
import sys
from pytorchtools import GaussianNoise
from adversarialbox.attacks import L2PGDAttack_AE, LinfPGDAttack_AE, WassDROAttack_AE
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
# from vgg_ae import Encoder, Decoder

from layers_compress import Quantizer, Generator, Encoder, AutoencoderQ
from pytorchtools import EarlyStopping, GaussianNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(latent_dim, L,std, train_loader, test_loader):
    # Setup
    img_size = (32, 32, 1)
    netG = Generator(img_size=img_size, latent_dim=latent_dim, dim=64).to(device)
    netE = Encoder(img_size, latent_dim, dim=64).to(device)
    netQ = Quantizer(np.linspace(-1., 1., L))
    
    model = AutoencoderQ(netE, netG, netQ)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=12, verbose=True)
    
    # Number of training epochs
    num_epochs = 50

    # Setup Adam optimizers for both G and D
#     betas = (0.5, 0.9)
    lr=2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
#     schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=200)
#     schedulerE = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE, T_max=200)
    

    
    # Begin Training
    batch_cntr = 0
    t = torch.zeros(1)
    loss_dist = F.mse_loss(t, t)
    loss_G = F.mse_loss(t, t)
    for epoch in range(1, num_epochs+1):
        sys.stdout.flush()
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch-1, num_epochs=num_epochs, width=8, verbose=2, always_stateful=False)
        if (epoch % 20 == 0):
            lr *= 0.2 
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Training
        model.train()
        train_loss = 0
        for idx, data in enumerate(train_loader):
            real_data = data[0].to(device)
            # adversarial training
            if epoch > 0:
                loss = F.mse_loss(real_data, model(real_data))*img_size[0]*img_size[1]
                train_loss += loss.item()
            else:
                pass
#                 theta = argmax_theta(train_loader, model)
#                 x_adv = torchvision.transforms.functional.rotate(x, theta)
#                 loss_adv = F.mse_loss(x_adv, model(x_adv))*img_size[0]*img_size[1]
#                 loss = loss_adv
#                 train_loss += loss.item()
#                 loss = (loss + loss_adv) / 2
#             batch_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

        kbar.update(idx, values=[("Distor_loss", train_loss/len(train_loader))])
        
        # Validation
        val_losses = []
        model.eval()
        for idx, data in enumerate(test_loader):
            real_data = data[0].to(device)
            fake_data = model(real_data)
            loss_dist = F.mse_loss(real_data, fake_data)*img_size[0]*img_size[1]
            val_losses.append(loss_dist.item())
        val_loss = np.mean(val_losses)
        kbar.add(1, values=[("val_loss", val_loss)])
        early_stopping(val_loss, netE)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#         scheduler.step()
#         schedulerE.step()
#         schedulerG.step()
        

    # Save nets
    nets = {'netE':model.encoder, 'netQ':model.quantizer, 'netG':model.decoder,  'latent_dim': latent_dim}
#     save_name = '../trained_no_robust2/ae_c_d{}L{}.pt'.format(latent_dim, L)
    save_name = f'../trained_robust_augmentation/ae_c_d{latent_dim}L{L}std{std:.2f}.pt'
    torch.save(nets, save_name)
    
if __name__ == '__main__':
    # Load data
    

    L =12
    latent_dim = 10
    rhos = [20.1, 50]
    for rho in rhos:
        std = np.sqrt(rho/1024)
        
        # Create augmented dataset------------------------
        transform_ood = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
    #         torchvision.transforms.RandomCrop(32, padding=4),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.RandomRotation((-70, 70)),
            torchvision.transforms.ToTensor(), 
            GaussianNoise(0., std=std) # clamps to [0,1]
        ])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        batch_size = 64

        mnist_ood_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform_ood)
        mnist_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform)
        trainset = torch.utils.data.ConcatDataset([mnist_ood_train, mnist_train])
        testset = torchvision.datasets.MNIST('../../data/', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
        
        
        print("Training std={}".format(str(std)))
        sys.stdout.flush()
        train(latent_dim, L, std, train_loader, test_loader)
        

