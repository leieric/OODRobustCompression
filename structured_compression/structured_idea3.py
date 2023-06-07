import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import numpy as np
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

from layers_compress import Quantizer, Generator, Encoder, AutoencoderQ, SingleEncMultiDec
from pytorchtools import EarlyStopping, GaussianNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(latent_dim, L, gamma, d_pair, train_loader, test_loader):

    d_list = np.cumsum(d_pair)
    
    img_size = (32, 32, 1)
    dec_list = [Generator(img_size=img_size, latent_dim=d, dim=64).to(device) for d in d_list]
    netE = Encoder(img_size, d_list[-1], dim=64).to(device)
    netQ = Quantizer(np.linspace(-1., 1., L))
    model = SingleEncMultiDec(netE, netQ, dec_list, d_list)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=12, verbose=True)
    
    # Number of training epochs
    num_epochs = 60

    # Setup Adam optimizers for both G and D
#     betas = (0.5, 0.9)
    lr=2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
#     schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=200)
#     schedulerE = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE, T_max=200)
    
    # define adversary
    adversary = WassDROAttack_AE(k=15, a = 1., gamma=gamma) 
    
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
        train_loss1 = train_loss2 = 0
        n0 = 0
        for idx, data in enumerate(train_loader):
            real_data = data[0].to(device)
            x = real_data
            ball = 0
            if np.random.uniform() < 0.5 and epoch > 10:
                # sample from ball
                ball = 1
                x = adv_train(real_data, None, model, adversary, ball).to(device) 
            x_hat = model(x, ball)
            loss = F.mse_loss(x, x_hat)*img_size[0]*img_size[1]
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ball==0:
                train_loss1 += loss.item()
                n0 += 1
            else:
                train_loss2 += loss.item()

        factor = 0 if n0 == len(train_loader) else 1/(len(train_loader)-n0)
        kbar.update(idx, values=[("Distor_stage1", train_loss1/n0), ("Distor_stage2", factor*train_loss2)])
        
        # Validation
        val_loss1 = val_loss2 = 0
        model.eval()
        for idx, data in enumerate(test_loader):
            real_data = data[0].to(device)
            x_hat1 = model(real_data, 0)
            x_hat2 = model(real_data, 1)
            val_loss1 +=  img_size[0]*img_size[1]*F.mse_loss(real_data, x_hat1).item()
            val_loss2 +=  img_size[0]*img_size[1]*F.mse_loss(real_data, x_hat2).item()
        kbar.add(1, values=[("val_stage1", val_loss1/len(test_loader)), ("val_stage2", val_loss2/len(test_loader))])
#         early_stopping(val_loss, None)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#         scheduler.step()
#         schedulerE.step()
#         schedulerG.step()
        

    # Save nets
    nets = {'netE':model.encoder, 'netQ':model.quantizer, 'dec_list':model.decoder_list,  'dim_list': model.dim_list, 'L': L, 'gamma': gamma}
#     save_name = '../trained_robust_WassDRO/ae_c_d{}L{}.pt'.format(latent_dim, L)
    save_name = f'../trained_structured_wass_ball/ae_c_d{d_pair[0]}-{d_pair[1]}L{L}gamma{gamma:.2f}.pt'
    torch.save(nets, save_name)
    
if __name__ == '__main__':
    # Load data

    transform_ood = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.RandomRotation((-70, 70)),
        torchvision.transforms.ToTensor(), 
#         GaussianNoise(0., 0.10)
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])
    batch_size = 64
    
#     mnist_ood_train = torchvision.datasets.MNIST('../datasets/', train=True, download=True, transform=transform_ood)
    mnist_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform)
    
#     trainset = torch.utils.data.ConcatDataset([mnist_ood_train, mnist_train])
    testset = torchvision.datasets.MNIST('../../data/', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
    

    L = 12
    gamma = 1
    dim_pairs = [(4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
#     dim_pairs = []
    for d_pair in dim_pairs:
        print(f"Training d_pair={d_pair[0]}-{d_pair[1]}")
        sys.stdout.flush()
        train(None, L, gamma, d_pair, train_loader, test_loader)
        
        
        
        
#     L = 12
#     for gamma in gammas:
#         print("Training gamma={}".format(str(gamma)))
#         sys.stdout.flush()
#         train(4, L, gamma, train_loader, test_loader)
