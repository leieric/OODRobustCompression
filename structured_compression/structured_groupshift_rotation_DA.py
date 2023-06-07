import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import numpy as np
import scipy.optimize
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

from layers_compress import Quantizer, Generator, Encoder,Encoder_MaxPool, AutoencoderQ, SingleEncMultiDec
from pytorchtools import EarlyStopping, GaussianNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def obj_rot(theta, img_iterator, model):
    x = next(img_iterator)[0].to(device)
    x_rotated = torchvision.transforms.functional.rotate(x, theta)
    x_hat = model(x_rotated, 1)
    loss = -32*32*F.mse_loss(x_rotated, x_hat).item()
    return loss
    
def argmax_theta(train_loader, model):
    img_iterator = iter(train_loader) #shuffles the train_loader
    obj = lambda theta: obj_rot(theta, img_iterator, model)
    theta_max = scipy.optimize.minimize_scalar(obj, bounds=[-180, 180], method='bounded').x
    return theta_max


def train(latent_dim, L, gamma, d_pair, train_loader, train_ood_loader, test_loader):

    d_list = np.cumsum(d_pair)
    
    img_size = (32, 32, 1)
    dec_list = [Generator(img_size=img_size, latent_dim=d, dim=64).to(device) for d in d_list]
    netE = Encoder(img_size, d_list[-1], dim=64).to(device)
    netQ = Quantizer(np.linspace(-1., 1., L))
    model = SingleEncMultiDec(netE, netQ, dec_list, d_list)
    
    #load model
#     saved = torch.load(f'../trained_rotations/ae_c_d{d_pair[0]}-{d_pair[1]}L{L}rotated_DA.pt')
#     model = SingleEncMultiDec(saved['netE'], saved['netQ'], saved['dec_list'], saved['dim_list'])
    
    
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
        train_loss = train_loss1 = train_loss2 = 0
        n0 = 0
        theta_save = 0
        for idx, (data, data_ood) in enumerate(zip(train_loader, train_ood_loader)):
            x = data[0].to(device)
            x_rot = data_ood[0].to(device)
            
            loss1 = img_size[0]*img_size[1]*F.mse_loss(x, model(x, 0))
            train_loss1 += loss1.item()
            loss2 = img_size[0]*img_size[1]*F.mse_loss(x_rot, model(x_rot, 1))
            train_loss2 += loss2.item()
            
            loss = 0.5*loss1+loss2
            train_loss += loss.item()
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            

        kbar.update(idx, values=[("Distor_stage1", train_loss1/len(train_loader)), ("Distor_stage2", train_loss2/len(train_loader)), ("Total_Distor", train_loss/len(train_loader))])
        
        # Validation
        val_loss1 = val_loss2 = 0
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                real_data = data[0].to(device)
                x_hat1 = model(real_data, 0)
                x_hat2 = model(real_data, 1)
                val_loss1 +=  img_size[0]*img_size[1]*F.mse_loss(real_data, x_hat1).item()
                val_loss2 +=  img_size[0]*img_size[1]*F.mse_loss(real_data, x_hat2).item()
            kbar.add(1, values=[("val_stage1", val_loss1/len(test_loader)), ("val_stage2", val_loss2/len(test_loader))])
        

    # Save nets
    nets = {'netE':model.encoder, 'netQ':model.quantizer, 'dec_list':model.decoder_list,  'dim_list': model.dim_list, 'L': L, 'gamma': gamma}
#     save_name = '../trained_robust_WassDRO/ae_c_d{}L{}.pt'.format(latent_dim, L)
    save_name = f'../trained_rotations/ae_c_d{d_pair[0]}-{d_pair[1]}L{L}rotated_DA.pt'
    torch.save(nets, save_name)
    
if __name__ == '__main__':
    # Load data

    # Create augmented dataset------------------------
    transform_ood = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation((-180, 180)),
        torchvision.transforms.ToTensor(), 
#             GaussianNoise(0., std=std) # clamps to [0,1]
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])
    batch_size = 64

    mnist_ood_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform_ood)
    mnist_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform)
    combined = torch.utils.data.ConcatDataset([mnist_ood_train, mnist_train])
    testset = torchvision.datasets.MNIST('../../data/', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                                shuffle=True)
    train_ood_loader = torch.utils.data.DataLoader(mnist_ood_train, batch_size=batch_size,
                                                shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                     shuffle=False)
    

    L = 12
    gamma = 1
#     dim_pairs = [(4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
    dim_pairs = [(7,3), (6, 4), (8, 2), (4, 6)]
#     dim_pairs = []
    for d_pair in dim_pairs:
        print(f"Training d_pair={d_pair[0]}-{d_pair[1]}")
        sys.stdout.flush()
        train(None, L, gamma, d_pair, train_loader, train_ood_loader, test_loader)
        
        
        
        
#     L = 12
#     for gamma in gammas:
#         print("Training gamma={}".format(str(gamma)))
#         sys.stdout.flush()
#         train(4, L, gamma, train_loader, test_loader)
