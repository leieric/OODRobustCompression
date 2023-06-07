import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pytorchtools import CNN, GaussianNoise
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack_AE, WassDROAttack_AE, L2PGDAttack_AE
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from layers_compress import Encoder, Generator, Quantizer, AutoencoderQ
import os
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                            torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST('../../data/', download=True, train=True, transform=transform)                                 
mnist_test = torchvision.datasets.MNIST('../../data/', download=True, train=False, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=12000, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=10000, shuffle=True)


L = 12
gamma_train = 500
# saved = torch.load(f'../trained_robust_WassDRO/ae_c_d4L{L}.pt')
saved = torch.load(f'../trained_gamma_sweep/ae_c_d4L{L}gamma{gamma_train}.pt')
model = AutoencoderQ(saved['netE'], saved['netG'], saved['netQ'])

def worstcase_distortion(model, gamma, loader):
    E = 0
    rho_hat = 0
    for data in loader:
        data = data[0].to(device)
        adversary = WassDROAttack_AE(model=model, k=20, a=1., gamma=gamma, loss_fn=nn.MSELoss(), transport_cost=nn.MSELoss())
        X_adv = adversary.perturb(data, None)
        rho_hat += F.mse_loss(X_adv, data).item()*32*32
        E += torch.mean(torch.norm(X_adv - model(X_adv), dim=(2,3))**2 - gamma*torch.norm(data-X_adv, dim=(2,3))**2).item()
    E = E/len(loader)
    rho_hat = rho_hat/len(loader)
    print(rho_hat)
    return rho_hat, (gamma*rho_hat + E)

def calc_E(model, gamma, loader):
    E = 0
    for data in loader:
        data = data[0].to(device)
        adversary = WassDROAttack_AE(model=model, a=1., gamma=gamma, k=20, loss_fn=nn.MSELoss(), transport_cost=nn.MSELoss())
        X_adv = adversary.perturb(data, None)
        E += torch.mean(torch.norm(X_adv - model(X_adv), dim=(2,3))**2 - gamma*torch.norm(data-X_adv, dim=(2,3))**2).item()
    E = E/len(loader)
    return E

def certificate(rho, gamma, E):
    return gamma*rho + E



# certificate
rhos = np.linspace(0., 18, 19)
E = calc_E(model, gamma_train, train_loader)
cert = []
for rho in rhos:
    print(f"rho: {rho}")
    sys.stdout.flush()
    cert += [certificate(rho, gamma_train, E)]

# worst case distortion
t = np.linspace(0., 30, 31)
gammas = 20000*0.75**t
wcd = []

for gamma in gammas:
    print(f"gamma: {gamma}")
    sys.stdout.flush()
    wcd += [worstcase_distortion(model, gamma, train_loader)]

savefile = {'wcd': wcd, 'cert': cert}
torch.save(savefile, f'../wcd_cert/wcd_cert_gamma{gamma_train}.pt')