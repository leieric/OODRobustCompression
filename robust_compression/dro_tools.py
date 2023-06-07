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

def est_radius(model, gamma, loader):
    rho_hat = 0
    adversary = WassDROAttack_AE(model=model, k=15, a=1, gamma=gamma)
    for data in loader:
        data = data[0].to(device)
        X_adv = adversary.perturb(data, None)
#         rho_hat += 32*32*F.mse_loss(X_adv, data).item()
        incr = torch.mean(torch.norm(data-X_adv, dim=(2,3))**2).item()
        rho_hat += incr
#         print(incr)
    return rho_hat/len(loader)

def est_radius_batch(model, gamma, data):
    adversary = WassDROAttack_AE(model=model, k=15, a=1, gamma=gamma)
    data = data[0].to(device)
    X_adv = adversary.perturb(data, None)
    return torch.mean(torch.norm(data-X_adv, dim=(2,3))**2).item()

    


