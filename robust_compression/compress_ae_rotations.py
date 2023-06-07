import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import scipy.optimize
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
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
# device = 'cpu'

def loopy(dl):
    while True:
        for x in iter(dl): yield x

def obj_rot(theta, batch, model):
#     x = next(img_iterator)[0].to(device)
    x = batch
#     print(theta, type(theta))
    x_rotated = torchvision.transforms.functional.rotate(x, theta)
    x_hat = model(x_rotated)
    loss = 32*32*F.mse_loss(x_rotated, x_hat).item()
    return loss

def obj_scipy(theta, *args):
    return -obj_rot(float(theta[0]), args[0], args[1])

def argmax_theta_random(batch, model, n_samp):
    x = batch
    angles = np.random.uniform(low=-180, high=180, size=n_samp)
    losses = [obj_rot(angle, batch, model) for angle in angles]
    return angles[np.argmax(losses)]
    
def argmax_theta(batch, model):
#     img_iterator = iter(train_loader) #shuffles the train_loader
#     obj_shgo = lambda theta: obj_rot(theta, batch, model)
#     obj_shgo = lambda theta: -obj_rot(float(theta[0]), batch, model)
#     theta_max = scipy.optimize.shgo(obj_shgo, bounds=[(-180, 180)], sampling_method='sobol', n=8, minimizer_kwargs={'method': 'SLSQP'}, options={'maxiter':40}).x[0]
    theta_max = scipy.optimize.differential_evolution(obj_scipy, bounds=[(-180, 180)], args=(batch, model), init='random', popsize=8, workers=1).x[0]
#     theta_max = scipy.optimize.minimize_scalar(obj, bounds=[-180, 180], method='bounded').x
    return theta_max

def gen_save_plot(batch, model, epoch, theta_max):
    thetas = np.linspace(-179, 180, 360)
    d_dro = np.zeros(len(thetas))
#     iterator = loopy(train_loader)
    for i,theta in enumerate(thetas):
        d_dro[i] = obj_rot(theta, batch, model)
    plt.figure()
    plt.plot(thetas, d_dro)
    plt.axvline(theta_max, color='k',linestyle='--')
    plt.xlabel('angle')
    plt.ylabel('distortion')
    plt.savefig(f'../plots_rotations2/fixed_epoch{epoch}.png')

def train(latent_dim, L, gamma, train_loader, test_loader):
    # Setup
    img_size = (32, 32, 1)
    netG = Generator(img_size=img_size, latent_dim=latent_dim, dim=64).to(device)
    netE = Encoder(img_size, latent_dim, dim=64).to(device)
    netQ = Quantizer(np.linspace(-1., 1., L))
    model = AutoencoderQ(netE, netG, netQ)
    
#     # load model
#     saved = torch.load(f'../trained_robust_rotations/ae_c_d{latent_dim}L{L}rotated_fixed.pt')
#     model = AutoencoderQ(saved['netE'], saved['netG'], saved['netQ'])
    
    
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
#     adversary = LinfPGDAttack_AE(k=10, loss_fn=nn.MSELoss())
#     adversary = WassDROAttack_AE(k=15, a = 1., gamma=gamma)
#     adversary = L2PGDAttack_AE(k=15, epsilon=4.15, a = 1., loss_fn=nn.MSELoss())
    
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
        train_loss = 0
        for idx, data in enumerate(train_loader):
            model.train()
            real_data = data[0].to(device)
            x = real_data
            theta_save = 0
            # adversarial training
            if epoch < 0:
                loss = F.mse_loss(real_data, model(real_data))*img_size[0]*img_size[1]
                train_loss += loss.item()
            else:
                model.eval()
                with torch.no_grad():
                    theta = argmax_theta(x, model)
#                 print(idx, theta)
#                 sys.stdout.flush()
                model.train()
                theta_save = theta
                x_adv = torchvision.transforms.functional.rotate(x, theta)
                loss_adv = F.mse_loss(x_adv, model(x_adv))*img_size[0]*img_size[1]
                loss = loss_adv
                train_loss += loss.item()
#                 loss = (loss + loss_adv) / 2
#             batch_loss += loss.item()
            if idx == len(train_loader)-1:
                model.eval()
                with torch.no_grad():
                    gen_save_plot(x, model, epoch, theta_save)
                model.train()
                
            model.zero_grad()
            loss.backward()
            optimizer.step()
            


        kbar.update(idx, values=[("Distor_loss", train_loss/len(train_loader)), ("theta", theta_save)])
        
        # Validation
        val_losses = []
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                real_data = data[0].to(device)
                fake_data = model(real_data)
                loss_dist = F.mse_loss(real_data, fake_data)*img_size[0]*img_size[1]
                val_losses.append(loss_dist.item())
        val_loss = np.mean(val_losses)
        kbar.add(1, values=[("val_loss", val_loss)])
        

    # Save nets
    nets = {'netE':model.encoder, 'netQ':model.quantizer, 'netG':model.decoder,  'latent_dim': latent_dim}
#     save_name = '../trained_no_robust2/ae_c_d{}L{}.pt'.format(latent_dim, L)
    save_name = f'../trained_robust_rotations2/ae_c_d{latent_dim}L{L}rotated.pt'
    torch.save(nets, save_name)
    
if __name__ == '__main__':
#     mp.set_start_method('spawn')
    # Load data

    transform_ood = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(), 
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])
    batch_size = 128
    
#     mnist_ood_train = torchvision.datasets.MNIST('../datasets/', train=True, download=True, transform=transform_ood)
    mnist_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform)
    
#     trainset = torch.utils.data.ConcatDataset([mnist_ood_train, mnist_train])
    testset = torchvision.datasets.MNIST('../../data/', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
    
#     L_vals = [4**i for i in range(1, 8)]
#     L_vals=[4, 8, 12, 16, 32]
#     L_vals = 
#     latent_dim = 8
#     gammas = [500, 100, 20, 4]
#     gamma = 0.36
    gamma=1
    L =12
#     latent_dims = [10, 9]
    latent_dims = [10]
    for latent_dim in latent_dims:
        print("Training d={}".format(str(latent_dim)))
        sys.stdout.flush()
        train(latent_dim, L, gamma, train_loader, test_loader)
        
        
#     L = 12
#     for gamma in gammas:
#         print("Training gamma={}".format(str(gamma)))
#         sys.stdout.flush()
#         train(4, L, gamma, train_loader, test_loader)
