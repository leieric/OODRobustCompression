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
import sklearn.metrics
import sys
from pytorchtools import GaussianNoise
from adversarialbox.attacks import L2PGDAttack_AE, LinfPGDAttack_AE, WassDROAttack_AE
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
# from vgg_ae import Encoder, Decoder

from layers_compress import Quantizer, Generator, Encoder,Encoder_MaxPool, AutoencoderQ, SingleEncMultiDec,StructuredRotation, rot_img, AnglePred
from pytorchtools import EarlyStopping, GaussianNoise, RotateAngles

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def argmax_theta(batch, model):
#     img_iterator = iter(train_loader) #shuffles the train_loader
#     obj_shgo = lambda theta: obj_rot(theta, batch, model)
#     obj_shgo = lambda theta: -obj_rot(float(theta[0]), batch, model)
#     theta_max = scipy.optimize.shgo(obj_shgo, bounds=[(-180, 180)], sampling_method='sobol', n=8, minimizer_kwargs={'method': 'SLSQP'}, options={'maxiter':40}).x[0]
    theta_max = scipy.optimize.differential_evolution(obj_scipy, bounds=[(-180, 180)], args=(batch, model), init='random', popsize=8).x[0]
#     theta_max = scipy.optimize.minimize_scalar(obj, bounds=[-180, 180], method='bounded').x
    return theta_max

def approx_accuracy(preds, angles):
  # checks if predicted angles are within 5 degrees of the true angle
  preds_idx = torch.argmax(preds, dim=1)
  return torch.logical_and((preds_idx <= angles + 5), (preds_idx >= angles - 5)).float().mean().item()
  
def approx_accuracy_regress(preds, angles):
  # checks if predicted angles are within 5 degrees of the true angle
  return torch.logical_and((preds <= angles + 5), (preds >= angles - 5)).float().mean().item()
  

def train(d_angle, L, d, train_loader, test_loader):
    print(f"unrotated d={d-d_angle}, angle pred d={d_angle}")
    img_size = (32, 32, 1)
    netG = Generator(img_size=img_size, latent_dim=d-d_angle, dim=64).to(device)
    netE = Encoder(img_size, d-d_angle, dim=64).to(device)
    netQ = Quantizer(np.linspace(-1., 1., L))
    netAngle = Encoder_MaxPool(img_size, d_angle, dim=64).to(device)
    # netAngle = AnglePred().to(device)
    
    # load trained unrotated and angle pred models
    # saved = torch.load(f'../trained_no_robust/ae_c_d{d-d_angle}L{L}.pt')
    # netE, netQ, netG = saved['netE'], saved['netQ'], saved['netG']
    # saved = torch.load('../trained_angle_pred/angle_net.pt')
    # netAngle = saved['angle_net']
    
    model = StructuredRotation(netE, netG, netQ, netAngle, L)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=12, verbose=True)
    
    # Number of training epochs
    warmup = 50
    num_epochs = 1+warmup
    

    lr=2e-4 
    lr_angle = 5e-4 #* 0.2**5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_angle_net = torch.optim.Adam(model.angle_net.parameters(), lr=lr_angle)
    scheduler_angle_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_angle_net, T_max=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # Begin Training
    batch_cntr = 0
    t = torch.zeros(1)
    loss_dist = F.mse_loss(t, t)
    loss_G = F.mse_loss(t, t)
    for epoch in range(1, num_epochs+1):
        if epoch == warmup+1:
          nets = {'angle_net':model.angle_net}
          save_name = f'../trained_angle_pred/angle_net_small.pt'
          torch.save(nets, save_name)
        sys.stdout.flush()
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch-1, num_epochs=num_epochs, width=8, verbose=2, always_stateful=False)
        if (epoch % 10 == 0):
            lr *= 0.2 
            lr_angle *=0.2
            optimizer_angle_net = torch.optim.Adam(model.angle_net.parameters(), lr=lr_angle)
            if (epoch > warmup):
              optimizer = torch.optim.Adam(model.parameters(), lr=lr)
              scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Training
        model.train()
        train_loss = angle_loss = distort_loss = loss_distort = 0
        theta_save = angle_pred_save = 0
        acc = 0
        for idx, data in enumerate(train_loader):
            # optimizer.zero_grad()
            x = data[0].to(device)
            if epoch <= warmup:
              # train the angle net
              # angles = torch.randint(high=360, size=(x.shape[0],)).to(device)
              angles = 360*torch.rand((x.shape[0],1)).to(device)
              x_rot = rot_img(x, (angles*np.pi/180)-np.pi, torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).to(device)
              # print(model.angle_net(x_rot).shape)
              # loss_angle = F.mse_loss(np.pi*model.angle_net(x_rot), angles)
              preds = model.angle_net(x_rot)
              # preds_idx = torch.argmax(preds, dim=1)
              # acc += sklearn.metrics.accuracy_score(angles.cpu().detach().numpy(), preds_idx.cpu().detach().numpy())
              acc += approx_accuracy_regress(preds, angles)
              # acc += (angles == preds_idx).float().mean().item()

              # loss_angle = F.cross_entropy(preds, angles)
              loss_angle = F.mse_loss(preds, angles)
              model.angle_net.zero_grad()
              loss_angle.backward()
              optimizer_angle_net.step()
            else: 
              model.eval()
              with torch.no_grad():
                  theta = argmax_theta(x, model)
              model.train()
              theta_save = theta
              x = torchvision.transforms.functional.rotate(x, theta)
              x_hat = model(x, train_angle_net=True)
              loss_distort = F.mse_loss(x, x_hat)*img_size[0]*img_size[1]
              
              # angle_pred = np.pi*model.angle_net(x)
              # angle_pred_save = angle_pred.mean().item()*180/np.pi
              # loss_angle = F.mse_loss((theta*np.pi/180)*torch.ones((x.shape[0], 1)).to(device), angle_pred)
              # loss = loss_angle + (epoch > warmup)*loss_distort
              loss = loss_distort
              model.zero_grad()
              loss.backward()
              optimizer.step()
          
            angle_loss += loss_angle.item()
            if (epoch > warmup):
              train_loss += loss.item()
              distort_loss += loss_distort.item()
        # scheduler_angle_net.step()
        kbar.update(idx, values=[("Total loss", train_loss/len(train_loader)), 
                    ("Angle loss", angle_loss/len(train_loader)), 
                    ("Angle acc", acc/len(train_loader)),
                    ("Distort loss", distort_loss/len(train_loader)),
                    ("theta", theta_save),
                    ("angle_pred", angle_pred_save)
                    ])
        
        # Validation
        val_loss1 = 0
        if epoch > warmup:
          model.eval()
          with torch.no_grad():
              for idx, data in enumerate(test_loader):
                  real_data = data[0].to(device)
                  x_hat1 = model(real_data)
                  val_loss1 +=  img_size[0]*img_size[1]*F.mse_loss(real_data, x_hat1).item()
        kbar.add(1, values=[("val_loss", val_loss1/len(test_loader))])
        

    # Save nets
    nets = {'netE':model.encoder, 'netQ':model.quantizer, 'netG':model.decoder,  'netAngle': model.angle_net, 'angleQ':model.angle_quantizer, 'L': L}
#     save_name = '../trained_robust_WassDRO/ae_c_d{}L{}.pt'.format(latent_dim, L)
    save_name = f'../trained_angle_pred/ae_c_d{d}-{d_angle}L{L}rotated_angle_pred.pt'
    torch.save(nets, save_name)
    
if __name__ == '__main__':
    # Load data
    
    k = int(sys.argv[1])

    transform_ood = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.RandomRotation((-90, 90)),
        RotateAngles(np.linspace(-180, 180, k)),
        torchvision.transforms.ToTensor(), 
#         GaussianNoise(0., 0.10)
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])
    batch_size = 128
    
    mnist_ood_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform_ood)
    mnist_train = torchvision.datasets.MNIST('../../data/', train=True, download=True, transform=transform)
    
#     trainset = torch.utils.data.ConcatDataset([mnist_ood_train, mnist_train])
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

#     dim_pairs = []
    print("Training")
    sys.stdout.flush()
    train(k, L, 10, train_loader, test_loader)
        
        
        
        
#     L = 12
#     for gamma in gammas:
#         print("Training gamma={}".format(str(gamma)))
#         sys.stdout.flush()
#         train(4, L, gamma, train_loader, test_loader)
