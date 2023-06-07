import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
import torch.nn.functional as F

from adversarialbox.utils import to_var

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- White-box attacks ---

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X
    
class LinfPGDAttack_AE(object):
    """ For autoencoder (unsupervised) models. perturb() function ignores the y input. Also is a torch implementation.
    """
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True, loss_fn=nn.MSELoss()):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = loss_fn

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
#             X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
#                 X_nat.shape).astype('float32')
            X = X_nat + 2*self.epsilon*torch.rand(X_nat.shape, device=device) - self.epsilon
        else:
#             X = np.copy(X_nat)
            X = torch.clone(X_nat)

        for i in range(self.k):
            X_var = to_var(X, requires_grad=True)

            scores = self.model(X_var)
            loss = self.loss_fn(scores, X_var)
            loss.backward()
#             grad = X_var.grad.data.cpu().numpy()
            grad = X_var.grad.data

            X += self.a * torch.sign(grad)
            
            # need elementwise clip since torch.clip only takes scalar min/max values
#             X = torch.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = torch.maximum(torch.minimum(X, X_nat + self.epsilon), X_nat - self.epsilon)
            X = torch.clip(X, 0, 1) # ensure valid pixel range

        return X
    
class L2PGDAttack_AE(object):
    """ For autoencoder (unsupervised) models. perturb() function ignores the y input. Also is a torch implementation.
    """
    def __init__(self, model=None, epsilon=4.15, k=40, a=1., 
        random_start=True, loss_fn=nn.MSELoss()):

        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = loss_fn

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
#             X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
#                 X_nat.shape).astype('float32')
            X = X_nat + 2*self.epsilon*torch.rand(X_nat.shape, device=device) - self.epsilon
        else:
#             X = np.copy(X_nat)
            X = torch.clone(X_nat)

        for i in range(1, self.k+1):
            X_var = to_var(X, requires_grad=True)

            scores = self.model(X_var)
            loss = self.loss_fn(scores, X_var)
            loss.backward()
#             grad = X_var.grad.data.cpu().numpy()
            grad = X_var.grad.data
            X += (self.a/(i**0.5)) * grad
            delta = X - X_nat
        
            # project onto l2 ball
            norm = torch.norm(delta, dim=(2,3)).unsqueeze(2).unsqueeze(2)
            scale = torch.minimum(norm, torch.from_numpy(np.array(self.epsilon)).to(device))
            delta = torch.div(delta, norm)*scale
#             if (i==self.k):
#                 print(delta.shape)
            
            X = torch.clip(X_nat+delta, 0, 1) # ensure valid pixel range

        return X
    
def batch_distortion(x1, x2):
#     norms = torch.norm(x1-x2, dim=(2,3))**2
#     return torch.mean(norms)
    return F.mse_loss(x1, x2, reduction='sum')
    
class WassDROAttack_AE(object):
    """ For autoencoder (unsupervised) models. perturb() function ignores the y input. Also is a torch implementation.
    """
    def __init__(self, model=None, gamma=0.04*9.21, epsilon=0.3, k=40, a=1, 
        random_start=False, loss_fn=nn.MSELoss(), transport_cost=nn.MSELoss()):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon # 
        self.gamma = gamma
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = batch_distortion
        self.cost = batch_distortion

    def perturb(self, X_nat, y, *args):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        scale = 32*32
#         if self.rand:
#             X = X_nat + 2*self.epsilon*torch.rand(X_nat.shape, device=device) - self.epsilon
#         else:
# #             X = np.copy(X_nat)
        X = X_nat.detach().clone()

        for i in range(1, self.k+1):
            X_var = to_var(X, requires_grad=True)
#             print(self.loss_fn(X_var, self.model(X_var, *args)).item(), self.cost(X_var, X_nat).item())
            loss = self.loss_fn(X_var, self.model(X_var, *args)) - self.gamma*self.cost(X_var, X_nat)
#             print(loss.item())
            loss.backward()
            grad = X_var.grad.data
#             print(grad)
            X += (self.a/(i**0.5)) * grad
            X = torch.clip(X, 0, 1) # ensure valid pixel range


        return X


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True, loss_fn=nn.CrossEntropyLoss()):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = loss_fn

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X


# --- Black-box attacks ---

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[len(X_sub_prev)+ind] = X_sub[ind] + lmbda * grad_val #???

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub
