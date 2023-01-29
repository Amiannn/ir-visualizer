import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn

criterion = nn.CrossEntropyLoss()

def kl_loss(mu, sigma):
    loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return loss

def reconstruction_loss(y, label):
    loss = criterion(y, label)
    return loss

def variational_loss(y, label, mu, sigma, beta=0.9999999):
    loss = beta * reconstruction_loss(y, label) + (1 - beta) * kl_loss(mu, sigma)
    return loss