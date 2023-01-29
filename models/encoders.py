import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn

from .utils.template import ResData

class Encoder(nn.Module):
    def __init__(self, latent_dims):  
        super(Encoder, self).__init__()
        self.encoder_lin = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )
        self.linear1 = nn.Linear(10, latent_dims)
        self.linear2 = nn.Linear(10, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x     = self.encoder_lin(x)
        mu    =  self.linear1(x)
        sigma = torch.exp(self.linear2(x))
        z     = mu + sigma * self.N.sample(mu.shape)
        
        return ResData(
            logits=x,
            mean=mu,
            sigma=sigma,
            latent=z
        )