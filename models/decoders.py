import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn

from .utils.template import ResData

class Decoder(nn.Module):
    def __init__(self, latent_dims, n_class):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )
        self.linear1 = nn.Linear(10, n_class)
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.linear1(x)

        return ResData(
            logits=x,
            mean=x
        )