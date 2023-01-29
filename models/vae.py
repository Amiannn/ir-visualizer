import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn

from .encoders import Encoder
from .decoders import Decoder
from .utils.template import ResData


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, n_class, use_vae=True):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims, n_class)
        self.use_vae = use_vae

    def forward(self, x):
        enc_out = self.encoder(x)
        z = enc_out.latent if self.use_vae else enc_out.mean
        dec_out = self.decoder(z)
        dec_out.enc_out = enc_out
        return dec_out 