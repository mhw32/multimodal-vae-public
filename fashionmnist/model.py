from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# MAP from index to the interpretable label
LABEL_IX_TO_STRING = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 
                      4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 
                      9: 'Ankle boot'}


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(MVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.text_encoder  = TextEncoder(n_latents)
        self.text_decoder  = TextDecoder(n_latents)
        self.experts       = ProductOfExperts()
        self.n_latents     = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, image=None, text=None):
        mu, logvar = self.infer(image, text)
        # reparametrization trick to sample
        z          = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = self.image_decoder(z)
        txt_recon  = self.text_decoder(z)
        return img_recon, txt_recon, mu, logvar

    def infer(self, image=None, text=None): 
        batch_size = image.size(0) if image is not None else text.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)
        if image is not None:
            img_mu, img_logvar = self.image_encoder(image)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if text is not None:
            txt_mu, txt_logvar = self.text_encoder(text)
            mu     = torch.cat((mu, txt_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, txt_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.n_latents = n_latents
        self.upsampler = nn.Sequential(
            nn.Linear(n_latents, 512),
            Swish(),
            nn.Linear(512, 128 * 7 * 7),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsampler(z)
        z = z.view(-1, 128, 7, 7)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py


class TextEncoder(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(10, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 10))

    def forward(self, z):
        z = self.net(z)
        return z  # NOTE: no softmax here. See train.py


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
