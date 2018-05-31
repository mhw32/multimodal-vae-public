from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from datasets import N_ATTRS


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(MVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.attrs_encoder = AttributeEncoder(n_latents)
        self.attrs_decoder = AttributeDecoder(n_latents)
        self.experts       = ProductOfExperts()
        self.n_latents     = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, image=None, attrs=None):
        mu, logvar  = self.infer(image, attrs)
        # reparametrization trick to sample
        z           = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        image_recon = self.image_decoder(z)
        attrs_recon = self.attrs_decoder(z)
        return image_recon, attrs_recon, mu, logvar

    def infer(self, image=None, attrs=None): 
        batch_size = image.size(0) if image is not None else attrs.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)
        if image is not None:
            image_mu, image_logvar = self.image_encoder(image)
            mu     = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if attrs is not None:
            attrs_mu, attrs_logvar = self.attrs_encoder(attrs)
            mu     = torch.cat((mu, attrs_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, attrs_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    This is the standard DCGAN architecture.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z). 

    This is the standard DCGAN architecture.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 5 * 5),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py


class AttributeEncoder(nn.Module):
    """Parametrizes q(z|y). 

    We use a single inference network that encodes 
    all 18 features.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(AttributeEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_ATTRS, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]


class AttributeDecoder(nn.Module):
    """Parametrizes p(y|z).

    We use a single generative network that decodes 
    all 18 features.

    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(AttributeDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, N_ATTRS))

    def forward(self, z):
        z = self.net(z)
        # not a one-hotted prediction: this returns a value
        # for every single index
        return z  # NOTE: no sigmoid here. See train.py


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
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
