from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class MVAE(nn.Module):
    def __init__(self, n_latents=250, use_cuda=False):
        super(MVAE, self).__init__()
        # define q(z|x_i) for i = 1...6
        self.image_encoder     = ImageEncoder(n_latents, 3)
        self.gray_encoder      = ImageEncoder(n_latents, 1)
        self.edge_encoder      = ImageEncoder(n_latents, 1)
        self.mask_encoder      = ImageEncoder(n_latents, 1)
        self.obscured_encoder  = ImageEncoder(n_latents, 3)
        self.watermark_encoder = ImageEncoder(n_latents, 3)
        # define p(x_i|z) for i = 1...6
        self.image_decoder     = ImageDecoder(n_latents, 3)
        self.gray_decoder      = ImageDecoder(n_latents, 1)
        self.edge_decoder      = ImageDecoder(n_latents, 1)
        self.mask_decoder      = ImageDecoder(n_latents, 1)
        self.obscured_decoder  = ImageDecoder(n_latents, 3)
        self.watermark_decoder = ImageDecoder(n_latents, 3)
        # define q(z|x) = q(z|x_1)...q(z|x_6)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents
        self.use_cuda = use_cuda`

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, image=None, gray=None, edge=None, mask=None, 
                obscured=None, watermark=None):
        mu, logvar = self.get_params(image=image, gray=gray, edge=edge, mask=mask,
                                     obscured=obscured, watermark=watermark)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on sample
        image_recon     = self.image_decoder(z)
        gray_recon      = self.gray_decoder(z)
        edge_recon      = self.edge_decoder(z)
        mask_recon      = self.mask_decoder(z)
        obscured_recon  = self.obscured_decoder(z)
        watermark_recon = self.watermark_decoder(z)

        return (image_recon, gray_recon, edge_recon, mask_recon, 
                rotated_recon, obscured_recon, mu, logvar)

    def get_params(self, image=None, gray=None, edge=None, 
                   mask=None, obscured=None, watermark=None):
        # define universal expert
        batch_size = get_batch_size(image, gray, edge, mask, obscured, watermark)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)

        if image is not None:
            image_mu, image_logvar = self.image_encoder(image)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if gray is not None:
            gray_mu, gray_logvar = self.gray_encoder(gray)
            mu = torch.cat((mu, gray_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, gray_logvar.unsqueeze(0)), dim=0)
        
        if edge is not None:
            edge_mu, edge_logvar = self.edge_encoder(edge)
            mu = torch.cat((mu, edge_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, edge_logvar.unsqueeze(0)), dim=0)

        if mask is not None:
            mask_mu, mask_logvar = self.mask_encoder(mask)
            mu = torch.cat((mu, mask_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, mask_logvar.unsqueeze(0)), dim=0)

        if obscured is not None:
            obscured_mu, obscured_logvar = self.obscured_encoder(obscured)
            mu = torch.cat((mu, obscured_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, obscured_logvar.unsqueeze(0)), dim=0)

        if watermark is not None:
            watermark_mu, watermark_logvar = self.watermark_encoder(watermark)
            mu = torch.cat((mu, watermark_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, watermark_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


def get_batch_size(*args):
    for arg in args:
        if arg is not None:
            return arg.size(0)


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    We will use this for every q(z|x_i) for all i.

    @param n_latents: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """
    def __init__(self, n_latents, n_channels=3):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 4, 2, 1, bias=False),
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

    We will use this for every p(x_i|z) for all i.

    @param n_latents: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """
    def __init__(self, n_latents, n_channels=3):
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
            nn.ConvTranspose2d(32, n_channels, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        return z  # no sigmoid!


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
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
