from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils.models import ProductOfExperts
from .utils import log_joint_estimate, log_marginal_estimate


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder via a product of experts.
    This is hardcoded to the MNIST design.

    We optimize a lower bound on the following:

        log p(x) + log p(y) + log p(x,y)

    @param z_dim: integer
                  number of latent dimensions.
    """
    def __init__(self, z_dim):
        super(MVAE, self).__init__()
        self.z_dim = z_dim
        self.image_encoder = ImageEncoder(self.z_dim)
        self.image_decoder = ImageDecoder(self.z_dim)
        self.label_encoder = LabelEncoder(self.z_dim)
        self.label_decoder = LabelDecoder(self.z_dim)
        self.poe = ProductOfExperts()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, image, label):
        mu, logvar = self._poe_inference(image, label)
        z = self.reparameterize(mu, logvar)

        recon_image = self.image_decoder(z)
        recon_label = self.label_decoder(z)

        return recon_image, recon_label, z, mu, logvar

    def get_prior_expert(self, batch_size, use_cuda):
        mu = Variable(torch.zeros((batch_size, self.z_dim)))
        logvar = Variable(torch.zeros((batch_size, self.z_dim)))

        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()

        return mu, logvar

    def _poe_inference(self, image, label):
        assert not (image is None and label is None)
        if image is None:
            batch_size = label.size(0)
            use_cuda = label.is_cuda
        else:
            batch_size = image.size(0)
            use_cuda = image.is_cuda

        mu_arr, logvar_arr = [], []
        prior_mu, prior_logvar = self.get_prior_expert(batch_size, use_cuda)

        if image is not None:
            image_mu, image_logvar = self.image_encoder(image)
            mu_arr.extend([prior_mu, image_mu])
            logvar_arr.extend([prior_logvar, image_logvar])

        if label is not None:
            label_mu, label_logvar = self.label_encoder(label)
            mu_arr.extend([prior_mu, label_mu])
            logvar_arr.extend([prior_logvar, label_logvar])

        mu = torch.stack(mu_arr)
        logvar = torch.stack(logvar_arr)
        mu, logvar = self.poe(mu, logvar)

        return mu, logvar

    def get_joint_marginal(self, image, label, n_samples=1):
        assert image is not None
        assert label is not None
        batch_size =  image.size(0)

        mu, logvar = self._poe_inference(image, label)

        mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        logvar = logvar.unsqueeze(1).repeat(1, n_samples, 1)

        z = self.reparameterize(mu, logvar)
        z2d = z.view(batch_size * n_samples, self.z_dim)

        recon_image_2d = self.image_decoder(z2d)
        recon_label_2d = self.label_decoder(z2d)

        recon_image_2d = recon_image_2d.view(batch_size * n_samples, 784)
        recon_image = recon_image_2d.view(batch_size, n_samples, 784)
        image = image.view(batch_size, 784)

        label_dim = recon_label_2d.size(1)
        recon_label = recon_label_2d.view(batch_size, n_samples, label_dim)

        log_p = log_joint_estimate(
            recon_image, image, recon_label, label, z, mu, logvar)

        return log_p


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = MNISTImageEncoder(self.z_dim)
        self.decoder = MNISTImageDecoder(self.z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        recon_data = self.decoder(z)

        return recon_data, z, mu, logvar

    def get_marginal(self, data, n_samples=1):
        batch_size =  data.size(0)

        mu, logvar = self.encoder(data)

        mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        logvar = logvar.unsqueeze(1).repeat(1, n_samples, 1)

        z = self.reparameterize(mu, logvar)
        z2d = z.view(batch_size * n_samples, self.z_dim)

        recon_data_2d = self.decoder(z2d)
        recon_data_2d = recon_data_2d.view(batch_size * n_samples, 784)
        recon_data = recon_data_2d.view(batch_size, n_samples, 784)
        data = data.view(batch_size, 784)

        log_p = log_marginal_estimate(recon_data, data, z, mu, logvar)

        return log_p


class ImageEncoder(nn.Module):
    """
    Parameterizes q(z|image). Uses DC-GAN architecture.

    https://arxiv.org/abs/1511.06434
    https://github.com/ShengjiaZhao/InfoVAE/blob/master/model_vae.py

    @param z_dim: integer
                  number of latent dimensions.
    """
    def __init__(self, z_dim):
        super(ImageEncoder, self).__init__()
        self.z_dim = z_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.z_dim * 2),
        )

    def forward(self, x):
        batch_size = x.size(0)
        h = self.conv_layers(x)
        h = h.view(batch_size, 128 * 7 * 7)
        h = self.fc_layers(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class ImageDecoder(nn.Module):
    """
    Parameterizes p(image|z). Uses DC-GAN architecture.

    https://arxiv.org/abs/1511.06434
    https://github.com/ShengjiaZhao/InfoVAE/blob/master/model_vae.py

    @param z_dim: integer
                  number of latent dimensions.
    """
    def __init__(self, z_dim):
        super(ImageDecoder, self).__init__()
        self.z_dim = z_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 7 * 7),
            nn.ReLU(),
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
        )

    def forward(self, z):
        batch_size = z.size(0)
        h = self.fc_layers(z)
        h = h.view(batch_size, 128, 7, 7)
        h = self.conv_layers(h)
        return h  # no sigmoid!


class LabelEncoder(nn.Module):
    """
    Parameterizes q(z|label).

    @param z_dim: integer
                  number of latent dimensions.
    @param hidden_dim: integer [default: 400]
                       number of hidden dimensions
    """
    def __init__(self, z_dim, hidden_dim=400):
        super(LabelEncoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Embedding(10, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.z_dim * 2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class LabelDecoder(nn.Module):
    """
    Parameterizes p(label|z).

    @param z_dim: integer
                  number of latent dimensions.
    @param hidden_dim: integer [default: 400]
                       number of hidden dimensions
    """
    def __init__(self, z_dim, hidden_dim=400):
        super(LabelDecoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 10)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return F.log_softmax(self.fc4(h), dim=1)
