from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.datasets import MNIST
from .datasets import FashionMNIST
from torchvision import transforms

from ..utils.utils import (
    bernoulli_log_pdf,
    categorical_log_pdf,
    gaussian_log_pdf,
    unit_gaussian_log_pdf,
    log_mean_exp,
)

VALID_DATASETS = ['MNIST', 'FashionMNIST']


def joint_elbo(recon_image, image, recon_label, label, z, mu, logvar,
               lambda_image=1., lambda_label=1., annealing_factor=1.):
    r"""Lower bound on the joint distribution over images and labels.

    @param recon_image: torch.Tensor
                        tensor of logits (pre-sigmoid) representing
                        each pixel.
    @param image: torch.Tensor
                  observation of an image
    @param recon_label: torch.Tensor
                        tensor of probabilities for each label
    @param label: torch.Tensor
                  observation of a label
    @param z: torch.Tensor
              latent sample
    @param mu: torch.Tensor
               mean of variational distribution
    @param logvar: torch.Tensor
                   log-variance of variational distribution
    """
    BCE = bernoulli_log_pdf(image, recon_image)
    LCE = categorical_log_pdf(label, recon_label)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.sum(KLD, dim=1)

    # lower bound on marginal likelihood
    ELBO = lambda_image * BCE + lambda_label * LCE + annealing_factor * KLD

    return torch.mean(ELBO)


def image_elbo(recon_image, image, z, mu, logvar, lambda_nll=1., annealing_factor=1.):
    r"""Lower bound on image evidence (bernoulli parameterization).

    @param recon_image: torch.Tensor
                        tensor of logits (pre-sigmoid) representing
                        each pixel.
    @param image: torch.Tensor
                  observation of an image
    @param z: torch.Tensor
              latent sample
    @param mu: torch.Tensor
               mean of variational distribution
    @param logvar: torch.Tensor
                   log-variance of variational distribution
    """
    BCE = bernoulli_log_pdf(image, recon_image)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.sum(KLD, dim=1)

    # lower bound on marginal likelihood
    ELBO = lambda_nll * BCE + annealing_factor * KLD

    return torch.mean(ELBO)


def label_elbo(recon_label, label, z, mu, logvar, lambda_nll=1., annealing_factor=1.):
    r"""Lower bound on label evidence (categorical parameterization).

    @param recon_label: torch.Tensor
                        tensor of probabilities for each label
    @param label: torch.Tensor
                  observation of a label
    @param z: torch.Tensor
              latent sample
    @param mu: torch.Tensor
               mean of variational distribution
    @param logvar: torch.Tensor
                   log-variance of variational distribution
    """
    CE = categorical_log_pdf(label, recon_label)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.sum(KLD, dim=1)

    # lower bound on marginal likelihood
    ELBO = lambda_nll * CE + annealing_factor * KLD

    return torch.mean(ELBO)


def log_joint_estimate(recon_image, image, recon_label, label, z, mu, logvar):
    r"""Estimate log p(x,y).

    @param recon_image: torch.Tensor (batch size x # samples x 784)
                        reconstructed means on bernoulli
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param recon_label: torch.Tensor (batch_size x # samples x n_class)
                        reconstructed logits
    @param label: torch.Tensor (batch_size)
                  original observed labels
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """
    batch_size, n_samples, z_dim = z.size()
    input_dim = image.size(1)
    label_dim = recon_label.size(2)
    image = image.unsqueeze(1).repeat(1, n_samples, 1)
    label = label.unsqueeze(1).repeat(1, n_samples)

    z2d = z.view(batch_size * n_samples, z_dim)
    mu2d = mu.view(batch_size * n_samples, z_dim)
    logvar2d = logvar.view(batch_size * n_samples, z_dim)
    recon_image_2d = recon_image.view(batch_size * n_samples, input_dim)
    image_2d = image.view(batch_size * n_samples, input_dim)
    recon_label_2d = recon_label.view(batch_size * n_samples, label_dim)
    label_2d = label.view(batch_size * n_samples)

    log_p_x_given_z_2d = bernoulli_log_pdf(image_2d, recon_image_2d)
    log_p_y_given_z_2d = categorical_log_pdf(label_2d, recon_label_2d)
    log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
    log_p_z_2d = unit_gaussian_log_pdf(z2d)

    log_weight_2d = log_p_x_given_z_2d + log_p_y_given_z_2d + \
                    log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p)


def log_marginal_estimate(recon_image, image, z, mu, logvar):
    r"""Estimate log p(x). NOTE: this is not the objective that
    should be directly optimized.

    @param recon_image: torch.Tensor (batch size x # samples x 784)
                        reconstructed means on bernoulli
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """
    batch_size, n_samples, z_dim = z.size()
    input_dim = image.size(1)
    image = image.unsqueeze(1).repeat(1, n_samples, 1)

    z2d = z.view(batch_size * n_samples, z_dim)
    mu2d = mu.view(batch_size * n_samples, z_dim)
    logvar2d = logvar.view(batch_size * n_samples, z_dim)
    recon_image_2d = recon_image.view(batch_size * n_samples, input_dim)
    image_2d = image.view(batch_size * n_samples, input_dim)

    log_p_x_given_z_2d = bernoulli_log_pdf(image_2d, recon_image_2d)
    log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
    log_p_z_2d = unit_gaussian_log_pdf(z2d)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p)


def gen_dataset(name, data_dir, train=True, transform=None):
    assert name in VALID_DATASETS, \
        "dataset <%s> not recognized." % name

    if name == 'FashionMNIST':
        dataset = FashionMNIST(data_dir, train=train, download=True,
                               transform=transform)
    elif name == 'MNIST':
        dataset = MNIST(data_dir, train=train, download=True,
                        transform=transform)

    return dataset


def binarize(x):
    x = transforms.ToTensor()(x)
    # https://github.com/ShengjiaZhao/InfoVAE/blob/master/dataset/dataset_mnist.py
    x = torch.round(x)
    return x


def dynamic_binarize(x):
    x = transforms.ToTensor()(x)
    x = torch.bernoulli(x)
    return x
