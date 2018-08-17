from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import torch
import shutil
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

LOG2PI = float(np.log(2.0 * math.pi))


def bernoulli_log_pdf(x, mu):
    """
    Log-likelihood of data given ~Bernoulli(mu)

    @param x: PyTorch.Tensor
              ground truth input
    @param mu: PyTorch.Tensor
               Bernoulli distribution parameters
               logits (no F.sigmoid)
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    log_pdf = -F.relu(mu) + torch.mul(x, mu) - torch.log(1. + torch.exp( -mu.abs() ))

    return torch.sum(log_pdf, dim=1)


def categorical_log_pdf(x, mu):
    """
    Log-likelihood of data given ~Cat(mu)

    @param x: PyTorch.Tensor
              ground truth input [batch_size]
    @param mu: PyTorch.Tensor
               Categorical distribution parameters
               log_softmax'd probabilities
               [batch_size x dims]
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    x_1hot = make_one_hot(x.data, mu.size(1))
    x_1hot = Variable(x_1hot, volatile=x.volatile)
    log_pdf = torch.sum(x_1hot * mu,  dim=1)
    return log_pdf


def gaussian_log_pdf(x, mu, logvar):
    """
    Log-likelihood of data given ~N(mu, exp(logvar))

    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - logvar / 2. - torch.pow(x - mu, 2) / (2. * torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def unit_gaussian_log_pdf(x):
    """
    Log-likelihood of data given ~N(0, 1)

    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - math.log(1.) / 2. - torch.pow(x, 2) / 2.
    return torch.sum(log_pdf, dim=1)


def make_one_hot(x, n_class):
    x = x.long()
    x_1hot = torch.FloatTensor(x.size(0), n_class)
    if x.is_cuda:
        x_1hot = x_1hot.cuda()
    x_1hot.zero_()
    x_1hot.scatter_(1, x.unsqueeze(1), 1)

    return x_1hot


def log_mean_exp(x, dim=1):
    """
    log(1/k * sum(exp(x))): this normalizes x.

    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))
