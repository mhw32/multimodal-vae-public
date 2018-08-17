from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    https://arxiv.org/pdf/1410.7827.pdf

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    @param eps: float [default: 1e-8]
                fudge factor for numerical stability
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1 / var  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        return pd_mu, pd_logvar
