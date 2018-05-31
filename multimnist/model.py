"""This model will be quite similar to mnist/model.py 
except we will need to be slightly fancier in the 
encoder/decoders for each modality. Likely, we will need 
convolutions/deconvolutions and RNNs.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from utils import n_characters, max_length
from utils import SOS, FILL


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(MVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.text_encoder  = TextEncoder(n_latents, n_characters, n_hiddens=200, 
                                         bidirectional=True)
        self.text_decoder  = TextDecoder(n_latents, n_characters, n_hiddens=200)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents

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
        # initialize the universal prior expert
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

    This task is quite a bit harder than MNIST so we probably need 
    to use an CNN of some form. This will be good to get us ready for
    natural images.

    @param n_latents: integer
                      size of latent vector
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 2 * 2),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 5, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 2, 2)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py


class TextEncoder(nn.Module):
    """Parametrizes q(z|y).

    We train an embedding layer from the 10 digit space
    to move to a continuous domain. The GRU is optionally 
    bidirectional.

    @param n_latents: integer
                      size of latent vector
    @param n_characters: integer
                         number of possible characters (10 for MNIST)
    @param n_hiddens: integer [default: 200]
                      number of hidden units in GRU
    @param bidirectional: boolean [default: True]
                          hyperparameter for GRU.
    """
    def __init__(self, n_latents, n_characters, n_hiddens=200, bidirectional=True):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(n_characters, n_hiddens)
        self.gru = nn.GRU(n_hiddens, n_hiddens, 1, dropout=0.1, 
                          bidirectional=bidirectional)
        self.h2p = nn.Linear(n_hiddens, n_latents * 2)  # hiddens to parameters
        self.n_latents = n_latents
        self.n_hiddens = n_hiddens
        self.bidirectional = bidirectional

    def forward(self, x):
        n_hiddens = self.n_hiddens
        n_latents = self.n_latents
        x = self.embed(x)
        x = x.transpose(0, 1)  # GRU expects (seq_len, batch, ...)
        x, h = self.gru(x, None)
        x = x[-1]  # take only the last value
        if self.bidirectional:
            x = x[:, :n_hiddens] + x[:, n_hiddens:]  # sum bidirectional outputs
        x = self.h2p(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).

    GRU for text decoding. Given a start token, sample a character
    via an RNN and repeat for a fixed length.

    @param n_latents: integer
                      size of latent vector
    @param n_characters: integer
                         size of characters (10 for MNIST)
    @param n_hiddens: integer [default: 200]
                      number of hidden units in GRU
    """
    def __init__(self, n_latents, n_characters, n_hiddens=200):
        super(TextDecoder, self).__init__()
        self.embed = nn.Embedding(n_characters, n_hiddens)
        self.z2h = nn.Linear(n_latents, n_hiddens)
        self.gru = nn.GRU(n_hiddens + n_latents, n_hiddens, 2, dropout=0.1)
        self.h2o = nn.Linear(n_hiddens + n_latents, n_characters)
        self.n_latents = n_latents
        self.n_characters = n_characters

    def forward(self, z):
        n_latents = self.n_latents
        n_characters = self.n_characters
        batch_size = z.size(0)
        # first input character is SOS
        c_in = Variable(torch.LongTensor([SOS]).repeat(batch_size))
        # store output word here
        words = Variable(torch.zeros(batch_size, max_length, n_characters))
        if z.is_cuda:
            c_in = c_in.cuda()
            words = words.cuda()
        # get hiddens from latents
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        # look through n_steps and generate characters
        for i in xrange(max_length):
            c_out, h = self.step(i, z, c_in, h)
            sample = torch.max(F.log_softmax(c_out, dim=1), dim=1)[1]
            words[:, i] = c_out
            c_in = sample
        return words  # (batch_size, seq_len, ...)

    def step(self, ix, z, c_in, h):
        c_in = swish(self.embed(c_in))
        c_in = torch.cat((c_in, z), dim=1)
        c_in = c_in.unsqueeze(0)
        c_out, h = self.gru(c_in, h)
        c_out = c_out.squeeze(0)
        c_out = torch.cat((c_out, z), dim=1)
        c_out = self.h2o(c_out)
        return c_out, h  # NOTE: no softmax here. See train.py


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
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
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
