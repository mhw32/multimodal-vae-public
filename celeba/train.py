from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from model import MVAE
from datasets import CelebAttributes
from datasets import N_ATTRS


def elbo_loss(recon_image, image, recon_attrs, attrs, mu, logvar,
              lambda_image=1.0, lambda_attrs=1.0, annealing_factor=1):
    """Bimodal ELBO loss function. 
    
    @param recon_image: torch.Tensor
                        reconstructed image
    @param image: torch.Tensor
                  input image
    @param recon_attrs: torch.Tensor
                        reconstructed attribute probabilities
    @param attrs: torch.Tensor
                  input attributes
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param lambda_attrs: float [default: 1.0]
                       weight for attribute BCE
    @param annealing_factor: integer [default: 1]
                             multiplier for KL divergence term
    @return ELBO: torch.Tensor
                  evidence lower bound
    """
    image_bce, attrs_bce = 0, 0  # default params
    
    if recon_image is not None and image is not None:
        image_bce = torch.sum(binary_cross_entropy_with_logits(
            recon_image.view(-1, 3 * 64 * 64), 
            image.view(-1, 3 * 64 * 64)), dim=1)

    if recon_attrs is not None and attrs is not None:
        for i in xrange(N_ATTRS):
            attr_bce = binary_cross_entropy_with_logits(
                recon_attrs[:, i], attrs[:, i])
            attrs_bce += attr_bce

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_image * image_bce + lambda_attrs * attrs_bce 
                      + annealing_factor * KLD)
    return ELBO


def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy

    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target 
            + torch.log(1 + torch.exp(-torch.abs(input))))


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


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MVAE(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=100,
                        help='size of the latent embedding [default: 100]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 100]')
    parser.add_argument('--annealing-epochs', type=int, default=20, metavar='N',
                        help='number of epochs to anneal KL for [default: 20]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate [default: 1e-4]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lambda-image', type=float, default=1.,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--lambda-attrs', type=float, default=10.,
                        help='multipler for attributes reconstruction [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')

    # crop the input image to 64 x 64
    preprocess_data = transforms.Compose([transforms.Resize(64),
                                          transforms.CenterCrop(64),
                                          transforms.ToTensor()])

    train_loader   = torch.utils.data.DataLoader(
        CelebAttributes(partition='train', data_dir='./data',
                        image_transform=preprocess_data),
        batch_size=args.batch_size, shuffle=True)
    N_mini_batches = len(train_loader)
    test_loader    = torch.utils.data.DataLoader(
        CelebAttributes(partition='val', data_dir='./data',
                        image_transform=preprocess_data),
        batch_size=args.batch_size, shuffle=False)

    model     = MVAE(args.n_latents)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()


    def train(epoch):
        model.train()
        train_loss_meter = AverageMeter()

        # NOTE: is_paired is 1 if the example is paired
        for batch_idx, (image, attrs) in enumerate(train_loader):
            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            if args.cuda:
                image     = image.cuda()
                attrs     = attrs.cuda()

            image      = Variable(image)
            attrs      = Variable(attrs)
            batch_size = len(image)

            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through model
            recon_image_1, recon_attrs_1, mu_1, logvar_1 = model(image, attrs)
            recon_image_2, recon_attrs_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_attrs_3, mu_3, logvar_3 = model(attrs=attrs)
                
            # compute ELBO for each data combo
            joint_loss = elbo_loss(recon_image_1, image, recon_attrs_1, attrs, mu_1, logvar_1, 
                                   lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs,
                                   annealing_factor=annealing_factor)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, 
                                   lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs,
                                   annealing_factor=annealing_factor)
            attrs_loss = elbo_loss(None, None, recon_attrs_3, attrs, mu_3, logvar_3, 
                                   lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs,
                                   annealing_factor=annealing_factor)
            train_loss = joint_loss + image_loss + attrs_loss
            train_loss_meter.update(train_loss.data[0], batch_size)
            
            # compute and take gradient step
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


    def test(epoch):
        model.eval()
        test_loss_meter = AverageMeter()

        pbar = tqdm(total=len(test_loader))
        for batch_idx, (image, attrs) in enumerate(test_loader):
            if args.cuda:
                image  = image.cuda()
                attrs  = attrs.cuda()
            
            image      = Variable(image, volatile=True)
            attrs      = Variable(attrs, volatile=True)
            batch_size = len(image)

            recon_image_1, recon_attrs_1, mu_1, logvar_1 = model(image, attrs)
            recon_image_2, recon_attrs_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_attrs_3, mu_3, logvar_3 = model(attrs=attrs)
            joint_loss = elbo_loss(recon_image_1, image, recon_attrs_1, attrs, mu_1, logvar_1, 
                                   lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, 
                                   lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs)
            attrs_loss = elbo_loss(None, None, recon_attrs_3, attrs, mu_3, logvar_3, 
                                   lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs)
            test_loss = joint_loss + image_loss + attrs_loss
            test_loss_meter.update(test_loss.data[0], batch_size)
            pbar.update()

        pbar.close()
        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        return test_loss_meter.avg

    
    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss      = test(epoch)
        is_best   = loss < best_loss
        best_loss = min(loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')   
