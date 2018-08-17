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
from torch import optim

from .utils import (
    image_elbo,
    gen_dataset,
    dynamic_binarize,
    VALID_DATASETS,
)
from .mnist_models import VAE
from ..utils.utils import AverageMeter, save_checkpoint
from torchvision import transforms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='MNIST|FashionMNIST')
    parser.add_argument('--out-dir', type=str, default='./trained_models',
                        help='where to save trained model [default: ./trained_models]')
    parser.add_argument('--z-dim', type=int, default=20,
                        help='size of the latent embedding [default: 20]')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 128]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 100]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--warm-up', action='store_true', default=False,
                        help='anneal kullback-leibler divergence')
    parser.add_argument('--annealing-epochs', type=int, default=20, metavar='N',
                        help='how many epochs to anneal for [default: 20]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility [default: 42]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.dataset in VALID_DATASETS, \
        "--dataset {%s} not recognized." %  args.dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train_dataset = gen_dataset(args.dataset, './data/%s' % args.dataset,
                                train=True, transform=dynamic_binarize)
    test_dataset = gen_dataset(args.dataset, './data/%s' % args.dataset,
                               train=False, transform=dynamic_binarize)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    N_mini_batches = len(train_loader)

    model = VAE(args.z_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()


    def get_annealing_factor(batch_idx, epoch):
        """Annealing factor see (Bowman, 2015)."""
        if args.warm_up and epoch < args.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = (
                float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                float(args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        return annealing_factor


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = len(data)

            if args.cuda:
                data = data.cuda()

            data = Variable(data)

            optimizer.zero_grad()

            recon_data, z, mu, logvar = model(data)

            recon_data = recon_data.view(batch_size, 784)
            data = data.view(batch_size, 784)

            annealing_val = get_annealing_factor(batch_idx, epoch)

            elbo = image_elbo(recon_data, data, z, mu, logvar,
                              annealing_factor=annealing_val)
            train_loss = -elbo

            loss_meter.update(-train_loss.data[0], batch_size)

            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing Factor: {:.3f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg, annealing_val))

        print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))

        return loss_meter.avg


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()

        pbar = tqdm(total=len(test_loader))
        for batch_idx, (data, _) in enumerate(test_loader):
            batch_size = len(data)

            if args.cuda:
                data = data.cuda()

            data = Variable(data, volatile=True)

            lle = model.get_marginal(data, n_samples=10)
            loss_meter.update(lle.data[0], batch_size)
            pbar.update()

        pbar.close()
        print('====> Test Epoch: {}\tLog Joint Estimate: {:.4f}'.format(epoch, loss_meter.avg))

        return loss_meter.avg


    best_loss = sys.maxint
    track_loss = np.zeros(args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        track_loss[epoch - 1] = test_loss

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'cmd_line_args': args,
        }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)
