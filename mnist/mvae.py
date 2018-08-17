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

from .mnist_utils import (
    joint_elbo,
    image_elbo,
    label_elbo,
    gen_dataset,
    dynamic_binarize,
    VALID_DATASETS,
)
from .mnist_models import MVAE
from ..utils.utils import AverageMeter, save_checkpoint
from torchvision import transforms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='MNIST|FashionMNIST')
    parser.add_argument('--out-dir', type=str, default='./trained_models',
                        help='where to save trained model [default: ./trained_models]')
    parser.add_argument('--lambda-image', type=float, default=1.,
                        help='multiplier for image [default: 1]')
    parser.add_argument('--lambda-label', type=float, default=50,
                        help='multiplier for label [default: 50]')
    parser.add_argument('--z-dim', type=int, default=64,
                        help='size of the latent embedding [default: 64]')
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

    model = MVAE(args.z_dim)
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
        image_meter = AverageMeter()
        label_meter = AverageMeter()
        joint_meter = AverageMeter()

        for batch_idx, (image, label) in enumerate(train_loader):
            batch_size = len(image)

            if args.cuda:
                image, label = image.cuda(), label.cuda()

            image = Variable(image)
            label = Variable(label)

            optimizer.zero_grad()

            recon_image_1, recon_label_1, z_1, mu_1, logvar_1 = model(image, label)
            recon_image_2, _, z_2, mu_2, logvar_2 = model(image, None)
            _, recon_label_3, z_3, mu_3, logvar_3 = model(None, label)

            recon_image_1 = recon_image_1.view(batch_size, 784)
            recon_image_2 = recon_image_2.view(batch_size, 784)
            image = image.view(batch_size, 784)

            annealing_val = get_annealing_factor(batch_idx, epoch)

            elbo_joint = joint_elbo(recon_image_1, image, recon_label_1, label, z_1, mu_1, logvar_1,
                                    lambda_image=args.lambda_image, lambda_label=args.lambda_label,
                                    annealing_factor=annealing_val)
            elbo_image = image_elbo(recon_image_2, image, z_2, mu_2, logvar_2,
                                    lambda_nll=args.lambda_image,
                                    annealing_factor=annealing_val)
            elbo_label = label_elbo(recon_label_3, label, z_3, mu_3, logvar_3,
                                    lambda_nll=args.lambda_label,
                                    annealing_factor=annealing_val)

            melbo = elbo_joint + elbo_image + elbo_label
            train_loss = -melbo

            loss_meter.update(-train_loss.data[0], batch_size)
            joint_meter.update(elbo_joint.data[0], batch_size)
            image_meter.update(elbo_image.data[0], batch_size)
            label_meter.update(elbo_label.data[0], batch_size)

            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ({:.4f}|{:.4f}|{:.4f})\tAnnealing Factor: {:.3f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg,
                    joint_meter.avg, image_meter.avg, label_meter.avg, annealing_val))

        print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))

        return loss_meter.avg


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()

        pbar = tqdm(total=len(test_loader))
        for batch_idx, (image, label) in enumerate(test_loader):
            batch_size = len(image)

            if args.cuda:
                image, label = image.cuda(), label.cuda()

            image = Variable(image, volatile=True)
            label = Variable(label, volatile=True)

            lle = model.get_joint_marginal(image, label, n_samples=10)
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
