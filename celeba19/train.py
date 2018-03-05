from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from model import MVAE

sys.path.append('../celeba')
from datasets import N_ATTRS
from datasets import CelebAttributes


def elbo_loss(recon, data, mu, logvar, lambda_image=1.0, 
              lambda_attr=1.0, annealing_factor=1.):
    """Compute the ELBO for an arbitrary number of data modalities.

    @param recon: list of torch.Tensors/Variables
                  Contains one for each modality.
    @param data: list of torch.Tensors/Variables
                 Size much agree with recon.
    @param mu: Torch.Tensor
               Mean of the variational distribution.
    @param logvar: Torch.Tensor
                   Log variance for variational distribution.
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param lambda_attr: float [default: 1.0]
                        weight for attribute BCE
    @param annealing_factor: float [default: 1]
                             Beta - how much to weight the KL regularizer.
    """
    assert len(recon) == len(data), "must supply ground truth for every modality."
    n_modalities = len(recon)
    batch_size   = mu.size(0)

    BCE  = 0  # reconstruction cost
    for ix in xrange(n_modalities):
        # dimensionality > 1 implies an image
        if len(recon[ix].size()) > 1:
            recon_ix = recon[ix].view(batch_size, -1)
            data_ix  = data[ix].view(batch_size, -1)
            BCE += lambda_image * torch.sum(binary_cross_entropy_with_logits(recon_ix, data_ix), dim=1)
        else:  # this is for an attribute
            BCE += lambda_attr * binary_cross_entropy_with_logits(recon[ix], data[ix])
    KLD  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(BCE + annealing_factor * KLD)
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


def tensor_2d_to_list(x):
    # convert a 2D tensor to a list of 1D tensors.
    n_dims = x.size(1)
    list_of_tensors = []
    for i in xrange(n_dims):
        list_of_tensors.append(x[:, i])
    return list_of_tensors


def enumerate_combinations(n):
    """Enumerate entire pool of combinations.
    
    We use this to define the domain of ELBO terms, 
    (the pool of 2^19 ELBO terms).

    @param n: integer
              number of features (19 for Celeb19)
    @return: a list of ALL permutations
    """
    combos = []
    for i in xrange(2, n):  # 1 to n - 1
        _combos = list(combinations(range(n), i))
        combos  += _combos

    combos_np = np.zeros((len(combos), n))
    for i in xrange(len(combos)):
        for idx in combos[i]:
            combos_np[i][idx] = 1

    combos_np = combos_np.astype(np.bool)
    return combos_np


def sample_combinations(pool, size=1):
    """Return boolean list of which data points to use to compute a modality.
    Ignore combinations that are all True or only contain a single True.

    @param pool: np.array
                 enumerating all possible combinations.
    @param size: integer (default: 1)
                 number of combinations to sample.
    """
    n_modalities = pool.shape[1]
    pool_size    = len(pool)
    pool_sums    = np.sum(pool, axis=1)
    pool_dist    = np.bincount(pool_sums)
    pool_space   = np.where(pool_dist > 0)[0]

    sample_pool  = np.random.choice(pool_space, size, replace=True)
    sample_dist  = np.bincount(sample_pool)
    if sample_dist.size < n_modalities:
        zeros_pad   = np.zeros(n_modalities - sample_dist.size).astype(np.int)
        sample_dist = np.concatenate((sample_dist, zeros_pad))
    
    sample_combo = []
    for ix in xrange(n_modalities):
        if sample_dist[ix] > 0:
            pool_i  = pool[pool_sums == ix]
            combo_i = np.random.choice(range(pool_i.shape[0]),    
                                       size=sample_dist[ix], 
                                       replace=False)
            sample_combo.append(pool_i[combo_i])

    sample_combo = np.concatenate(sample_combo)
    return sample_combo


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
    parser.add_argument('--approx-m', type=int, default=1,
                        help='number of ELBO terms to approx. the full MVAE objective [default: 1]')
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

    train_loader = torch.utils.data.DataLoader(
        CelebAttributes(partition='train', data_dir='./data',
                        image_transform=preprocess_data),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        CelebAttributes(partition='val', data_dir='./data',
                        image_transform=preprocess_data),
        batch_size=args.batch_size, shuffle=False)

    model     = MVAE(args.n_latents)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.cuda:
        model.cuda()

    # enumerate all combinations so we can sample from this
    # every gradient step. NOTE: probably not the most efficient
    # way to do this but oh well.
    combination_pool = enumerate_combinations(19)


    def train(epoch):
        model.train()
        train_loss_meter = AverageMeter()

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
            image         = Variable(image)
            attrs         = Variable(attrs)
            attrs         = tensor_2d_to_list(attrs)  # convert tensor to list
            batch_size    = len(image)

            # refresh the optimizer
            optimizer.zero_grad()

            train_loss    = 0  # accumulate train loss here so we don't store a lot of things.
            n_elbo_terms  = 0  # track number of ELBO terms

            # compute ELBO using all data (``complete")
            recon_image, recon_attrs, mu, logvar = model(image, attrs)
            train_loss += elbo_loss([recon_image] + recon_attrs, [image] + attrs, mu, logvar, 
                                    lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs,
                                    annealing_factor=annealing_factor)
            n_elbo_terms += 1  # keep track of how many terms there are

            # compute ELBO using only image data
            recon_image, _, mu, logvar = model(batch_size, image=image)
            train_loss += elbo_loss([recon_image], [image], mu, logvar, 
                                    lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs,
                                    annealing_factor=annealing_factor)
            n_elbo_terms += 1  # keep track of how many terms there are
            
            # compute ELBO using only text data
            for ix in xrange(len(attrs)):
                _, recon_attrs, mu, logvar = model(attrs=[attrs[k] if k == ix else None 
                                                          for k in xrange(len(attrs))])
                train_loss += elbo_loss([recon_attrs[ix]], [attrs[ix]], mu, logvar, 
                                        annealing_factor=annealing_factor)
                n_elbo_terms += 1

            # sample some number of terms
            if args.approx_m > 0:
                sample_combos = sample_combinations(combination_pool, size=args.approx_m)
                for sample_combo in sample_combos:
                    attrs_combo = sample_combo[1:]
                    recon_image, recon_attrs, mu, logvar = model(image=image if sample_combo[0] else None, 
                                                                 attrs=[attrs[ix] if attrs_combo[ix] else None 
                                                                        for ix in xrange(attrs_combo.size)])
                    if sample_combo[0]:  # check if image is present
                        elbo = elbo_loss([recon_image] + [recon_attrs[ix] for ix in xrange(attrs_combo.size) if attrs_combo[ix]],
                                         [image] + [attrs[ix] for ix in xrange(attrs_combo.size) if attrs_combo[ix]],
                                         mu, logvar, annealing_factor=annealing_factor)
                    else:
                        elbo = elbo_loss([recon_attrs[ix] for ix in xrange(attrs_combo.size) if attrs_combo[ix]],
                                         [attrs[ix] for ix in xrange(attrs_combo.size) if attrs_combo[ix]],
                                         mu, logvar, annealing_factor=annealing_factor)
                    train_loss += elbo
                    n_elbo_terms += 1

            assert n_elbo_terms == (len(attrs) + 1) + 1 + args.approx_m  # N + 1 + M
            train_loss_meter.update(train_loss.data[0], len(image))
            
            # compute and take gradient step
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


    def test(epoch):
        model.eval()
        test_loss = 0

        # for simplicitly, here i'm only going to track the joint loss. 
        for batch_idx, (image, attrs) in enumerate(test_loader):
            if args.cuda:
                image, attrs = image.cuda(), attrs.cuda()
            image      = Variable(image, volatile=True)
            attrs      = Variable(attrs, volatile=True)
            batch_size = image.size(0)
            attrs      = tensor_2d_to_list(attrs)
            # compute the elbo using all data.
            recon_image, recon_attrs, mu, logvar = model(image, attrs)
            test_loss += elbo_loss([recon_image] + recon_attrs, [image] + attrs, mu, logvar).data[0]

        test_loss /= len(test_loader)
        print('====> Test Loss: {:.4f}'.format(test_loss))
        return test_loss


    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = test(epoch)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_dir)   
