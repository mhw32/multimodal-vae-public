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
from torchvision.utils import save_image

from model import MVAE
from datasets import N_MODALITIES


def elbo_loss(recon_image, image, recon_gray, gray, recon_edge, edge, recon_mask, mask, 
              recon_rotated, rotated, recon_obscured, obscured, mu, logvar, annealing_factor=1.):
    BCE = 0
    if recon_image is not None and image is not None:
        recon_image, image = recon_image.view(-1, 3 * 64 * 64), image.view(-1, 3 * 64 * 64)
        image_BCE = torch.sum(binary_cross_entropy_with_logits(recon_image, image), dim=1)
        BCE += image_BCE

    if recon_gray is not None and gray is not None:
        recon_gray, gray = recon_gray.view(-1, 1 * 64 * 64), gray.view(-1, 1 * 64 * 64)
        gray_BCE = torch.sum(binary_cross_entropy_with_logits(recon_gray, gray), dim=1)
        BCE += gray_BCE

    if recon_edge is not None and edge is not None:
        recon_edge, edge = recon_edge.view(-1, 1 * 64 * 64), edge.view(-1, 1 * 64 * 64)
        edge_BCE = torch.sum(binary_cross_entropy_with_logits(recon_edge, edge), dim=1)
        BCE += edge_BCE

    if recon_mask is not None and mask is not None:
        recon_mask, mask = recon_mask.view(-1, 1 * 64 * 64), mask.view(-1, 1 * 64 * 64)
        mask_BCE = torch.sum(binary_cross_entropy_with_logits(recon_mask, mask), dim=1)
        BCE += mask_BCE

    if recon_obscured is not None and obscured is not None:
        recon_obscured, obscured = recon_obscured.view(-1, 3 * 64 * 64), obscured.view(-1, 3 * 64 * 64)
        obscured_BCE = torch.sum(binary_cross_entropy_with_logits(recon_obscured, obscured), dim=1)
        BCE += obscured_BCE

    if recon_watermark is not None and watermark is not None:
        recon_watermark, watermark = recon_watermark.view(-1, 3 * 64 * 64), watermark.view(-1, 3 * 64 * 64)
        watermark_BCE = torch.sum(binary_cross_entropy_with_logits(recon_watermark, watermark), dim=1)
        BCE += watermark_BCE

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # NOTE: we use lambda_i = 1 for all i since each modality is roughly equal
    ELBO = torch.mean(BCE / float(N_MODALITIES) + annealing_factor * KLD)
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
    parser.add_argument('--n-latents', type=int, default=250,
                        help='size of the latent embedding (default: 250)')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--annealing-epochs', type=int, default=20, metavar='N',
                        help='number of epochs to anneal KL for [default: 20]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    new_directories = ['./results', './results/image', './results/gray', './results/edge', 
                       './results/mask', './results/obscured', './results/watermark']
    
    for new_dir in new_directories:
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

    train_loader = torch.utils.data.DataLoader(
        datasets.CelebVision(partition='train', data_dir='./data'),
        batch_size=args.batch_size, shuffle=True)
    N_mini_batches = len(train_loader)
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebVision(partition='val', data_dir='./data'),
        batch_size=args.batch_size, shuffle=False)

    model = MVAE(args.n_latents, use_cuda=args.cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.cuda:
        model.cuda()


    def train(epoch):
        model.train()
        train_loss_meter = AverageMeter()

        for batch_idx, (image, gray_image, edge_image, mask_image, 
                        rotated_image, obscured_image) in enumerate(train_loader):
            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            if args.cuda:
                image            = image.cuda()
                gray_image       = gray_image.cuda()
                edge_image       = edge_image.cuda()
                mask_image       = mask_image.cuda()
                obscured_image   = obscured_image.cuda()
                watermark_image  = watermark_image.cuda()
            
            image           = Variable(image)
            gray_image      = Variable(gray_image)
            edge_image      = Variable(edge_image)
            mask_image      = Variable(mask_image)
            obscured_image  = Variable(obscured_image)
            watermark_image = Variable(watermark_image)
            batch_size      = image.size(0)

            # refresh the optimizer
            optimizer.zero_grad()
            
            # compute reconstructions using all the modalities
            (joint_recon_image, joint_recon_gray, joint_recon_edge, 
             joint_recon_mask, joint_recon_obscured, joint_recon_watermark,
             joint_mu, joint_logvar) = model(image, gray_image, edge_image, 
                                             mask_image, obscured_image, watermark_image)

             # compute reconstructions using each of the individual modalities
            (image_recon_image, image_recon_gray, image_recon_edge, 
             image_recon_mask, image_recon_obscured, image_recon_watermark,
             image_mu, image_logvar) = model(image=image)

            (gray_recon_image, gray_recon_gray, gray_recon_edge, 
             gray_recon_mask, gray_recon_obscured, gray_recon_watermark,
             gray_mu, gray_logvar) = model(gray=gray_image)

            (edge_recon_image, edge_recon_gray, edge_recon_edge, 
             edge_recon_mask, edge_recon_obscured, edge_recon_watermark,
             edge_mu, edge_logvar) = model(edge=edge_image)

            (mask_recon_image, mask_recon_gray, mask_recon_edge, 
             mask_recon_mask, mask_recon_obscured, mask_recon_watermark, 
             mask_mu, mask_logvar) = model(mask=mask_image)

            (obscured_recon_image, obscured_recon_gray, obscured_recon_edge, 
             obscured_recon_mask, obscured_recon_obscured, obscured_recon_watermark, 
             obscured_mu, obscured_logvar) = model(obscured=obscured_image)

            (watermark_recon_image, watermark_recon_gray, watermark_recon_edge, 
             watermark_recon_mask, watermark_recon_obscured, watermark_recon_watermark, 
             watermark_mu, watermark_logvar) = model(watermark=watermark_image)

            # compute joint loss
            joint_train_loss = elbo_loss(joint_recon_image, image, 
                                         joint_recon_gray, gray_image, 
                                         joint_recon_edge, edge_image,
                                         joint_recon_mask, mask_image,
                                         joint_recon_obscured, obscured_image, 
                                         joint_recon_watermark, watermark_image, 
                                         joint_mu, joint_logvar, 
                                         annealing_factor=annealing_factor)

            # compute loss with unimodal inputs
            image_train_loss = elbo_loss(image_recon_image, image, 
                                         image_recon_gray, gray_image, 
                                         image_recon_edge, edge_image,
                                         image_recon_mask, mask_image, 
                                         image_recon_obscured, obscured_image,
                                         image_recon_watermark, watermark_image,
                                         image_mu, image_logvar, 
                                         annealing_factor=annealing_factor)

            gray_train_loss = elbo_loss(gray_recon_image, image, 
                                        gray_recon_gray, gray_image, 
                                        gray_recon_edge, edge_image,
                                        gray_recon_mask, mask_image, 
                                        gray_recon_obscured, obscured_image,
                                        gray_recon_watermark, watermark_image,
                                        gray_mu, joint_logvar, 
                                        annealing_factor=annealing_factor)

            edge_train_loss = elbo_loss(edge_recon_image, image, 
                                        edge_recon_gray, gray_image, 
                                        edge_recon_edge, edge_image,
                                        edge_recon_mask, mask_image, 
                                        edge_recon_obscured, obscured_image,
                                        edge_recon_watermark, watermark_image,
                                        edge_mu, edge_logvar, 
                                        annealing_factor=annealing_factor)

            mask_train_loss = elbo_loss(mask_recon_image, image, 
                                        mask_recon_gray, gray_image, 
                                        mask_recon_edge, edge_image,
                                        mask_recon_mask, mask_image, 
                                        mask_recon_obscured, obscured_image,
                                        mask_recon_watermark, watermark_image,
                                        mask_mu, mask_logvar, 
                                        annealing_factor=annealing_factor)

            obscured_train_loss = elbo_loss(obscured_recon_image, image, 
                                            obscured_recon_gray, gray_image, 
                                            obscured_recon_edge, edge_image,
                                            obscured_recon_mask, mask_image, 
                                            obscured_recon_obscured, obscured_image,
                                            obscured_recon_watermark, watermark_image,
                                            obscured_mu, obscured_logvar, 
                                            annealing_factor=annealing_factor)

            watermark_train_loss = elbo_loss(watermark_recon_image, image, 
                                             watermark_recon_gray, gray_image, 
                                             watermark_recon_edge, edge_image,
                                             watermark_recon_mask, mask_image, 
                                             watermark_recon_obscured, obscured_image,
                                             watermark_recon_watermark, watermark_image,
                                             watermark_mu, watermark_logvar, 
                                             annealing_factor=annealing_factor)

            train_loss = joint_train_loss + image_train_loss + gray_train_loss \
                         + edge_train_loss + mask_train_loss + obscured_train_loss \
                         + watermark_train_loss
            train_loss_meter.update(train_loss.data[0], len(image))

            # compute and take gradient step
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing Factor: {:.3f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


    def test(epoch):
        model.eval()
        test_loss = 0

        pbar = tqdm(total=len(test_loader))
        for batch_idx, (image, gray_image, edge_image, mask_image, 
                        obscured_image, watermark_image) in enumerate(test_loader):
            if args.cuda:
                image           = image.cuda()
                gray_image      = gray_image.cuda()
                edge_image      = edge_image.cuda()
                mask_image      = mask_image.cuda()
                obscured_image  = obscured_image.cuda()
                watermark_image = watermark_image.cuda()
            
            image           = Variable(image)
            gray_image      = Variable(gray_image)
            edge_image      = Variable(edge_image)
            mask_image      = Variable(mask_image)
            obscured_image  = Variable(obscured_image)
            watermark_image = Variable(watermark_image)
            batch_size      = image.size(0)

            # for ease, only compute the joint loss
            (joint_recon_image, joint_recon_gray, joint_recon_edge, 
             joint_recon_mask, joint_recon_obscured, joint_recon_watermark,
             joint_mu, joint_logvar) = model(batch_size, image, gray_image, edge_image, 
                                             mask_image, obscured_image, watermark_image)
        
            test_loss += loss_function(joint_recon_image, image, 
                                       joint_recon_gray, gray_image, 
                                       joint_recon_edge, edge_image,
                                       joint_recon_mask, mask_image, 
                                       joint_recon_obscured, obscured_image, 
                                       joint_recon_watermark, watermark_image, 
                                       joint_mu, joint_logvar).data[0]

            if batch_idx == 0:
                # from time to time, plot the reconstructions to see how well the model is learning
                n = min(batch_size, 8)
                image_comparison = torch.cat(
                    [image[:n], 
                    F.sigmoid(joint_recon_image).view(args.batch_size, 3, 64, 64)[:n]])
                gray_comparison = torch.cat([
                    gray_image[:n], 
                    F.sigmoid(joint_recon_gray).view(args.batch_size, 1, 64, 64)[:n]])
                edge_comparison = torch.cat([
                    edge_image[:n], 
                    F.sigmoid(joint_recon_edge).view(args.batch_size, 1, 64, 64)[:n]])
                mask_comparison = torch.cat([
                    mask_image[:n], 
                    F.sigmoid(joint_recon_mask).view(args.batch_size, 1, 64, 64)[:n]])
                obscured_comparison = torch.cat([
                    obscured_image[:n], 
                    F.sigmoid(joint_recon_obscured).view(args.batch_size, 3, 64, 64)[:n]])
                watermark_comparison = torch.cat([
                    watermark_image[:n], 
                    F.sigmoid(joint_recon_watermark).view(args.batch_size, 3, 64, 64)[:n]])
                # save these reconstructions
                save_image(image_comparison.data.cpu(), 
                           './results/image/reconstruction_%d.png' % epoch, nrow=n)
                save_image(gray_comparison.data.cpu(), 
                           './results/gray/reconstruction_%d.png' % epoch, nrow=n)
                save_image(edge_comparison.data.cpu(), 
                           './results/edge/reconstruction_%d.png' % epoch, nrow=n)
                save_image(mask_comparison.data.cpu(), 
                           './results/mask/reconstruction_%d.png' % epoch, nrow=n)
                save_image(obscured_comparison.data.cpu(), 
                           './results/obscured/reconstruction_%d.png' % epoch, nrow=n)
                save_image(watermark_comparison.data.cpu(), 
                           './results/watermark/reconstruction_%d.png' % epoch, nrow=n)

            pbar.update()

        pbar.close()
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
        }, is_best, folder='./trained_models')   
