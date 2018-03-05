from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from train import load_checkpoint
from datasets import ATTR_IX_TO_KEEP, N_ATTRS
from datasets import ATTR_TO_IX_DICT, IX_TO_ATTR_DICT
from datasets import tensor_to_attributes
from datasets import CelebAttributes


def fetch_celeba_image(attr_str):
    """Return a random image from the CelebA dataset with label.

    @param label: string
                  name of the attribute (see ATTR_TO_IX_DICT)
    @return: torch.autograd.Variable
             CelebA image
    """
    loader = torch.utils.data.DataLoader(
        CelebAttributes(
            partition='test',
            image_transform=transforms.Compose([transforms.Resize(64),
                                                transforms.CenterCrop(64),
                                                transforms.ToTensor()])),
        batch_size=128, shuffle=False)
    images, attrs = [], []
    for batch_idx, (image, attr) in enumerate(loader):
        images.append(image)
        attrs.append(attr)
    images  = torch.cat(images).cpu().numpy()
    attrs   = torch.cat(attrs).cpu().numpy()
    attr_ix = ATTR_IX_TO_KEEP.index(ATTR_TO_IX_DICT[attr_str])
    images  = images[attrs[:, attr_ix] == 1]
    image   = images[np.random.choice(np.arange(images.shape[0]))]
    image   = torch.from_numpy(image).float() 
    image   = image.unsqueeze(0)
    return Variable(image, volatile=True)


def fetch_celeba_attrs(attr_str):
    """Return a random image from the CelebA dataset with label.

    @param label: string
                  name of the attribute (see ATTR_TO_IX_DICT)
    @return: torch.autograd.Variable
             Variable wrapped around an integer.
    """
    attrs          = torch.zeros(N_ATTRS)
    attr_ix        = ATTR_IX_TO_KEEP.index(ATTR_TO_IX_DICT[attr_str])
    attrs[attr_ix] = 1
    return Variable(attrs.unsqueeze(0), volatile=True)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--n-samples', type=int, default=64, 
                        help='Number of images and texts to sample [default: 64]')
    # condition sampling on a particular images
    parser.add_argument('--condition-on-image', type=int, default=None,
                        help='If True, generate text conditioned on an image.')
    # condition sampling on a particular text
    parser.add_argument('--condition-on-text', type=int, default=None, 
                        help='If True, generate images conditioned on a text.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    # mode 1: unconditional generation
    if not args.condition_on_image and not args.condition_on_attrs:
        mu      = Variable(torch.Tensor([0]))
        std     = Variable(torch.Tensor([1]))
        if args.cuda:
            mu  = mu.cuda()
            std = std.cuda()
    # mode 2: generate conditioned on image
    elif args.condition_on_image and not args.condition_on_attrs:
        image      = fetch_celeba_image(args.condition_on_image)
        if args.cuda:
            image  = image.cuda()
        mu, logvar = model.get_params(image=image)
        std        = logvar.mul(0.5).exp_()
    # mode 3: generate conditioned on attrs
    elif args.condition_on_attrs and not args.condition_on_image:
        attrs      = fetch_celeba_attrs(args.condition_on_attrs)
        if args.cuda:
            attrs  = attrs.cuda()
        mu, logvar = model.get_params(attrs=attrs)
        std        = logvar.mul(0.5).exp_()
    # mode 4: generate conditioned on image and attrs
    elif args.condition_on_attrs and args.condition_on_image:
        image      = fetch_celeba_image(args.condition_on_image)
        attrs      = fetch_celeba_attrs(args.condition_on_attrs)
        if args.cuda:
            image  = image.cuda()
            attrs  = attrs.cuda()
        mu, logvar = model.get_params(image=image, attrs=attrs)
        std        = logvar.mul(0.5).exp_()

    # sample from uniform gaussian
    sample      = Variable(torch.randn(args.n_samples, model.n_latents))
    if args.cuda:
        sample  = sample.cuda()
    # sample from particular gaussian by multiplying + adding
    mu          = mu.expand_as(sample)
    std         = std.expand_as(sample)
    sample      = sample.mul(std).add_(mu)
    # generate image and text
    image_recon = F.sigmoid(model.image_decoder(sample)).cpu().data
    attrs_recon = F.sigmoid(model.attrs_decoder(sample)).cpu().data

    # save image samples to filesystem
    save_image(image_recon.view(args.n_samples, 3, 64, 64),
               './sample_image.png')
    # save text samples to filesystem
    sample_attrs = []
    for i in xrange(attrs_recon.size(0)):
        attrs = tensor_to_attributes(attrs_recon[i])
        sample_attrs.append(','.join(attrs))
    with open('./sample_attrs.txt', 'w') as fp:
        for attrs in sample_attrs:
            fp.write('%s\n' % attrs)
