from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from train import load_checkpoint
from datasets import obscure_image
from datasets import add_watermark

# this is the same loader used in datasets.py
image_transform = transforms.Compose([transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor()])


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--condition-file', type=str, 
                        help='if specified, condition on this image.')
    parser.add_argument('--condition-type', type=str, 
                        help='image|gray|edge|mask|obscured|watermark')
    parser.add_argument('--n-samples', type=int, default=1, 
                        help='Number of images and texts to sample [default: 1]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.condition_type:
        assert args.condition_type in ['image', 'gray', 'edge', 'mask', 'obscured', 'watermark']

    if not os.path.isdir('./samples'):
        os.makedirs('./samples')

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    if args.condition_file and args.condition_type:
        image = Image.open(args.condition_file)        
        if args.condition_type == 'image':
            image = image.convert('RGB')
            image = image_transform(image).unsqueeze(0)
            save_image(image, './samples/sample_image.png')
            if args.cuda:
                image = image.cuda()
            image = Variable(image, volatile=True)
            mu, logvar = model.get_params(1, image=image)
        elif args.condition_type == 'gray':
            image = image.convert('L')
            image = image_transform(image).unsqueeze(0)
            save_image(image, './samples/sample_gray.png')
            if args.cuda:
                image = image.cuda()
            image = Variable(image, volatile=True)
            mu, logvar = model.get_params(1, gray=image)
        elif args.condition_type == 'edge':
            image = image.convert('L')
            image = image_transform(image).unsqueeze(0)
            save_image(image, './samples/sample_edge.png')
            if args.cuda:
                image = image.cuda()
            image = Variable(image, volatile=True)
            mu, logvar = model.get_params(1, edge=image)
        elif args.condition_type == 'mask':
            image = image.convert('L')
            image = 1 - image_transform(image).unsqueeze(0)
            save_image(image, './samples/sample_mask.png')
            if args.cuda:
                image = image.cuda()
            image = Variable(image, volatile=True)
            mu, logvar = model.get_params(1, mask=image)
        elif args.condition_type == 'obscured':
            image = image.convert('RGB')
            image = obscure_image(image)
            image = image_transform(image).unsqueeze(0)
            save_image(image, './samples/sample_obscured.png')
            if args.cuda:
                image = image.cuda()
            image = Variable(image, volatile=True)
            mu, logvar = model.get_params(1, obscured=image)
        elif args.condition_type == 'watermark':
            image = image.convert('RGB')
            image = add_watermark(image)
            image = image_transform(image).unsqueeze(0)
            save_image(image, './samples/sample_watermark.png')
            if args.cuda:
                image = image.cuda()
            image = Variable(image, volatile=True)
            mu, logvar = model.get_params(1, watermark=image)
        std = logvar.mul(0.5).exp_()
    else:  # sample from uniform Gaussian prior
        mu = Variable(torch.Tensor([0]))
        std = Variable(torch.Tensor([1]))
        if args.cuda:
            mu = mu.cuda()
            std = std.cuda()

    # sample from uniform gaussian
    sample = Variable(torch.randn(args.n_samples, model.n_latents))
    if args.cuda:
        sample = sample.cuda()
    
    # sample from particular gaussian by multiplying + adding
    mu = mu.expand_as(sample)
    std = std.expand_as(sample)
    sample = sample.mul(std).add_(mu)

    # generate image and text
    image_recon = F.sigmoid(model.image_decoder(sample)).cpu().data
    gray_recon = F.sigmoid(model.gray_decoder(sample)).cpu().data
    edge_recon = F.sigmoid(model.edge_decoder(sample)).cpu().data
    mask_recon = F.sigmoid(model.mask_decoder(sample)).cpu().data
    obscured_recon = F.sigmoid(model.obscured_decoder(sample)).cpu().data
    watermark_recon = F.sigmoid(model.watermark_decoder(sample)).cpu().data

    # save image samples to filesystem
    save_image(image_recon, './samples/sample_image.png')
    save_image(gray_recon, './samples/sample_gray.png')
    save_image(edge_recon, './samples/sample_edge.png')
    save_image(mask_recon, './samples/sample_mask.png')
    save_image(rotated_recon, './samples/sample_rotated.png')
    save_image(obscured_recon, './samples/sample_obscured.png')
    save_image(watermark_recon, './samples/sample_watermark.png')
