from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from train import load_checkpoint


def fetch_mnist_image(label):
    """Return a random image from the MNIST dataset with label.

    @param label: integer
                  a integer from 0 to 9
    @return: torch.autograd.Variable
             MNIST image
    """
    mnist_dataset = datasets.MNIST('./data', train=False, download=True, 
                                   transform=transforms.ToTensor())
    images = mnist_dataset.test_data.numpy()
    labels = mnist_dataset.test_labels.numpy()
    images = images[labels == label]
    image  = images[np.random.choice(np.arange(images.shape[0]))]
    image  = torch.from_numpy(image).float() 
    image  = image.unsqueeze(0)
    return Variable(image, volatile=True)


def fetch_mnist_text(label):
    """Randomly generate a number from 0 to 9.

    @param label: integer
                  a integer from 0 to 9
    @return: torch.autograd.Variable
             Variable wrapped around an integer.
    """
    text = torch.LongTensor([label])
    return Variable(text, volatile=True)


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
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    # mode 1: unconditional generation
    if not args.condition_on_image and not args.condition_on_text:
        mu      = Variable(torch.Tensor([0]))
        std     = Variable(torch.Tensor([1]))
        if args.cuda:
            mu  = mu.cuda()
            std = std.cuda()
    # mode 2: generate conditioned on image
    elif args.condition_on_image and not args.condition_on_text:
        image      = fetch_mnist_image(args.condition_on_image)
        if args.cuda:
            image  = image.cuda()
        mu, logvar = model.infer(image=image)
        std        = logvar.mul(0.5).exp_()
    # mode 3: generate conditioned on text
    elif args.condition_on_text and not args.condition_on_image:
        text       = fetch_mnist_text(args.condition_on_text)
        if args.cuda:
            text   = text.cuda()
        mu, logvar = model.infer(text=text)
        std        = logvar.mul(0.5).exp_()
    # mode 4: generate conditioned on image and text
    elif args.condition_on_text and args.condition_on_image:
        image      = fetch_mnist_image(args.condition_on_image)
        text       = fetch_mnist_text(args.condition_on_text)
        if args.cuda:
            image  = image.cuda()
            text   = text.cuda()
        mu, logvar = model.infer(image=image, text=text)
        std        = logvar.mul(0.5).exp_()

    # sample from uniform gaussian
    sample     = Variable(torch.randn(args.n_samples, model.n_latents))
    if args.cuda:
        sample = sample.cuda()
    # sample from particular gaussian by multiplying + adding
    mu         = mu.expand_as(sample)
    std        = std.expand_as(sample)
    sample     = sample.mul(std).add_(mu)
    # generate image and text
    img_recon  = F.sigmoid(model.image_decoder(sample)).cpu().data
    txt_recon  = F.log_softmax(model.text_decoder(sample), dim=1).cpu().data
    
    # save image samples to filesystem
    save_image(img_recon.view(args.n_samples, 1, 28, 28),
               './sample_image.png')
    # save text samples to filesystem
    with open('./sample_text.txt', 'w') as fp:
        txt_recon_np = txt_recon.numpy()
        txt_recon_np = np.argmax(txt_recon_np, axis=1).tolist()
        for i, item in enumerate(txt_recon_np):
            fp.write('Text (%d): %s\n' % (i, item))
