"""
This script generates a dataset similar to the MultiMNIST dataset
described in [1]. However, we remove any translation.

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys

import random
import numpy as np
import numpy.random as npr
from PIL import Image
from random import shuffle
from scipy.misc import imresize

import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset


class MultiMNIST(Dataset):
    """Images with 0 to 4 digits of non-overlapping MNIST numbers.

    @param root: string
                 path to dataset root
    @param train: boolean [default: True]
           whether to return training examples or testing examples
    @param transform: ?torchvision.Transforms
                      optional function to apply to training inputs
    @param target_transform: ?torchvision.Transforms
                             optional function to apply to training outputs
    """
    processed_folder = 'multimnist'
    training_file    = 'training.pt'
    test_file        = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root             = os.path.expanduser(root)
        self.transform        = transform
        self.target_transform = target_transform
        self.train            = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img        = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img    = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        if self._check_exists():
            return
        make_dataset(self.root, self.processed_folder, 
                     self.training_file, self.test_file)


# -- code for generating MultiMNIST torch objects. --
# INSTRUCTIONS: run this file.

def sample_one(canvas_size, mnist, resize=True, translate=True):
    i     = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i]
    if resize:  # resize only if user specified
        scale   = 0.1 * np.random.randn() + 1.3
        resized = imresize(digit, 1. / scale)
    else:
        resized = digit
    w           = resized.shape[0]
    assert w == resized.shape[1]
    padding   = canvas_size - w
    if translate:  # translate only if user specified
        pad_l = np.random.randint(0, padding)
        pad_r = np.random.randint(0, padding)
        pad_width  = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    else:
        pad_l = padding // 2
        pad_r = padding // 2
        pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    return positioned, label


def sample_multi(num_digits, canvas_size, mnist, resize=True, translate=True):
    canvas = np.zeros((canvas_size, canvas_size))
    labels = []
    for _ in range(num_digits):
        positioned_digit, label = sample_one(canvas_size, mnist, resize=resize,
                                             translate=translate)
        canvas += positioned_digit
        labels.append(label)
    
    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multi(num_digits, canvas_size, mnist, 
                            resize=resize, translate=translate)
    else:
        return canvas, labels


def mk_dataset(n, mnist, min_digits, max_digits, canvas_size, 
               resize=True, translate=True):
    x = []
    y = []
    for _ in range(n):
        num_digits     = np.random.randint(min_digits, max_digits + 1)
        canvas, labels = sample_multi(num_digits, canvas_size, mnist,
                                      resize=resize, translate=translate)
        x.append(canvas)
        y.append(labels)
    return np.array(x, dtype=np.uint8), y


def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='./data', train=True, download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='./data', train=False, download=True))
    
    train_data = {
        'digits': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels
    }

    test_data = {
        'digits': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels
    }

    return train_data, test_data


def make_dataset(root, folder, training_file, test_file, min_digits=0, max_digits=2,
                 resize=True, translate=True):
    if not os.path.isdir(os.path.join(root, folder)):
        os.makedirs(os.path.join(root, folder))

    np.random.seed(681307)
    train_mnist, test_mnist = load_mnist()
    train_x, train_y = mk_dataset(60000, train_mnist, min_digits, max_digits, 50,
                                  resize=resize, translate=translate)
    test_x, test_y = mk_dataset(10000, test_mnist, min_digits, max_digits, 50,
                                resize=resize, translate=translate)
    
    train_x = torch.from_numpy(train_x).byte()
    test_x = torch.from_numpy(test_x).byte()

    training_set = (train_x, train_y)
    test_set = (test_x, test_y)

    with open(os.path.join(root, folder, training_file), 'wb') as f:
        torch.save(training_set, f)

    with open(os.path.join(root, folder, test_file), 'wb') as f:
        torch.save(test_set, f)


def sample_one_fixed(canvas_size, mnist, pad_l, pad_r, scale=1.3):
    i = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i]
    resized = imresize(digit, 1. / scale)
    w = resized.shape[0]
    assert w == resized.shape[1]
    padding = canvas_size - w
    pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
    positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    return positioned, label


def sample_multi_fixed(num_digits, canvas_size, mnist, reverse=False, 
                       scramble=False, no_repeat=False):
    canvas = np.zeros((canvas_size, canvas_size))
    labels = []
    pads = [(4, 4), (4, 23), (23, 4), (23, 23)]
    for i in range(num_digits):
        if no_repeat:  # keep trying to generate examples that are 
                       # not already in previously generated labels
            while True:
                positioned_digit, label = sample_one_fixed(
                    canvas_size, mnist, pads[i][0], pads[i][1])
                if label not in labels:
                    break
        else:
            positioned_digit, label = sample_one_fixed(
                canvas_size, mnist, pads[i][0], pads[i][1])
        
        canvas += positioned_digit
        labels.append(label)
    
    if reverse and random.random() > 0.5:
        labels = labels[::-1]

    if scramble:
        random.shuffle(labels)

    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multi_fixed(num_digits, canvas_size, mnist, reverse=reverse, 
                                  scramble=scramble, no_repeat=no_repeat)
    else:
        return canvas, labels


def mk_dataset_fixed(n, mnist, min_digits, max_digits, canvas_size, 
                     reverse=False, scramble=False, no_repeat=False):
    x = []
    y = []
    for _ in range(n):
        num_digits = np.random.randint(min_digits, max_digits + 1)
        canvas, labels = sample_multi_fixed(num_digits, canvas_size, mnist, reverse=reverse, 
                                            scramble=scramble, no_repeat=no_repeat)
        x.append(canvas)
        y.append(labels)
    return np.array(x, dtype=np.uint8), y


def make_dataset_fixed(root, folder, training_file, test_file, 
                       min_digits=0, max_digits=3, reverse=False, 
                       scramble=False, no_repeat=False):
    if not os.path.isdir(os.path.join(root, folder)):
        os.makedirs(os.path.join(root, folder))

    np.random.seed(681307)
    train_mnist, test_mnist = load_mnist()
    train_x, train_y = mk_dataset_fixed(60000, train_mnist, min_digits, max_digits, 50, 
                                        reverse=reverse, scramble=scramble, no_repeat=no_repeat)
    test_x, test_y = mk_dataset_fixed(10000, test_mnist, min_digits, max_digits, 50, 
                                      reverse=reverse, scramble=scramble, no_repeat=no_repeat)
    
    train_x = torch.from_numpy(train_x).byte()
    test_x = torch.from_numpy(test_x).byte()

    training_set = (train_x, train_y)
    test_set = (test_x, test_y)

    with open(os.path.join(root, folder, training_file), 'wb') as f:
        torch.save(training_set, f)

    with open(os.path.join(root, folder, test_file), 'wb') as f:
        torch.save(test_set, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-digits', type=int, default=0, 
                        help='minimum number of digits to add to an image')
    parser.add_argument('--max-digits', type=int, default=4,
                        help='maximum number of digits to add to an image')
    parser.add_argument('--no-resize', action='store_true', default=False,
                        help='if True, fix the image to be MNIST size')
    parser.add_argument('--no-translate', action='store_true', default=False,
                        help='if True, fix the image to be in the center')
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='If True, ignore resize/translate options and generate')
    parser.add_argument('--scramble', action='store_true', default=False,
                        help='If True, scramble labels and generate. Only does something if fixed is True.')
    parser.add_argument('--reverse', action='store_true', default=False, 
                        help='If True, reverse flips the labels i.e. 4321 instead of 1234 with 0.5 probability.')
    parser.add_argument('--no-repeat', action='store_true', default=False,
                        help='If True, do not generate images with multiple of the same label.')
    args = parser.parse_args()
    args.resize = not args.no_resize
    args.translate = not args.no_translate
    
    if args.no_repeat and not args.fixed:
        raise Exception('Must have --fixed if --no-repeat is supplied.')

    if args.scramble and not args.fixed:
        raise Exception('Must have --fixed if --scramble is supplied.')

    if args.reverse and not args.fixed:
        raise Exception('Must have --fixed if --reverse is supplied.')

    if args.reverse and args.scramble:
        print('Found --reversed and --scrambling. Overriding --reversed.')
        args.reverse = False

    # Generate the training set and dump it to disk. (Note, this will
    # always generate the same data, else error out.)
    if args.fixed:
        make_dataset_fixed('./data', 'multimnist', 'training.pt', 'test.pt',
                           min_digits=args.min_digits, max_digits=args.max_digits,
                           reverse=args.reverse, scramble=args.scramble, 
                           no_repeat=args.no_repeat)
    else:  # if not fixed, then make classic MultiMNIST dataset
        # VAEs in general have trouble handling translation and rotation,
        # likely resulting in blurry reconstructions without additional
        # attention mechanisms. See AIR [1].
        make_dataset('./data', 'multimnist', 'training.pt', 'test.pt',
                     min_digits=args.min_digits, max_digits=args.max_digits,
                     resize=args.resize, translate=args.translate)
