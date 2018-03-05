from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import copy
import random
import numpy as np
import numpy.random as npr
from PIL import Image
from random import shuffle
from scipy.misc import imresize

import torch
from torch.utils.data.dataset import Dataset

VALID_PARTITIONS = {'train': 0, 'val': 1, 'test': 2}
# go from label index to interpretable index
ATTR_TO_IX_DICT  = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18, 
                    'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38, 
                    'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21, 
                    'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26, 
                    'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32, 
                    'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22, 
                    'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3, 
                    'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23, 
                    'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}
# we only keep 18 of the more visually distinctive features
# See [1] Perarnau, Guim, et al. "Invertible conditional gans for 
#         image editing." arXiv preprint arXiv:1611.06355 (2016).
ATTR_IX_TO_KEEP  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
IX_TO_ATTR_DICT  = {v:k for k, v in ATTR_TO_IX_DICT.iteritems()}
N_ATTRS          = len(ATTR_IX_TO_KEEP)
ATTR_TO_PLOT     = ['Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Wavy_Hair']


class CelebAttributes(Dataset):
    """Define dataset of images of celebrities and attributes.
    
    The user needs to have pre-defined the Anno and Eval folder from 
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    @param partition: string
                      train|val|test [default: train]
                      See VALID_PARTITIONS global variable.
    @param data_dir: string
                     path to root of dataset images [default: ./data]
    @param image_transform: ?torchvision.Transforms
                            optional function to apply to training inputs
    @param attr_transform: ?torchvision.Transforms
                           optional function to apply to training outputs
    """
    def __init__(self, partition='train', data_dir='./data', 
                 image_transform=None, attr_transform=None):
        self.partition       = partition
        self.image_transform = image_transform
        self.attr_transform  = attr_transform
        self.data_dir        = data_dir
        assert partition in VALID_PARTITIONS.keys()
        self.image_paths     = load_eval_partition(partition, data_dir=data_dir)
        self.attr_data       = load_attributes(self.image_paths, partition,
                                               data_dir=data_dir)
        self.size            = int(len(self.image_paths))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_dir, 'img_align_celeba', 
                                  self.image_paths[index])
        attr       = self.attr_data[index]
        image      = Image.open(image_path).convert('RGB')

        if self.image_transform is not None:
            image  = self.image_transform(image)

        if self.attr_transform is not None:
            attr   = self.attr_transform(attr)

        return image, attr

    def __len__(self):
        return self.size


def load_eval_partition(partition, data_dir='./data'):
    """After downloading the dataset, we can load a subset for
    training or testing.

    @param partition: string
                      which subset to use (train|val|test)
    @param data_dir: string [default: ./data]
                     where the images are saved
    """
    eval_data = []
    with open(os.path.join(data_dir, 'Eval/list_eval_partition.txt')) as fp:
        rows  = fp.readlines()
        for row in rows:
            path, label = row.strip().split(' ')
            label       = int(label)
            if label == VALID_PARTITIONS[partition]:
                eval_data.append(path)
    return eval_data


def load_attributes(paths, partition, data_dir='./data'):
    """Load the attributes into a torch tensor.

    @param paths: string
                  a numpy array of attributes (1 or 0)
    @param partition: string
                      which subset to use (train|val|test)
    @param data_dir: string [default: ./data]
                     where the images are saved
    """
    if os.path.isfile(os.path.join(data_dir, 'Anno/attr_%s.npy' % partition)):
        attr_data = np.load(os.path.join(data_dir, 'Anno/attr_%s.npy' % partition))
    else:
        attr_data = []
        with open(os.path.join(data_dir, 'Anno/list_attr_celeba.txt')) as fp:
            rows    = fp.readlines()
            for ix, row in enumerate(rows[2:]):
                row = row.strip().split()
                path, attrs = row[0], row[1:]
                if path in paths:
                    attrs   = np.array(attrs).astype(int)
                    attrs[attrs < 0] = 0
                    attr_data.append(attrs)
        attr_data = np.vstack(attr_data).astype(np.int64)
    attr_data     = torch.from_numpy(attr_data).float()
    return attr_data[:, ATTR_IX_TO_KEEP]


def tensor_to_attributes(tensor):
    """Use this for the <image_transform>.

    @param tensor: PyTorch Tensor
                   D dimensional tensor
    @return attributes: list of strings
    """
    attrs  = []
    n      = tensor.size(0)
    tensor = torch.round(tensor)
    for i in xrange(n):
        if tensor[i] > 0.5:
            attr = IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]]
            attrs.append(attr)
    return attrs
