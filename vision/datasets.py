from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import numpy as np
from copy import deepcopy
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

N_MODALITIES = 6
VALID_PARTITIONS = {'train': 0, 'val': 1, 'test': 2}


class CelebVision(Dataset):
    """Define dataset of images of celebrities with a series of 
    transformations applied to it.
    
    The user needs to have pre-defined the Anno and Eval folder from 
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    @param partition: string
                      train|val|test [default: train]
                      See VALID_PARTITIONS global variable.
    @param data_dir: string
                     path to root of dataset images [default: ./data]
    """
    def __init__(self, partition='train', data_dir='./data'):
        super(CelebVision, self).__init__()
        self.partition = partition
        self.data_dir = data_dir
        assert partition in VALID_PARTITIONS.keys()
        
        # load a list of images for the user-chosen partition
        self.image_paths = load_eval_partition(partition, data_dir=data_dir)
        self.size = int(len(self.image_paths))
        
        # resize image to 64 x 64
        self.image_transform = transforms.Compose([transforms.Resize(64),
                                                   transforms.CenterCrop(64),
                                                   transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_dir, 'img_align_celeba',
                                  self.image_paths[index])
        gray_path  = os.path.join(self.data_dir, 'img_align_celeba_grayscale',
                                  self.image_paths[index])
        edge_path  = os.path.join(self.data_dir, 'img_align_celeba_edge',
                                  self.image_paths[index])
        mask_path  = os.path.join(self.data_dir, 'img_align_celeba_mask', 
                                  self.image_paths[index])

        # open PIL Image -- these are fixed versions of image that we save
        image      = Image.open(image_path).convert('RGB')
        gray_image = Image.open(gray_path).convert('L')
        edge_image = Image.open(edge_path).convert('L')
        mask_image = Image.open(mask_path).convert('L')

        # add blocked to image
        obscured_image  = Image.open(image_path).convert('RGB')
        obscured_image  = obscure_image(obscured_image)

        # add watermark to image
        watermark_image = Image.open(image_path).convert('RGB')
        watermark_image = add_watermark(obscured_image, 
                                        watermark_path='./watermark.png')

        image           = self.image_transform(image)
        gray_image      = self.image_transform(grayscale_image)
        edge_image      = self.image_transform(edge_image)
        mask_image      = self.image_transform(mask_image)
        obscured_image  = self.image_transform(obscured_image)
        watermark_image = self.image_transform(watermark_image)
        # masks are normally white with black lines but we want to 
        # be consistent with edges and MNIST-stuff, we so make the background
        # black and the lines white.
        mask_image = 1 - mask_image

        # return everything as a bundle
        return (image, grayscale_image, edge_image, 
                mask_image, obscured_image, watermark_image)

    def __len__(self):
        return self.size


def obscure_image(image):
    """Block image vertically in half with black pixels.

    @param image: np.array
                  color image
    @return: np.array
             color image with vertically blocked pixels
    """
    image_npy = deepcopy(np.asarray(image))
    # we obscure half height because should be easier to complete
    # a face given vertical half than horizontal half
    center_h = image_npy.shape[1] // 2
    image_npy[:, center_h + 1:, :] = 0
    image = Image.fromarray(image_npy)
    return image


def add_watermark(image, watermark_path='./watermark.png'):
    """Overlay image of watermark on color image.

    @param image: np.array
                  color image
    @param watermark_path: string
                           path to fixed watermark image
                           [default: ./watermark.png]
    @return: np.array
             color image with overlayed watermark
    """
    watermark = Image.open(watermark_path)
    nw, nh = image.size[0], image.size[1]
    watermark = watermark.resize((nw, nh), Image.BICUBIC)
    image.paste(watermark, (0, 0), watermark)
    return image
