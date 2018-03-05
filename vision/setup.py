"""Grayscale, edge detection, and facial landmarks are pre-computed
prior to training. Obscuring and watermarks are done in-place in 
datasets.py. 

>>> python setup.py grayscale ./data/images ./data/grayscale
>>> python setup.py edge ./data/images ./data/edge
>>> python setup.py mask ./data/images ./data/mask
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import dlib
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from skimage import feature
from imutils import face_utils
from collections import OrderedDict

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


def build_grayscale_dataset(in_dir, out_dir):
    """Generate a dataset of grayscale images.

    @param in_dir: string
                   input directory of images.
    @param out_dir: string
                    output directory of images.
    """
    image_paths = os.listdir(in_dir)
    n_images = len(image_paths)
    for i, image_path in enumerate(image_paths):
        print('Building grayscale dataset: [%d/%d] images.' % (i + 1, n_images))
        image_full_path = os.path.join(in_dir, image_path)
        image = Image.open(image_full_path)
        image = image.convert('RGB').convert('L')
        image.save(os.path.join(out_dir, image_path))


def build_edge_dataset(in_dir, out_dir, sigma=3):
    """Generate a dataset of (canny) edge-detected images.

    @param in_dir: string
                   input directory of images.
    @param out_dir: string
                    output directory of images.
    @param sigma: float (default: 3)
                  smoothness for edge detection.
    """
    image_paths = os.listdir(in_dir)
    n_images = len(image_paths)
    for i, image_path in enumerate(image_paths):
        print('Building edge-detected dataset: [%d/%d] images.' % (i + 1, n_images))
        image_full_path = os.path.join(in_dir, image_path)
        image = Image.open(image_full_path).convert('L')
        image_npy = np.asarray(image).astype(np.float) / 255.
        image_npy = feature.canny(image_npy, sigma=sigma)
        image_npy = image_npy.astype(np.uint8) * 255
        image = Image.fromarray(image_npy)
        image.save(os.path.join(out_dir, image_path))


def build_mask_dataset(in_dir, out_dir, model_path):
    """Generate a dataset of segmentation masks from images.

    @param in_dir: string
                   input directory of images.
    @param out_dir: string
                    output directory of images.
    @param model_path: string
                       path to HOG model for facial features.
    """
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    image_paths = os.listdir(in_dir)
    n_images = len(image_paths)
    for i, image_path in enumerate(image_paths):
        print('Building face-mask dataset: [%d/%d] images.' % (i + 1, n_images))
        image_full_path = os.path.join(in_dir, image_path)

        image = cv2.imread(image_full_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        try:
            rect = rects[0]  # we are only going to use the first one

            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            output = visualize_facial_landmarks(image, shape)
            cv2.imwrite(os.path.join(out_dir, image_path), output)
        except:
            # if for some reason no bounding box is found, send blank.
            output = np.ones_like(image) * 255
            cv2.imwrite(os.path.join(out_dir, image_path), output)


def visualize_facial_landmarks(image, shape, colors=None):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = np.ones_like(image) * 255

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
 
        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, (0, 0, 0), 2)
 
        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, (0, 0, 0), -1)

    return overlay


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, 
                        help='grayscale||edge|mask')
    parser.add_argument('in_dir', type=str, help='where images are located')
    parser.add_argument('out_dir', type=str, help='where images are to be saved')
    args = parser.parse_args()

    if args.type == 'grayscale':
        build_grayscale_dataset(args.in_dir, args.out_dir)
    elif args.type == 'edge':
        build_edge_dataset(args.in_dir, args.out_dir, sigma=2)
    elif args.type == 'mask':
        build_mask_dataset(args.in_dir, args.out_dir, 
                           './data/shape_predictor_68_face_landmarks.dat')
