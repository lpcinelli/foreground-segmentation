import argparse
import os
import warnings

import glob2 as glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data  # Torch dataset class

import torchvision.transforms as vision_transforms
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as F

from ..transforms import *
from ..utils.img_utils import IMG_EXTENSIONS
from .file import DataFile

DEFAULT_SHAPE = (2, 192, 256)

class CDNetDataset(data.Dataset):

    def __init__(self, manifest_path, database_path, transform=None, **kwargs):
        '''
        Inits an ImageFile instance.

        @param csv_file csv file path containing the targets.
        @param transform Images transforms.
        @param target_transform Labels transform.
        @param loader Images loader.
        '''

        # Loads data
        self.database_path = database_path
        training = kwargs.get('training', False)
        shrink_data = kwargs.pop('shrink_data', False)
        data = self.from_file(manifest_path, training, shrink_data)

        if len(data) == 0:
            raise(RuntimeError("Found 0 images in path: " + manifest_path + "\n" +
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        # Main data
        self.data = data

        # Saving data
        self.manifest_path = manifest_path

        self.input_shape = kwargs.pop('input_shape', DEFAULT_SHAPE)
        if not transform:
            transform = transforms(self.input_shape, **kwargs)
        self.transform = transform

    def __getitem__(self, index):
        '''
        Returns image and target values for a given index.

        @param index Input index.

        @return The image and its respective target.
        '''

        # Get single data
        (input_, bg_model), (target, roi) = self.data[index]

        input_ = self.loader(input_)
        bg_model = self.loader(bg_model)
        target = self.loader(target, to_gray=True)
        roi = self.loader(roi, to_gray=True)

        # Transforming image
        if self.transform is not None:
            input_, target, roi = self.transform((input_, bg_model, target, roi))

        # Return
        return input_, target, roi

    def __len__(self):
        '''
        Returns samples size.

        @return Current data number of samples.
        '''
        return len(self.data)

    def loader(self, path, to_gray=False):
        img = default_loader(path)

        if to_gray:
            img = F.to_grayscale(img)

        return img

    def from_file(self, csv_file, training, shrink_data):
        # Opening file
        dataset = pd.read_csv(csv_file)
        imgs, targets = [], []

        for _, row in dataset.iterrows():
            if training and shrink_data and row['negative_only']:
                continue

            input_path = os.path.join(self.database_path, row['video_type'],
                                      row['video_name'], 'input',
                                      row['input_frame'])

            bg_path = os.path.join(self.database_path, row['video_type'],
                                   row['video_name'], 'bg_model.jpg')

            target_path = os.path.join(self.database_path, row['video_type'],
                                       row['video_name'], 'groundtruth',
                                       row['target_frame'])

            roi_path = os.path.join(self.database_path, row['video_type'],
                                    row['video_name'], 'ROI.bmp')

            imgs.append((input_path, bg_path))
            targets.append((target_path, roi_path))

        return list(zip(imgs, targets))


def transforms(input_shape,training=False, augmentation=False):

    if not training and augmentation is not False:
        raise ValueError('Combinations of parameters not permitted. '
                         'training=False, augmentation=True')

    C, H, W = input_shape

    compose = [Resize((H, W)), MergeChannels(), ToTensor()]

    if C < 3:
        compose = [Grayscale(1)] + compose
    if augmentation is True:
            compose = [RandomHorizontalFlip()] + compose

    return vision_transforms.Compose(compose)
