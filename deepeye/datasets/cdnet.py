import argparse
import os
import warnings

import glob2 as glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data  # Torch dataset class

import torchvision.transforms as vision_transforms
import torchvision.transforms.functional as F
from torchvision.datasets.folder import default_loader

from ..transforms import *
from ..utils.img_utils import IMG_EXTENSIONS
from .file import DataFile

DEFAULT_SHAPE = (2, 192, 256)


class CDNetDataset(data.Dataset):
    def __init__(self,
                 manifest_path,
                 database_path,
                 transform=None,
                 training=False,
                 shrink_data=False,
                 input_shape=DEFAULT_SHAPE,
                 augmentation=False):
        '''
        Inits an ImageFile instance.

        Args:
            csv_file: csv file path containing the targets.
            transform: Images transforms.
            target_transform: Labels transform.
            loader: Images loader.
        '''

        # Loads data
        self.database_path = database_path
        training = training
        shrink_data = shrink_data
        data, self.names = self.from_file(manifest_path, training, shrink_data)

        if len(data) == 0:
            raise (RuntimeError(
                "Found 0 images in path: " + manifest_path + "\n" +
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        # Main data
        self.data = data

        # Saving data
        self.manifest_path = manifest_path

        self.input_shape = input_shape
        if not transform:
            transform = transforms(self.input_shape, training, augmentation)
        self.transform = transform

    def __getitem__(self, index):
        '''
        Returns image and target values for a given index.

        @param index Input index.

        @return The image and its respective target.
        '''

        if isinstance(index, str):
            idxs = [
                idx for idx, name in enumerate(self.names) if index in name
            ]
            if len(idxs) == 0:
                raise KeyError('Value not found')
            if len(idxs) > 1:
                raise KeyError('Non-unique key')

            data = self.data[idxs[0]]
        else:
            data = self.data[index]
        # Get single data
        (input_, bg_model), (target, roi) = data

        input_ = self.loader(input_)
        bg_model = self.loader(bg_model)
        target = self.loader(target, to_gray=True)
        roi = self.loader(roi, to_gray=True)

        # Transforming image
        if self.transform is not None:
            input_, target, roi = self.transform((input_, bg_model, target,
                                                  roi))
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
        imgs, targets, names = [], [], []

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
            names.append(input_path)

        return list(zip(imgs, targets)), names


def transforms(input_shape, training=False, augmentation=False):

    if not training and augmentation is not False:
        raise ValueError('Combinations of parameters not permitted. '
                         'training=False, augmentation=True')

    C, H, W = input_shape

    compose = [BinarizeTarget()]

    if C < 3:
        compose = compose + [Grayscale(1)]

    compose = compose + [
        Resize((H, W), (INTERPOLATION['BICUBIC'], INTERPOLATION['BICUBIC'],
                        INTERPOLATION['NEAREST'], INTERPOLATION['NEAREST'])),
    ]
    if augmentation is True:
        compose = compose + [RandomHorizontalFlip()]

    compose = compose + [MergeChannels(), ToTensor(), RoiCrop()]

    return vision_transforms.Compose(compose)
