import os
import numpy as np
import argparse
import warnings
import pandas as pd
import glob2 as glob

import torch
import torch.utils.data as data  # Torch dataset class

import torchvision.transforms as vision_transforms
from torchvision.datasets.folder import default_loader

from .file import DataFile

from ..transforms import ToTensor


class CDNetDataset(data.Dataset):

    def __init__(self, manifest_path, database_path, transform=None):
        '''
        Inits an ImageFile instance.

        @param csv_file csv file path containing the targets.
        @param transform Images transforms.
        @param target_transform Labels transform.
        @param loader Images loader.
        '''

        # Loads data
        self.database_path = database_path
        data = self.from_file(manifest_path)

        if len(data) == 0:
            raise(RuntimeError("Found 0 images in path: " + manifest_path + "\n" +
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        # Main data
        self.data = data

        # Saving data
        self.manifest_path = manifest_path
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
        target = self.loader(target)
        roi = self.loader(roi)

        # Transforming image
        if self.transform is not None:
            input_, target, roi = self.transform((input_, bg_model, target, roi))

        # Return
        return input_, bg_model, target, roi

    def __len__(self):
        '''
        Returns samples size.

        @return Current data number of samples.
        '''
        return len(self.data)

    def loader(self, path):
        return default_loader(path)

    def from_file(self, csv_file):
        # Opening file
        dataset = pd.read_csv(csv_file)
        imgs, targets = [], []

        for i, row in dataset.iterrows():
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

def transforms(num_channels, training=False, augmentation=False):

    if not training and augmentation:
        raise ValueError('Combinations of parameters not permitted. '
                         'training=False, augmentation=True')

    compose = [ToTensor()]

    if not augmentation:
        compose = [vision_transforms.CenterCrop(224)] + compose
    else:
        compose = [vision_transforms.RandomSizedCrop(224),
                   vision_transforms.RandomHorizontalFlip()] + compose

    if not training:
        compose = [vision_transforms.Scale(W)] + compose

    return vision_transforms.Compose(compose)

