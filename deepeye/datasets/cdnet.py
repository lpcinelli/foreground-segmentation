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

    def __init__(self, file_path, transform=None, target_transform=None):
        '''
        Inits an ImageFile instance.

        @param csv_file csv file path containing the targets.
        @param transform Images transforms.
        @param target_transform Labels transform.
        @param loader Images loader.
        '''

        # Loads data
        data = self.from_file(file_path)

        if len(data) == 0:
            raise(RuntimeError("Found 0 images in path: " + file_path + "\n" +
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        # Saving data
        self.file_path = file_path
        self.transform = transform
        self.target_transform = target_transform

        # Main data
        self.data = data

    def __getitem__(self, index):
        '''
        Returns image and target values for a given index.

        @param index Input index.

        @return The image and its respective target.
        '''

        # Get single data
        input_, target = self.data[index]

        input_ = self.loader(input_)

        # Transforming image
        if self.transform is not None:
            input_ = self.transform(input_)

        # Transforming target labels
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return
        return input_, target

    def __len__(self):
        '''
        Returns samples size.

        @return Current data number of samples.
        '''
        return len(self.data)

    def loader(self):
        return NotImplementedError

    def from_file(self, file_path):
        return NotImplementedError

def stats(args):
    from .preprocessing import get_mean_and_std

    datasets = ForestDataset(args.csv_file, args.imgs_dir,
                             transform=ToTensor())

    mean, std = get_mean_and_std(datasets)

    print('Mean: %s\nStd: %s' % (mean, std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Forest dataset preprocessing')
    parser.add_argument("csv_file", type=str)

    subparsers = parser.add_subparsers()

    parser_split = subparsers.add_parser('split')
    parser_split.add_argument("rate", nargs='+', type=float)
    parser_split.add_argument("--output", nargs='+', type=str, default=None)
    parser_split.set_defaults(func=split)

    parser_stats = subparsers.add_parser('stats')
    parser_stats.add_argument("--imgs-dir", type=str,
                              default='data/train-tif')
    parser_stats.set_defaults(func=stats)

    args = parser.parse_args()
    args.func(args)
