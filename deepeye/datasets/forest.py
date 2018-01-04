import os
import numpy as np
import argparse
import warnings
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.exceptions import NotFittedError

import torchvision.transforms as vision_transforms

from .file import DataFile

from ..utils.img_utils import default_loader
from ..utils import img_utils
from ..transforms import ToTensor

MEAN3 = [0.3114, 0.3405, 0.2988]
STD3 = [0.1672, 0.1438, 0.1373]
MEAN4 = [0.0742, 0.0632, 0.0450, 0.0957]
STD4 =  [0.0278, 0.0246, 0.0258, 0.0286]
W, H = 256, 256


class ForestDataset(DataFile):
    '''
    Dataset wrapping images and target labels for Kaggle contest
    Planet: Understanding the Amazon from Space
    '''

    def __init__(self, csv_file, imgs_dir, img_ext=None, exceptions=None,
                 transform=None, target_transform=None, **kwargs):

        self.exceptions = exceptions
        self.binarizer = MultiLabelBinarizer(classes=self.classes)

        self.imgs_dir = imgs_dir
        self.img_ext = img_ext or ('.jpg' if 'jpg' in imgs_dir else '.tif')

        self.num_channels = 4 if '.tif' in self.img_ext else 3

        self.input_shape = kwargs.get('input_shape',
                                      (self.num_channels, None, None)
                                      )

        if not transform:
            transform = transforms(self.num_channels, **kwargs)
            self.input_shape = default_input_shape(self.num_channels,
                                                   **kwargs)

        super(ForestDataset, self).__init__(csv_file,
                                            transform,
                                            target_transform)

    def from_file(self, csv_file):
        # Opening file
        dataset = pd.read_csv(csv_file)
        imgs, targets = [], []

        for i, row in dataset.iterrows():
            img, target = row['image_name'], row['tags']

            path = os.path.join(self.imgs_dir, img + self.img_ext)

            if img_utils.is_image_file(path) and os.path.isfile(path):
                target = [c for c in target.split(' ') \
                                if c not in self.exceptions]
                if target:
                    imgs.append(path)
                    targets.append(target)

        targets = self.binarizer.fit_transform(targets).astype(np.float32)

        return list(zip(np.array(imgs), targets))

    def loader(self, path):
        return default_loader(path)

    @property
    def classes(self):
        return [c for c in ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
                'blow_down', 'clear', 'cloudy', 'conventional_mine',
                'cultivation', 'habitation', 'haze', 'partly_cloudy',
                'primary', 'road', 'selective_logging', 'slash_burn',
                'water'] if c not in self.exceptions]


def normalize(num_channels):
    if num_channels == 3:
        return vision_transforms.Normalize(MEAN3, STD3)
    elif num_channels == 4:
        return vision_transforms.Normalize(MEAN4, STD4)

    raise ValueError('Wrong number of channels: %d' % num_channels)


def transforms(num_channels, training=False, augmentation=False):

    if not training and augmentation:
        raise ValueError('Combinations of parameters not permitted. '
                         'training=False, augmentation=True')

    compose = [ToTensor(), normalize(num_channels)]

    if not augmentation:
        compose = [vision_transforms.CenterCrop(224)] + compose
    else:
        compose = [vision_transforms.RandomSizedCrop(224),
                   vision_transforms.RandomHorizontalFlip()] + compose

    if not training:
        compose = [vision_transforms.Scale(W)] + compose

    return vision_transforms.Compose(compose)


def default_input_shape(num_channels, training=False, augmentation=False):
    return (num_channels, 224, 224)


def split(args):
    rate = np.array(args.rate)

    if np.sum(rate) <= 0 or np.sum(rate) > 1:
        raise ValueError('rate sum must be in (0,1)')

    if np.alltrue(np.sort(rate) != rate):
        raise ValueError('rate must be in increasing order')

    filename, ext = os.path.splitext(args.csv_file)

    outputs = args.output or ['%s-split-%d%s' % (filename, i, ext)
                              for i in range(rate.size + 1)]
    if rate.size + 1 != len(outputs):
        raise ValueError('len(size) + 1 differs from len(output)')

    dataset = pd.read_csv(args.csv_file, index_col='image_name')
    print('Total samples: %d' % len(dataset))

    indices = (rate*(len(dataset))).astype(np.int)

    datasets = np.split(dataset, indices)

    rate = np.stack([rate, [1.]])

    for i, (dt, out) in enumerate(zip(datasets, outputs)):

        print('\tSaving to %s' % out)
        print('\t\tSplit rate: %f' % rate[i])
        print('\t\tSamples: %d' % len(dt))
        dt.to_csv(out)


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
