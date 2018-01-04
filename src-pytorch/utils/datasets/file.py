import numpy as np  # numpy library

import torch
import torch.utils.data as data  # Torch dataset class

from ..utils.img_utils import IMG_EXTENSIONS


class DataFile(data.Dataset):
    '''
    Reads data from a file
    '''

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
