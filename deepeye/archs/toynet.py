import math
import warnings

import torch.nn.functional as F
from torch import nn
from torch.nn import init

__all__ = ['ToyNet', 'toynet']


class ToyNet(nn.Module):
    """ Toy network

    Hint: better works with images with W:32, H:32
    """

    def __init__(self, input_shape, num_classes=17):
        super(ToyNet, self).__init__()

        C, W, H = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        self.classifier = nn.Sequential(
            nn.Linear(int((W * H * 64) / 16), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes))

        self._weights_init()

    def forward(self, x):
        # Extracting the features
        x = self.features(x)

        # Flatten Layer
        x = x.view(x.size(0), -1)

        # Projecting to num_classes
        x = self.classifier(x)

        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


def toynet(pretrained=False, **kwargs):
    if pretrained:
        warnings.warn('No pretrained model available. ' +
                      'Falling back to pretrained=True')
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    return ToyNet(input_shape, **kwargs)
