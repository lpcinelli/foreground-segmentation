from torch import nn
from torch.nn import init
import torch.nn.functional as F

import math
import warnings

from ..utils.generic_utils import compute_pad_size, compute_output_size

__all__ = [
    'LeNetDecoder', 'lenetdecoder', 'LeNetDeconv', 'lenetdeconv',
    'LeNetUpSample', 'lenetupsample'
]

UPSAMPLING_MODES = ['nearest', 'bilinear']


class LeNet(nn.Module):
    """ Base Lenet(ish) network that compresses information
        and on which all variants are built upon.
    """

    def __init__(self, input_shape):
        super(LeNet, self).__init__()

        C, H, W = input_shape

        self.block1 = nn.Sequential(
            nn.Conv2d(C, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=3, padding=1),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
        )

        self.pad1 = compute_pad_size((H, W), (3, 3), (2, 2))
        self.pool1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=self.pad1,
            return_indices=True,
            ceil_mode=True)

        out1 = compute_output_size((H, W), (3, 3), (2, 2), self.pad1)
        self.pad2 = compute_pad_size(out1, (3, 3), (2, 2))
        self.pool2 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=self.pad2,
            return_indices=True,
            ceil_mode=True)

        out2 = compute_output_size(out1, (3, 3), (2, 2), self.pad2)
        self.pad3 = compute_pad_size(out2, (3, 3), (2, 2))
        self.pool3 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=self.pad3,
            return_indices=True,
            ceil_mode=True)

    def forward(self, x):
        x = self.block1(x)
        x, pool1_idx = self.pool1(x)

        x = self.block2(x)
        x, pool2_idx = self.pool2(x)

        x = self.block3(x)
        x, pool3_idx = self.pool3(x)

        return x, (pool1_idx, pool2_idx, pool3_idx)


class LeNetUpSample(nn.Module):
    """ Upsampled variant of LeNet
        This is a simpler variant which does not rely on complex
        reconstruction methods, instead it employs a naive
        interpolation (nearest neighbor/bilinear) at the very end
        to recover the original size.
    """

    def __init__(self, input_shape, upsampling='bilinear'):
        super(LeNetUpSample, self).__init__()

        C, H, W = input_shape

        self.base = LeNet(input_shape)

        self.classifier = nn.Sequential(
            nn.Conv2d(120, 1, kernel_size=1, padding=0),
            nn.Upsample((H, W), mode=upsampling))

    def forward(self, x):
        # Extracting the features
        x, _ = self.base(x)

        # Projecting to probability map
        x = self.classifier(x)

        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


class LeNetDeconv(nn.Module):
    """ Upsampled variant of LeNet
        This variant is based on DeconvNet [Noh], using
        transposed convolutions (aka deconvolutions) to
        rebuild the feature map. However, here there is no
        unpooling to recover the map size and the transp.
        conv. also assumes this role, simultaneously enlarging
        and populating the feature maps.
    """

    def __init__(self, input_shape, upsampling='bilinear'):
        super(LeNetDeconv, self).__init__()

        C, H, W = input_shape

        self.base = LeNet(input_shape)

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(120, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0),
            nn.Upsample((H, W), mode=upsampling))

    def forward(self, x):
        # Extracting the features
        x, _ = self.base(x)

        # Reconstructing feature map
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Projecting to probability map
        x = self.classifier(x)

        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


class LeNetDecoder(nn.Module):
    """ Upsampled variant of LeNet
        This variant is based on SegNet [Badrinarayanan] and
        reconstructs the image through successive unpoolings
        to enlarge the feature map size and 2D convolutions
        to populate them.
    """

    def __init__(self, input_shape, upsampling='bilinear'):
        super(LeNetDecoder, self).__init__()

        C, H, W = input_shape

        self.base = LeNet(input_shape)

        self.block1 = nn.Sequential(
            nn.Conv2d(120, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0),
            nn.Upsample((H, W), mode=upsampling))

        self.unpool1 = nn.MaxUnpool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=self.base.pad3)
        self.unpool2 = nn.MaxUnpool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=self.base.pad2)
        self.unpool3 = nn.MaxUnpool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=self.base.pad1)

        self._weights_init()

    def forward(self, x):
        # Extracting the features
        x, pool_idx = self.base(x)

        # Reconstructing feature map
        x = self.unpool1(x, pool_idx[-1])
        x = self.block1(x)

        x = self.unpool2(x, pool_idx[-2])
        x = self.block2(x)

        x = self.unpool3(x, pool_idx[-3])
        x = self.block3(x)

        # Projecting to probability map
        x = self.classifier(x)

        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


def lenetupsample(pretrained=False, **kwargs):
    if pretrained:
        warnings.warn('No pretrained model available. ' +
                      'Falling back to pretrained=False')
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    upsampling = kwargs.pop('upsampling', None)
    if upsampling is None:
        upsampling = 'bilinear'
    elif upsampling not in UPSAMPLING_MODES:
        raise ValueError(
            'Invalid upsampling mode. Options are {}'.format(UPSAMPLING_MODES))

    return LeNetUpSample(input_shape, upsampling)


def lenetdeconv(pretrained=False, **kwargs):
    if pretrained:
        warnings.warn('No pretrained model available. ' +
                      'Falling back to pretrained=False')
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    upsampling = kwargs.pop('upsampling', None)
    if upsampling is None:
        upsampling = 'bilinear'
    elif upsampling not in UPSAMPLING_MODES:
        raise ValueError(
            'Invalid upsampling mode. Options are {}'.format(UPSAMPLING_MODES))

    return LeNetDeconv(input_shape, upsampling)


def lenetdecoder(pretrained=False, **kwargs):
    if pretrained:
        warnings.warn('No pretrained model available. ' +
                      'Falling back to pretrained=False')
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    upsampling = kwargs.pop('upsampling', None)
    if upsampling is None:
        upsampling = 'bilinear'
    elif upsampling not in UPSAMPLING_MODES:
        raise ValueError(
            'Invalid upsampling mode. Options are {}'.format(UPSAMPLING_MODES))

    return LeNetDecoder(input_shape, upsampling)


def lenetdilate(pretrained=False, **kwargs):
    if pretrained:
        warnings.warn('No pretrained model available. ' +
                      'Falling back to pretrained=False')
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    upsampling = kwargs.pop('upsampling', None)
    if upsampling is None:
        upsampling = 'bilinear'
    elif upsampling not in UPSAMPLING_MODES:
        raise ValueError(
            'Invalid upsampling mode. Options are {}'.format(UPSAMPLING_MODES))

    raise NotImplementedError
    # return LeNetDilate(input_shape, upsampling)
