import warnings

import torch.nn.functional as F
from torch import nn
from torch.nn import init

__all__ = [
    'LeNetDecoder', 'lenetdecoder', 'LeNetDeconv', 'lenetdeconv',
    'LeNetUpSample', 'lenetupsample'
]

UPSAMPLING_MODES = ['nearest', 'bilinear']


class LeNet(nn.Module):
    """ Base Lenet(ish) network that compresses information and on which all
    variants are built upon.
    """

    def __init__(self, input_shape, return_indices=False, return_sizes=False):
        """
        Arguments:
            input_shape {tuple} --  nb of channels (C), height (H), width (W)

        Keyword Arguments:
            return_indices {bool} -- return the pooling indices or not
            (default: {False})
            return_sizes {bool} -- whether to return the input size after each
            block or not (default: {False})
        """

        super(LeNet, self).__init__()

        C, _, _ = input_shape
        self.return_indices = return_indices
        self.return_sizes = return_sizes

        self.block1 = nn.Sequential(
            nn.Conv2d(C, 6, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            return_indices=True,
            ceil_mode=True)

        self.pool2 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            return_indices=True,
            ceil_mode=True)

        self.pool3 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            return_indices=True,
            ceil_mode=True)

    def forward(self, x):
        x = self.block1(x)
        size1 = x.size()
        x, pool1_idx = self.pool1(x)

        x = self.block2(x)
        size2 = x.size()
        x, pool2_idx = self.pool2(x)

        x = self.block3(x)
        size3 = x.size()
        x, pool3_idx = self.pool3(x)

        out = [x]

        if self.return_indices == True:
            out += [(pool1_idx, pool2_idx, pool3_idx)]

        if self.return_sizes == True:
            out += [(size1, size2, size3)]

        return out


class LeNetUpSample(nn.Module):
    """ Upsampled variant of LeNet.
        This is a simpler variant which does not rely on complex reconstruction
        methods, instead it employs a naive interpolation at the very end to
        recover the original size. Options are: bilinear and nearest neighbor
    """

    def __init__(self, input_shape, upsampling='bilinear'):
        """
        Arguments:
            input_shape {tuple} --  nb of channels (C), height (H), width (W)

        Keyword Arguments:
            upsampling {str} - interpolation type (default: {'bilinear'})
        """

        super(LeNetUpSample, self).__init__()

        _, H, W = input_shape
        self.upsampling = upsampling
        self.base = LeNet(input_shape)

        self.classifier = nn.Sequential(
            nn.Conv2d(120, 1, kernel_size=1, padding=0))

    def forward(self, x):
        orig_size = x.size()[-2:]

        # Extracting the features
        x = self.base(x)[0]

        # Projecting to probability map
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size, mode=self.upsampling)

        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


class LeNetDeconv(nn.Module):
    """ Upsampled variant of LeNet
        This variant is based on DeconvNet [Noh], using  transposed convolutions
        (aka deconvolutions) to rebuild the feature map. However, here there is
        no unpooling to recover the map size and the transp. conv. also assumes
        this role, simultaneously enlarging and populating the feature maps.
    """

    def __init__(self, input_shape):
        """
        Arguments:
            input_shape {tuple} --  nb of channels (C), height (H), width (W)
        """

        super(LeNetDeconv, self).__init__()

        _, H, W = input_shape
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
            nn.Conv2d(6, 1, kernel_size=1, padding=0))

    def forward(self, x):
        orig_size = x.size()[-2:]

        # Extracting the features
        x = self.base(x)[0]

        # Reconstructing feature map
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Projecting to probability map
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size, mode='bilinear')

        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


class LeNetDecoder(nn.Module):
    """ Upsampled variant of LeNet
        This variant is based on SegNet [Badrinarayanan] and reconstructs the
        mage through successive unpoolings to enlarge the feature map size and
        2D convolutions to populate them.
    """

    def __init__(self, input_shape, upsampling='bilinear'):
        """
        Arguments:
            input_shape {tuple} --  nb of channels (C), height (H), width (W)

        Keyword Arguments:
            upsampling {str} - interpolation type (default: {'bilinear'})
        """
        super(LeNetDecoder, self).__init__()

        _, H, W = input_shape
        self.upsampling = upsampling
        self.base = LeNet(input_shape, return_indices=True, return_sizes=True)

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
            nn.Conv2d(6, 1, kernel_size=1, padding=0))

        self.unpool1 = nn.MaxUnpool2d(kernel_size=(3, 3), stride=(2, 2))
        self.unpool2 = nn.MaxUnpool2d(kernel_size=(3, 3), stride=(2, 2))
        self.unpool3 = nn.MaxUnpool2d(kernel_size=(3, 3), stride=(2, 2))

        self._weights_init()

    def forward(self, x):
        orig_size = x.size()[-2:]

        # Extracting the features
        x, pool_idx, sizes = self.base(x)

        # Reconstructing feature map
        x = self.unpool1(x, pool_idx[-1], output_size=sizes[-1])
        x = self.block1(x)

        x = self.unpool2(x, pool_idx[-2], output_size=sizes[-2])
        x = self.block2(x)

        x = self.unpool3(x, pool_idx[-3], output_size=sizes[-3])
        x = self.block3(x)

        # Projecting to probability map
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size, mode=self.upsampling)

        return x

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)


def lenetupsample(pretrained=False, **kwargs):
    """Helper function to instantiate a lenet with reconstruction by upsampling

    Keyword Arguments:
        pretrained {bool} -- Use pretrained weights (default: {False})

    Returns:
        LeNetUpSample -- network model of lenet with the desired reconstruction
        mode
    """

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
    """Helper function to instantiate a lenet with reconstruction by
    deconvolution

    Keyword Arguments:
        pretrained {bool} -- Use pretrained weights (default: {False})

    Raises:
        ValueError -- [description]
        ValueError -- [description]

    Returns:
        LeNetDeconv -- the desired lenet LeNetDeconv
    """

    if pretrained:
        warnings.warn('No pretrained model available. ' +
                      'Falling back to pretrained=False')
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    return LeNetDeconv(input_shape)


def lenetdecoder(pretrained=False, **kwargs):
    """Helper function to instantiate a lenet with reconstruction by "decoder",
    stack of convolution and bilinear upsampling

    Keyword Arguments:
        pretrained {bool} -- Use pretrained weights (default: {False})

    Returns:
        LeNetDecoder -- the chosen model
    """

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
