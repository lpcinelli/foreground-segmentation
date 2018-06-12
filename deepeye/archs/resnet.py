import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from ..utils.generic_utils import rgb2gray

__all__ = [
    'resnet', 'resnet18', 'resnet20', 'resnet32', 'resnet34', 'resnet50',
    'resnet101', 'resnet152'
]

UPSAMPLING_MODES = ['upsample', 'deconv', 'deconvshallow', 'decodeshallow']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def identity_up(x, pool_idx, output_size):
    return x


def identity_down(x):
    return (x, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 residue=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residue = residue

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residue == True:
            out += residual

        out = self.relu(out)

        return out


class BasicBlockUp(nn.Module):
    expansion = 1 / 2

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 stride=1,
                 upsample=None,
                 residue=True,
                 mirror=True):
        super(BasicBlockUp, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            inplanes,
            inplanes,
            kernel_size=3,
            stride=1 if mirror is True else stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride if mirror is True else 1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.upsample = upsample
        self.stride = stride
        self.residue = residue

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        if self.residue == True:
            out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 midplanes,
                 stride=1,
                 downsample=None,
                 residue=True):
        super(Bottleneck, self).__init__()
        outplanes = midplanes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(
            midplanes,
            midplanes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.residue = residue

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residue == True:
            out += residual

        out = self.relu(out)

        return out


class BottleneckUp(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 midplanes,
                 outplanes,
                 stride=1,
                 upsample=None,
                 residue=True):
        super(BottleneckUp, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.ConvTranspose2d(
            midplanes,
            midplanes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        self.residue = residue

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        if self.residue == True:
            out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 input_shape,
                 block,
                 layers,
                 num_classes=1,
                 dilation=1,
                 dilation_growth=2,
                 return_indices=False,
                 return_sizes=False,
                 skip_connection=False,
                 preblock=None,
                 inplanes=64):
        super(ResNet, self).__init__()

        C, _, _ = input_shape
        self.inplanes = inplanes
        self.return_indices = return_indices
        self.return_sizes = return_sizes
        self.skip_connection = skip_connection
        self.nb_blocks = len(layers)
        std_block = block

        if preblock is None:
            self.conv1 = nn.Conv2d(
                C,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            maxpool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                return_indices=True,
                ceil_mode=True)
        else:
            # How should preblock be parsed:
            # {conv:[inplanes, kernel_size, stride, padding],
            #  pool:[kernel_size, stride, padding]}

            assert('conv' in preblock and len (preblock['conv']) is 4),\
                '\'conv\' key in preblock settings should have '\
                '[inplanes, kernel_size, stride, padding] as params'
            assert('pool' in preblock and
                    (preblock['pool'] is None or len (preblock['pool']) is 3)),\
                '\'pool\' key in preblock settings should have '\
                '[kernel_size, stride, padding] as params or be None'

            self.inplanes = preblock['conv'][0]
            self.conv1 = nn.Conv2d(C, *preblock['conv'], bias=False)
            if preblock['pool'] is not None:
                maxpool = nn.MaxPool2d(
                    *preblock['pool'], return_indices=True, ceil_mode=True)
            else:
                maxpool = identity_down

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = maxpool

        for i, layer in enumerate(layers):

            if isinstance(layer, int):
                blocks = layer
                channels = 2**(6 + i)
                stride = 2 if i > 0 else 1
                dilate = dilation / dilation_growth**(self.nb_blocks - (i + 1))
            elif isinstance(layer, list) and isinstance(layer[1], dict):
                blocks = layer[0]
                channels = layer[1].get('depth', 2**(6 + i))
                stride = layer[1].get('stride', 2 if i > 0 else 1)
                block = layer[1].get('block', std_block)
                dilate = layer[1].get('dilation', dilation / dilation_growth**
                                      (self.nb_blocks - (i + 1)))
            else:
                raise ValueError('Elements of {} should either be int '.format(
                    layers) + 'or list of [int, dict]')

            self.__dict__['_modules']['layer{}'.format(
                i + 1)] = self._make_down_layer(
                    block, channels, blocks, stride=stride).apply(
                        partial(self._nostride_dilate, dilate=dilate))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_down_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and dilate > 1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                min_dilate = int(max(1, dilate // 2))
                if m.kernel_size == (3, 3):
                    m.dilation = (min_dilate, min_dilate)
                    m.padding = (min_dilate, min_dilate)
            # other convoluions
            else:
                min_dilate = int(max(1, dilate))
                if m.kernel_size == (3, 3):
                    m.dilation = (min_dilate, min_dilate)
                    m.padding = (min_dilate, min_dilate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        size1 = x.size()
        x, pool_idx = self.maxpool(x)

        size2 = x.size()

        if self.skip_connection == True:
            intermediate_maps = [x]
            for i in range(self.nb_blocks):
                intermediate_maps.append(
                    self.__dict__['_modules']['layer{}'.format(i + 1)](
                        intermediate_maps[i]))
            out = [intermediate_maps]
        else:
            for i in range(self.nb_blocks):
                x = self.__dict__['_modules']['layer{}'.format(i + 1)](x)
            out = [[x]]

        if self.return_indices == True:
            out += [pool_idx]

        if self.return_sizes == True:
            out += [(size1, size2)]

        return out


class ResNetUpSample(nn.Module):
    """ Upsampled variant of ResNet
        This is a simpler variant which does not rely on complex
        reconstruction methods, instead it employs a naive
        interpolation (nearest neighbor/bilinear) at the very end
        to recover the original size.
    """

    def __init__(self, input_shape, block, layers, num_classes=1, **kwargs):
        super(ResNetUpSample, self).__init__()

        self.base = ResNet(
            # input_shape, block, layers, num_classes=1, dilation=dilation, inplanes=inplanes)
            input_shape,
            block,
            layers,
            num_classes=num_classes,
            **kwargs)
        last_layer = [
            obj for obj in self.base.modules() if isinstance(obj, nn.Conv2d)
        ][-1]

        self.classifier = nn.Sequential(
            nn.Conv2d(
                last_layer.out_channels, num_classes, kernel_size=1,
                padding=0))

    def forward(self, x):
        orig_size = x.size()

        # Extracting the features
        x = self.base(x)[0][0]

        # Projecting to belief map
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size[-2:], mode='bilinear')

        return x


class ResNetDeconv(nn.Module):
    """ Upsampled variant of ResNet.
        This variant is based on DeconvNet [Noh], using
        transposed convolutions (aka deconvolutions) to
        rebuild the feature map. However, here there is no
        unpooling to recover the map size just as there is no
        pooling on the downstream and so the transp. conv.
        also assumes this role, simultaneously enlarging
        and populating the feature maps.
        This deconvolutional decoder (practically) doubles
        model size

        block_up can either be ['BasicBlockUp', 'BottleneckUp']
    """

    def __init__(self,
                 input_shape,
                 block,
                 layers,
                 num_classes=1,
                 dilation=1,
                 dilation_growth=2,
                 block_up=None,
                 layers_up=None,
                 skip_connection=False,
                 inplanes=None,
                 preblock=None):
        super(ResNetDeconv, self).__init__()

        self.skip_connection = skip_connection

        if block_up is None:
            block_up = BasicBlockUp if block.__name__ == 'BasicBlock' \
                                   else BottleneckUp
        elif (block_up is not BasicBlockUp) and (block_up is not BottleneckUp):
            raise ValueError('Invalid upsampling block {}'.format(block_up))

        if layers_up is None:
            layers_up = layers[::-1]
        self.nb_blocks = len(layers_up)

        self.base = ResNet(
            input_shape,
            block,
            layers,
            num_classes=num_classes,
            dilation=dilation,
            dilation_growth=dilation_growth,
            return_indices=True,
            return_sizes=True,
            skip_connection=skip_connection,
            preblock=preblock)

        last_layer = [
            obj for obj in self.base.modules() if isinstance(obj, nn.Conv2d)
        ][-1]
        self.inplanes = last_layer.out_channels

        for i, layer in enumerate(layers_up):

            if isinstance(layer, int):
                blocks = layer
                channels = 2**(6 + (len(layers_up) - 1) - i)
                stride = 2 if i < (len(layers_up) - 1) else 1
                dilate = dilation / dilation_growth**i
            elif isinstance(layer, list) and isinstance(layer[1], dict):
                blocks = layer[0]
                channels = layer[1].get('depth', 2**(6 + i))
                stride = layer[1].get('stride', 2
                                      if i < (len(layers_up) - 1) else 1)
                block_up = layer[1].get('block', block_up)
                dilate = layer[1].get('dilation',
                                      dilation / dilation_growth**i)
            else:
                raise ValueError('Elements of {} should either be int '.format(
                    layers) + 'or list of [int, dict]')

            outplanes = 64 if i == (len(layers_up) - 1) else None

            self.__dict__['_modules']['layer{}'.format(
                i + 1)] = self._make_up_layer(
                    block_up,
                    channels,
                    blocks,
                    stride=stride,
                    outplanes=outplanes).apply(
                        partial(self._nostride_dilate, dilate=dilate))

        # this deconv/unpool should be condicioned to conv1 (preblock)
        # in ResNet which is now configurable
        if preblock is None:
            self.unpool = nn.MaxUnpool2d(
                kernel_size=(3, 3), stride=(2, 2), padding=1)
            self.deconv = nn.ConvTranspose2d(
                self.inplanes,
                2,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
        else:
            if preblock['pool'] is None:
                self.unpool = identity_up
            else:
                self.unpool = nn.MaxUnpool2d(*preblock['pool'])

            self.deconv = nn.ConvTranspose2d(
                self.inplanes, 2, *preblock['conv'][1:], bias=False)

        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(2, num_classes, kernel_size=1, padding=0)

    def _make_up_layer(self,
                       block,
                       planes,
                       blocks,
                       outplanes=None,
                       stride=1,
                       mirror=True):
        upsample = None
        if outplanes is None:
            outplanes = int(planes * block.expansion)

        if stride != 1 or self.inplanes != outplanes:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    outplanes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(outplanes),
            )

        if mirror is False:
            raise NotImplementedError

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.inplanes))
        layers.append(
            block(self.inplanes, planes, outplanes, stride, upsample))
        self.inplanes = outplanes

        return nn.Sequential(*layers)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and dilate > 1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                min_dilate = int(max(1, dilate // 2))
                if m.kernel_size == (3, 3):
                    m.dilation = (min_dilate, min_dilate)
                    m.padding = (min_dilate, min_dilate)
            # other convoluions
            else:
                min_dilate = int(max(1, dilate))
                if m.kernel_size == (3, 3):
                    m.dilation = (min_dilate, min_dilate)
                    m.padding = (min_dilate, min_dilate)

    def forward(self, x):
        orig_size = x.size()

        # Extracting the features
        features, pool_idx, size = self.base(x)

        # Reconstructing feature map
        x = self.layer1(features[-1])

        if self.skip_connection == True:
            for i in range(1, self.nb_blocks):
                x = self.__dict__['_modules']['layer{}'.format(i + 1)](
                    x + features[-(i + 1)])
        else:
            for i in range(1, self.nb_blocks):
                x = self.__dict__['_modules']['layer{}'.format(i + 1)](x)

        x = F.upsample(x, size=size[-1][-2:], mode='bilinear')
        x = self.unpool(x, pool_idx, output_size=size[-2])

        # this deconv should be condicioned to conv1 (preblock) in ResNet
        # which is now configurable
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Projecting to belief map
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size[-2:], mode='bilinear')

        return x


class ResNetDeconvShallow(nn.Module):
    """ Upsampled variant of ResNet
        This is a simpler variant which does not rely on complex
        reconstruction methods, instead it employs a naive
        interpolation (nearest neighbor/bilinear) at the very end
        to recover the original size.
    """

    def __init__(self,
                 input_shape,
                 block,
                 layers,
                 num_classes=1,
                 lin_recons=False,
                 **kwargs):
        super(ResNetDeconvShallow, self).__init__()

        self.base = ResNet(
            input_shape, block, layers, num_classes=num_classes, **kwargs)
        last_layer = [
            obj for obj in self.base.modules() if isinstance(obj, nn.Conv2d)
        ][-1]

        nb_channels = last_layer.out_channels

        conv_block = [
            nn.Conv2d(nb_channels, 2 * nb_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(2 * nb_channels),
        ]
        deconv_block = [
            nn.ConvTranspose2d(
                2 * nb_channels,
                nb_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.BatchNorm2d(nb_channels),
        ]

        if lin_recons is False:
            conv_block.append(nn.ReLU(inplace=True))
            deconv_block.append(nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(*conv_block)
        self.deconv1 = nn.Sequential(*deconv_block)

        self.classifier = nn.ConvTranspose2d(
            nb_channels,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1)

    def forward(self, x):
        orig_size = x.size()

        # Extracting the features
        x = self.base(x)[0][0]

        # (Mid)classifier (FC)
        x = self.conv1(x)

        # Reconstruct belief map
        x = self.deconv1(x)
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size[-2:], mode='bilinear')

        return x


class ResNetDecodeShallow(nn.Module):
    """ Upsampled variant of ResNet
        This is a simpler variant which does not rely on complex
        reconstruction methods, instead it employs a naive
        interpolation (nearest neighbor/bilinear) at the very end
        to recover the original size.
    """

    def __init__(self,
                 input_shape,
                 block,
                 layers,
                 num_classes=1,
                 dilation=1,
                 upsampling='bilinear',
                 preblock=None):
        super(ResNetDecodeShallow, self).__init__()

        _, H, W = input_shape
        self.upsampling = upsampling
        self.base = ResNet(
            input_shape,
            block,
            layers,
            num_classes=1,
            dilation=dilation,
            preblock=preblock)
        last_layer = [
            obj for obj in self.base.modules() if isinstance(obj, nn.Conv2d)
        ][-1]

        self.inplanes = last_layer.out_channels

        self.decoder = self._make_decoder(blocks=2)
        self.classifier = nn.Sequential(
            nn.Conv2d(self.inplanes, num_classes, kernel_size=1, padding=0))

    def _make_decoder(self, blocks):
        def _decoder_layer(inplane, outplane, scale_factor, mode):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode),
                nn.Conv2d(
                    inplane, outplane, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outplane),
                nn.ReLU(inplace=True))

        layers = []
        for i in range(blocks):
            layers.append(
                _decoder_layer(
                    self.inplanes,
                    self.inplanes // 2,
                    scale_factor=2,
                    mode=self.upsampling))
            self.inplanes //= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        orig_size = x.size()

        # Extracting the features
        x = self.base(x)[0][0]

        # Reconstruct image
        x = self.decoder(x)

        # Projecting to belief map
        x = self.classifier(x)

        x = F.upsample(x, size=orig_size[-2:], mode=self.upsampling)

        return x


# def linknet(pretrained=False, **kwargs):
#     input_shape = kwargs.pop('input_shape', None)
#     dilation = kwargs.pop('dilation', 1)
#     if not input_shape:
#         raise ValueError('input_shape is required')
#     if input_shape[1:] != (224, 224):
#         raise NotImplementedError
#     if up_mode not in UPSAMPLING_MODES:
#         raise ValueError(
#             'Incorrect reconstruction option ({}). Available options are {}'\
#             .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

#     model = ResNetDeconv(
#         input_shape,
#         BasicBlock,
#         [2, 2, 2, 2],
#         block_up=BottleneckUp,
#         # decoder=[(nb of block_up per block, block stride), ...]
#         decoder=[(1, 1), (1, 2), (1, 2), (1, 2)],
#         dilation=dilation,
#         skip_connection=True)

#     if pretrained:
#         _load_weights(model.base.load_state_dict, 'resnet18', C)
#     return model

MODELS = {
    k.lower(): v
    for k, v in globals().items() if k.startswith('ResNet') and len(k) > 6
}


def resnet(**kwargs):
    """Constructs a generic ResNet model.
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()

    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    layers = kwargs.pop('layers', None)
    if layers is None:
        raise ValueError(
            'layers should be a list with the number of layers per block '\
            'and (optionally) other layer configurations')
    block_type = eval(kwargs.pop('block_type', 'BasicBlock'))

    model = MODELS[''.join(['resnet', up_mode])](input_shape, block_type,
                                                 layers, **kwargs)
    return model


def resnet20(**kwargs):
    """Constructs a CIFAR10 ResNet-20 model.
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()
    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    layers = [[3, {
        'depth': 16,
        'stride': 1
    }], [3, {
        'depth': 32,
        'stride': 2
    }], [3, {
        'depth': 64,
        'stride': 2
    }]]

    preblock = {'conv': [16, 3, 1, 1], 'pool': None}

    model = MODELS[''.join(['resnet', up_mode])](
        input_shape, BasicBlock, layers, preblock=preblock, **kwargs)

    return model


def resnet32(**kwargs):
    """Constructs a CIFAR10 ResNet-32 model.
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()
    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    layers = [[5, {
        'depth': 16,
        'stride': 1
    }], [5, {
        'depth': 32,
        'stride': 2
    }], [5, {
        'depth': 64,
        'stride': 2
    }]]

    preblock = {'conv': [16, 3, 1, 1], 'pool': None}

    model = MODELS[''.join(['resnet', up_mode])](
        input_shape, BasicBlock, layers, preblock=preblock, **kwargs)

    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()
    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    model = MODELS[''.join(['resnet', up_mode])](input_shape, BasicBlock,
                                                 [2, 2, 2, 2], **kwargs)

    if pretrained:
        _load_weights(model.base.load_state_dict, 'resnet18', input_shape[0])
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()
    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    model = MODELS[''.join(['resnet', up_mode])](input_shape, BasicBlock,
                                                 [3, 4, 6, 3], **kwargs)

    if pretrained:
        _load_weights(model.base.load_state_dict, 'resnet34', input_shape[0])
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()
    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    model = MODELS[''.join(['resnet', up_mode])](input_shape, Bottleneck,
                                                 [3, 4, 6, 3], **kwargs)

    if pretrained:
        _load_weights(model.base.load_state_dict, 'resnet50', input_shape[0])
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()
    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    model = MODELS[''.join(['resnet', up_mode])](input_shape, Bottleneck,
                                                 [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_weights(model.base.load_state_dict, 'resnet101', input_shape[0])
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')

    up_mode = kwargs.pop('up_mode', 'None').lower()
    if up_mode not in UPSAMPLING_MODES:
        raise ValueError(
            'Incorrect reconstruction option ({}). Available options are {}'\
            .format(up_mode, ", ".join(x for x in UPSAMPLING_MODES)))

    model = MODELS[''.join(['resnet', up_mode])](input_shape, Bottleneck,
                                                 [3, 8, 36, 3], **kwargs)

    if pretrained:
        _load_weights(model.base.load_state_dict, 'resnet152', input_shape[0])
    return model


def _load_weights(load_dict, base_resnet, C, first_layer=True):
    """ Loads weights pretrained on ImageNet.
        Handles nb of channels other than 3 (RGB)
        Args:
            load_dict (func): model's load_state_dict() function handle
            base_resnet (str): resnet name
            C (int): number of input channels
    """
    ref = model_zoo.load_url(model_urls[base_resnet])
    conv1 = ref.pop('conv1.weight')
    if first_layer is False:
        # Do not load pretrained weights for the first conv layer
        pass
    elif C == 1:
        #  Input image is in grayscale
        load_dict(
            {
                'conv1.weight': nn.Parameter(rgb2gray(conv1).data)
            }, strict=False)
    elif C == 2:
        #  Input image is 2 different images in grayscale, one in each channel
        load_dict(
            {
                'conv1.weight':
                nn.Parameter(
                    torch.cat(
                        (rgb2gray(conv1), rgb2gray(conv1.clone())), 1).data)
            },
            strict=False)
    elif C == 3:
        load_dict({'conv1.weight': conv1}, strict=False)
    elif C == 6:
        #  Input image is 2 different images both in RGB space
        load_dict(
            {
                'conv1.weight':
                nn.Parameter(torch.cat((conv1, conv1.clone()), 1).data)
            },
            strict=False)
    else:
        raise ValueError('Invalid number of channels ({}) for input_shape'.\
                         format(C))
    load_dict(ref, strict=False)
