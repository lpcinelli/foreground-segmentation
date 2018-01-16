import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from functools import partial


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockUp(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicBlockUp, self).__init__()
        self.conv1 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3,
                                        stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3,
                                        stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, midplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        outplanes = midplanes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out

class BottleneckUp(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BottleneckUp, self).__init__()
        midplanes = inplanes//self.expansion
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.ConvTranspose2d(midplanes, midplanes, kernel_size=3,
                                        stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_shape, block, layers, num_classes=1, dilation=1,
                 return_indices=False, return_sizes=False):
        super(ResNet, self).__init__()

        C, W, H = input_shape
        self.inplanes = 64
        self.return_indices = return_indices
        self.return_sizes = return_sizes

        self.conv1 = nn.Conv2d(C, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                    return_indices=True, ceil_mode=True)
        self.layer1 = self._make_down_layer(block, 64, layers[0])

        self.layer2 = self._make_down_layer(block, 128, layers[1], stride=2).apply(
                      partial(self._nostride_dilate, dilate=dilation/4))
        self.layer3 = self._make_down_layer(block, 256, layers[2], stride=2).apply(
                      partial(self._nostride_dilate, dilate=dilation/2))
        self.layer4 = self._make_down_layer(block, 512, layers[3], stride=2).apply(
                      partial(self._nostride_dilate, dilate=dilation))

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
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
                min_dilate = int(max(1, dilate//2))
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = [x]

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

    def __init__(self, input_shape, block, layers, num_classes=1, dilation=1,
                 upsampling='bilinear'):
        super(ResNetUpSample, self).__init__()

        _, H, W = input_shape
        self.base = ResNet(input_shape, block, layers, num_classes=1,
                           dilatation=dilation)
        last_layer = [obj for obj in self.base.modules() if isinstance(obj, nn.Conv2d)][-1]

        self.classifier = nn.Sequential(
            nn.Conv2d(last_layer.out_channels, 1, kernel_size=1, padding=0))

    def forward(self, x):
        orig_size = x.size()

        # Extracting the features
        x = self.base(x)[0]

        # Projecting to belif map
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size, mode=self.upsampling)

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

        blocks can either be ['BasicBlockUp', 'BottleneckUp']
    """

    def __init__(self, input_shape, block, layers, num_classes=1, dilation=1,
                 upsampling='bilinear'):
        super(ResNetDeconv, self).__init__()

        _, H, W = input_shape
        self.upsampling = upsampling
        self.base = ResNet(input_shape, block, layers, num_classes=1,
                           dilation=dilation, return_indices=True,
                           return_sizes=True)
        last_layer = [obj for obj in self.base.modules() if isinstance(obj, nn.Conv2d)][-1]
        self.inplanes = last_layer.out_channels

        block = BasicBlockUp if block.__name__ == 'BasicBlock' else BottleneckUp

        self.layer1 = self._make_up_layer(block, layers[3], stride=2).apply(
                      partial(self._nostride_dilate, dilate=dilation))
        self.layer2 = self._make_up_layer(block, layers[2], stride=2).apply(
                      partial(self._nostride_dilate, dilate=dilation/2))
        self.layer3 = self._make_up_layer(block, layers[1], stride=2).apply(
                      partial(self._nostride_dilate, dilate=dilation/4))
        self.layer4 = self._make_up_layer(block, layers[0], outplanes=64)

        self.unpool = nn.MaxUnpool2d(kernel_size=(3, 3), stride=(2, 2),
                                     padding=1)
        self.deconv = nn.ConvTranspose2d(self.inplanes, 2, kernel_size=7,
                                         stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(2, 1, kernel_size=1, padding=0)

    def _make_up_layer(self, block, blocks, outplanes=None, stride=1):
        upsample = None
        if outplanes is None:
            outplanes = self.inplanes // 2

        if stride != 1 or self.inplanes != outplanes :
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, outplanes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, outplanes, stride, upsample))
        self.inplanes = outplanes
        return nn.Sequential(*layers)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and dilate > 1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                min_dilate = int(max(1, dilate//2))
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
        orig_size = x.size()[-2:]

        # Extracting the features
        x, pool_idx, size = self.base(x)

        # Reconstructing feature map
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.upsample(x, size=size[-1][-2:], mode='bilinear')
        x = self.unpool(x, pool_idx, output_size=size[-2])

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = F.upsample(x, scale_factor=2, mode=self.upsampling)

        # Projecting to belief map
        x = self.classifier(x)
        x = F.upsample(x, size=orig_size, mode=self.upsampling)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')
    model = ResNet(input_shape, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained and input_shape[0] == 3:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')
    model = ResNet(input_shape, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained and input_shape[0] == 3:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')
    model = ResNet(input_shape, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained and input_shape[0] == 3:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')
    model = ResNet(input_shape, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained and input_shape[0] == 3:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_shape = kwargs.pop('input_shape', None)
    if not input_shape:
        raise ValueError('input_shape is required')
    model = ResNet(input_shape, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained and input_shape[0] == 3:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
