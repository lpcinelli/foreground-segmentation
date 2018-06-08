import random

import numpy as np
import torch
from PIL import Image

import torchvision.transforms.functional as F
from torchvision.transforms import transforms

INTERPOLATION = {
    'NEAREST': Image.NEAREST,
    'BILINEAR': Image.BILINEAR,
    'BICUBIC': Image.BICUBIC,
    'LANCZOS': Image.LANCZOS,
}

class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``. Other options are ``PIL.Image.NEAREST``,
            ``PIL.Image.BICUBIC`` and ``PIL.Image.LANCZOS``.
    """
    def __init__(self, size, interpolation):
        super().__init__(size, interpolation)

        if isinstance(interpolation, int):
            interpolation = [interpolation]*4

        if len(interpolation) != 4:
            raise ValueError('Number of interepolarion methods should '
                             'be the same as the number of images (4)')

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return tuple(
            map(lambda x, interp:
                    F.resize(x, self.size, interp), imgs, self.interpolation))


class CenterCrop(transforms.CenterCrop):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return tuple(map(lambda x: F.center_crop(x, self.size), imgs))


class RandomCrop(transforms.RandomCrop):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            imgs = tuple(map(lambda x: F.pad(x, self.padding), imgs))

        i, j, h, w = self.get_params(imgs[0], self.size)

        return tuple(map(lambda x: F.crop(x, i, j, h, w), imgs))


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return tuple(map(lambda x: F.hflip(x), imgs))
        return imgs


class ColorJitter(transforms.ColorJitter):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        input_, bg_model, target, roi = imgs
        return (transform(input_), transform(bg_model), target, roi)


class RandomRotation(transforms.RandomRotation):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __call__(self, imgs):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return tuple(
            map(lambda x: F.rotate(x, angle, self.resample, self.expand, self.center),
                imgs))


class Grayscale(transforms.Grayscale):
    """Convert image to grayscale.
    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b
    """

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.
        Returns:
            PIL Image: Randomly grayscaled image.
        """
        input_, bg_model, target, roi = imgs

        return (F.to_grayscale(
            input_, num_output_channels=self.num_output_channels),
                F.to_grayscale(
                    bg_model, num_output_channels=self.num_output_channels),
                target, roi)


class MergeChannels(object):
    def __call__(self, imgs):
        input_, bg_model, target, roi = imgs
        return (np.dstack((bg_model, input_)), target, roi)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (W x H x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pics):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return tuple(map(lambda x: F.to_tensor(x), pics))


class RoiCrop(object):
    """Crop the given image Tensor according to Region-Of-Interest (ROI)
    """

    def __call__(self, pics):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be cropped.
        Returns:
            Tensor: Cropped image.
        """
        # roi = pics[-1]
        # background = Image.new('L', roi.size)

        # return tuple(map(lambda img:
        #     Image.composite(img, background.convert(img.mode),
        #                     roi.convert(img.mode)), pics))

        input_, target, roi = pics
        return (input_ * roi, target * roi, roi)


class BinarizeTarget(object):
    """Removes undesired classes hard shadow (50), unknown motion (170)
    and outside roi (85) by forcing them to be static (0). At the end there is
    only static/background/negative and motion/foreground/positive.
    """

    def __call__(self, pics):

        input_, bg_model, target, roi = pics
        target = Image.fromarray(
                    (np.array(target.convert('L')) == 255).astype(
                        np.uint8)*255)
        # target = (target > 0.67).float()
        # target = (target > 0.83).float()
        return (input_, bg_model, target, roi)
