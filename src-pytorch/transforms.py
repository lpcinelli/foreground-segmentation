import numpy as np
import torch

from torchvision import transforms


class ToTensor(transforms.ToTensor):
    """ Extend default ToTensor functionality to handle .tif images uint 16-bit
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray) and pic.dtype == np.uint16:
            # handle numpy array
            img = torch.from_numpy(
                pic.transpose((2, 0, 1)).astype(np.float32))
            # backard compability
            return img.float().div(2**16-1)

        return super(ToTensor, self).__call__(pic)
