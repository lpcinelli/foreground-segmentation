import torch

import torch.nn.functional as F

from torch.nn.modules.module import Module


class MaskedBinaryCrossEntropy(Module):
    def forward(self, input, target, roi=roi):
        return torch.mean(
            F.binary_cross_entropy(
                torch.sigmoid(input), target, size_average=False) * roi)


bce = MaskedBinaryCrossEntropy


class MaskedDiceLoss(Module):
    def __init__(self, th, eps=1e-12):
        super().__init__()
        self.th = th
        self.eps = eps

    def forward(self, input, target, roi=roi):
        input = (torch.sigmoid(input) > self.th).float()
        intersect = (input * target * roi).view(input.size(0), -1).sum(1)
        union = ((input + target) * roi).view(input.size(0), -1).sum(1)

        return (1 - 2 * intersect / (union + self.eps)).mean()
