import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module


class MaskedBinaryCrossEntropy(Module):
    def forward(self, input, target, roi=None):
        roi = roi or torch.ones_like(input)
        return F.binary_cross_entropy(
               torch.sigmoid(input), target, size_average=False) * roi).sum()/roi.sum()


bce = MaskedBinaryCrossEntropy


class MaskedDiceLoss(Module):
    def __init__(self, th= 0.5, smooth=1, eps=1e-12):
        super().__init__()
        self.th = th
        self.eps = eps
        self.smooth = smooth

    def forward(self, input, target, roi=None):
        roi = roi or torch.ones_like(input)
        input = (torch.sigmoid(input) > self.th).float()
        intersect = (input * target * roi).view(input.size(0), -1).sum(1)
        union = ((input + target) * roi).view(input.size(0), -1).sum(1)

        return (1 - (2 * intersect + self.smooth) / ((union + self.eps)).mean() + self.smooth)
