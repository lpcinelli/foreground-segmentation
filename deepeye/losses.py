import torch

import torch.nn.functional as F

from torch.nn.modules.module import Module


class BinaryCrossEntropy(Module):
    def forward(self, input, target):
        return F.binary_cross_entropy(torch.sigmoid(input), target)


bce = BinaryCrossEntropy
