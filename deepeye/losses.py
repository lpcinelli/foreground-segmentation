import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module


class MaskedBinaryCrossEntropy(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target, roi=None):
        if roi is None:
            roi = torch.ones_like(input)
        return (F.binary_cross_entropy_with_logits(
            input, target, reduce=False) * roi).sum() / roi.sum()


mbce = MaskedBinaryCrossEntropy


class BinaryCrossEntropy(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target, roi=None):
        return super().forward(input, target)


bce = BinaryCrossEntropy


class MaskedHardDiceLoss(Module):
    """Non-differentiable (due to threshold operation on input arg)
       Sørensen–Dice coefficient with mask for comparing the similarity in
       region of interest of two batch of data.
       Coefficient ranges between 0 to 1, 1 if totally matchs.
       Inputs:
                threshold (float): threshold value above which is 1, else 0
                eps (float): small value to avoid division by zero
                smooth (float): negative value added to both num. and den.
                                (laplacian/additive smoothing)
                per_img (bool): If True, computes loss per image and then
                         averages, else computes it per batch
       Shape:
                input (tensor): `(N, *)` where `*` means, any number of
                       additional dimensions
                target (tensor): `(N, *)`, same shape as the input
                roi (tensor): `(N, *)`, same shape as the input
    """

    def __init__(self, threshold=0.5, smooth=1, eps=1e-12, per_img=False):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        self.smooth = smooth
        self.per_img = per_img

    def forward(self, input, target, roi=None):
        if roi is None:
            roi = torch.ones_like(input)
        input = (torch.sigmoid(input) > self.threshold).float()
        intersect = (input * target * roi).view(input.size(0), -1).sum(1)
        union = ((input + target) * roi).view(input.size(0), -1).sum(1)

        if self.per_img is True:
            return (1 - (2 * intersect + self.smooth) /
                    ((union + self.eps) + self.smooth)).mean()

        return 1 - ((2 * intersect.sum() + self.smooth) /
                    (union.sum() + self.eps + self.smooth))


harddice = MaskedHardDiceLoss


class MaskedSoftDiceLoss(Module):
    """Non-differentiable Sørensen–Dice coefficient for comparing the
       similarity in the region of interest of two batch of data.
       Coefficient ranges between 0 to 1, 1 if totally matchs.
       Inputs:
                eps (float): small value to avoid division by zero
                smooth (float): negative value added to both num. and den.
                                (laplacian/additive smoothing)
                per_img (bool): If True, computes loss per image and then
                         averages, else computes it per batch
       Shape:
                input (tensor): `(N, *)` where `*` means, any number of
                       additional dimensions
                target (tensor): `(N, *)`, same shape as the input
                roi (tensor): `(N, *)`, same shape as the input
    """

    def __init__(self, threshold=0.5, smooth=1, eps=1e-12, per_img=False):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        self.smooth = smooth
        self.per_img = per_img

    def forward(self, input, target, roi=None):
        if roi is None:
            roi = torch.ones_like(input)
        input = torch.sigmoid(input)
        intersect = (input * target * roi).view(input.size(0), -1).sum(1)
        union = ((input + target) * roi).view(input.size(0), -1).sum(1)

        if self.per_img is True:
            return (1 - (2 * intersect + self.smooth) /
                    ((union + self.eps) + self.smooth)).mean()

        return 1 - ((2 * intersect.sum() + self.smooth) /
                    (union.sum() + self.eps + self.smooth))


softdice = MaskedSoftDiceLoss
