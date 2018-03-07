import numpy as np


class Meter(object):
    """ Stores the last value and the cummulative sum
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.avg = val


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def reset(self):
        super().reset()
        self.count = 0

    def update(self, val, n=1):
        super().update(val, n)
        self.count += n
        self.avg = self.sum / self.count


class CMMeter(object):
    """ Confusion matrix meter, i.e., computes TP, TN, FP, and FN globally
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, tp, tn, fp, fn, n=1):
        self.tp += tp * n
        self.tn += tn * n
        self.fp += fp * n
        self.fn += fn * n


class History(AverageMeter):
    """ Stores values and computes some metrics """

    def reset(self):
        super().reset()
        self.vals = []

    def update(self, val, n=1):
        super().update(val, n)
        self.vals.append(val * n)


def conv2_out_size(input_size,
                   kernel_size,
                   stride=1,
                   padding=0,
                   dilation=1,
                   ceil_mode=False):
    """ Computes height and width size of the output tensor
        Inputs:
                input_size: tuple with height h and width w of the input tensor
                kernel_size: tuple with  kernel dimensions
                stride: int or tuple with kernel's stride along each dim
                padding: int or tuple with the (half the)  amount of padding
                         on each dim
                dilation: int or tuple with that controls the spacing between
                          kernel points
        Returns:
                (output height, output width)

    """
    if isinstance(kernel_size, (int, float)):
        kernel_size = (int(kernel_size), int(kernel_size))

    if isinstance(stride, (int, float)):
        stride = (int(stride), int(stride))

    if isinstance(padding, (int, float)):
        padding = (int(padding), int(padding))

    if isinstance(dilation, (int, float)):
        dilation = (int(dilation), int(dilation))

    return (
        int((input_size[0] + 2 * padding[0] - dilation[0] *
             (kernel_size[0] - 1) - 1) / stride[0] + 1 + int(ceil_mode) * 0.5),
        int((input_size[1] + 2 * padding[1] - dilation[1] *
             (kernel_size[1] - 1) - 1) / stride[1] + 1 + int(ceil_mode) * 0.5))


def find_threshold(y_pred, y_true, metric, min_val=0, max_val=1.0, eps=1e-6):
    '''
    Finds the best threshold considering a metric and a training set.

    @param y_pred Current predicted values to be thresholded.
    @param y_true Target labels.
    @param metric Metric to be maximized.
    @param min_val Minimum possible predicted value.
    @param max_val Maximum possible predicted value.
    @param eps Minimum interval

    @return Best threshold and metric value.
    '''

    # Set initial values
    thrs_cur = (min_val + max_val) / 2.0
    thrs_low = min_val
    thrs_hgh = max_val

    # Bisection algorithm
    while (True):

        # Initializing bisection algorithm
        thrs_cur = (min_val + max_val) / 2.0
        thrs_low = (thrs_cur + min_val) / 2.0
        thrs_hgh = (thrs_cur + max_val) / 2.0

        # Scores
        scr_low = metric(y_true, y_pred, thrs_low)
        scr_hgh = metric(y_true, y_pred, thrs_hgh)

        # Testing
        if (scr_low >= scr_hgh):
            # Update
            max_val = thrs_hgh
        else:
            min_val = thrs_low

        # End condition
        if (abs(max_val - min_val) <= eps):
            break

    # Return
    return thrs_cur, scr_low


def rgb2gray(weights):
    """ Converts weights pretrained on RGB to grayscale
        Args:
            weights (torch.Tensor): model's weights of size (?, 3, ?, ?)
        Returns:
            torch.Tensor of size (?, 1, ?, ?)
    """
    return (0.2989 * weights[:, 0, :, :] + 0.5870 * weights[:, 1, :, :] +
            0.1140 * weights[:, 2, :, :]).unsqueeze(1)
