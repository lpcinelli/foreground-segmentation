""" Those metrics are restricted to binary classification task

TODO
    Add docstring for each method
"""
import torch


def _sanitize(y_true, y_pred, roi):
    if type(y_pred) == torch.autograd.Variable:
        y_pred = y_pred.data

    # Flatten
    y_pred = y_pred.view(y_pred.shape[0], -1)

    if type(y_pred) not in [torch.ByteTensor, torch.cuda.ByteTensor]:
        raise ValueError('y_pred must be a ByteTensor, got {}'.format(
            type(y_pred)))

    if type(y_true) == torch.autograd.Variable:
        y_true = y_true.data

    # Flatten
    y_true = y_true.view(y_true.shape[0], -1)

    if type(y_true) not in [torch.ByteTensor, torch.cuda.ByteTensor]:
        raise ValueError('y_true must be a ByteTensor, got {}'.format(
            type(y_true)))

    if roi is None:
        roi = torch.ones_like(y_pred)
    else:
        if type(roi) == torch.autograd.Variable:
            roi = roi.data

        # Flatten
        roi = roi.view(roi.shape[0], -1)

        if type(roi) not in [torch.ByteTensor, torch.cuda.ByteTensor]:
            raise ValueError('roi must be a ByteTensor, got {}'.format(
                type(roi)))


    return y_true, y_pred, roi


def _tn(y_true, y_pred, roi):
    return torch.sum((((y_pred == 0) & (y_true == 0)) * roi).float(), dim=-1)


def _tp(y_true, y_pred, roi):
    return torch.sum((((y_pred == 1) & (y_true == 1)) * roi).float(), dim=-1)


def _fp(y_true, y_pred, roi):
    return torch.sum((((y_pred == 1) & (y_true == 0)) * roi).float(), dim=-1)


def _fn(y_true, y_pred, roi):
    return torch.sum((((y_pred == 0) & (y_true == 1)) * roi).float(), dim=-1)


def acc_score(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tn = _tn(y_true, y_pred, roi)
    tp = _tp(y_true, y_pred, roi)

    return torch.mean((tp + tn) / (roi.float().sum(dim=-1) + eps))


def prec_score(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)

    return torch.mean(tp / (tp + fp + eps))


def recall_score(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)

    return torch.mean(tp / (tp + fn + eps))


def f1_score(y_true, y_pred, roi=None, eps=1e-12):
    # Dice's index : 2*TP/(2*TP + FP + FN)
    return fbeta_score(y_true, y_pred, 1, roi, eps)


def fbeta_score(y_true, y_pred, beta, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)
    beta2 = beta**2

    tp = _tp(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)

    return torch.mean(
        (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp + eps))


def false_pos_rate(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    fp = _fp(y_true, y_pred, roi)
    tn = _tn(y_true, y_pred, roi)

    return torch.mean(fp / (fp + tn + eps))


def false_neg_rate(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    fn = _fn(y_true, y_pred, roi)
    tp = _tp(y_true, y_pred, roi)

    return torch.mean(fn / (fn + tp + eps))


def true_pos_rate(y_true, y_pred, roi=None, eps=1e-12):
    # or sensitivity
    return 1 - false_neg_rate(y_true, y_pred, roi, eps)


def true_neg_rate(y_true, y_pred, roi=None, eps=1e-12):
    # or specificity
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    fp = _fp(y_true, y_pred, roi)
    tn = _tn(y_true, y_pred, roi)

    return torch.mean(tn / (fp + tn + eps))


def IoU_score(y_true, y_pred, roi=None, eps=1e-12):
    # Jaccard's index : TP/(TP + FP + FN)
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)

    return torch.mean(tp / (tp + fp + fn + eps))


def total_error(y_true, y_pred, roi=None, eps=1e-12):
    # Total error: (FN + FP)/(TP + FP + TN + FN)
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    tn = _tn(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)

    return torch.mean((fn + fp) / (tp + fp + tn + fn + eps))
