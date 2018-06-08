""" Those metrics are restricted to binary classification task

TODO
    Add docstring for each method
"""
import torch


def _sanitize(y_true, y_pred, roi):
    y_pred = y_pred.data
    y_true = y_true.data

    # Flatten
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_true.shape[0], -1)

    if y_pred.dtype is not torch.uint8:
        raise ValueError('y_pred must be torch.uint8, got {}'.format(
            y_pred.dtype))

    if y_true.dtype is not torch.uint8:
        raise ValueError('y_true must be torch.uint8, got {}'.format(
            y_true.dtype))

    if roi is None:
        roi = torch.ones_like(y_pred)
    else:
        roi = roi.data

        # Flatten
        roi = roi.view(roi.shape[0], -1)

        if roi.dtype is not torch.uint8:
            raise ValueError('roi must be torch.uint8, got {}'.format(
                roi.dtype))

    return y_true, y_pred, roi


def _tn(y_true, y_pred, roi):
    return torch.sum((((y_pred == 0) & (y_true == 0)) * roi).float())


def _tp(y_true, y_pred, roi):
    return torch.sum((((y_pred == 1) & (y_true == 1)) * roi).float())


def _fp(y_true, y_pred, roi):
    return torch.sum((((y_pred == 1) & (y_true == 0)) * roi).float())


def _fn(y_true, y_pred, roi):
    return torch.sum((((y_pred == 0) & (y_true == 1)) * roi).float())


def tp_tn_fp_fn(y_true, y_pred, roi=None):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    return _tp(y_true, y_pred, roi), _tn(y_true, y_pred, roi), _fp(
        y_true, y_pred, roi), _fn(y_true, y_pred, roi)


def acc_score(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tn = _tn(y_true, y_pred, roi)
    tp = _tp(y_true, y_pred, roi)
    return _acc_score(tp, tn, None, None, eps)


def _acc_score(tp, tn, fp, fn, eps=1e-12):
    return ((tp + tn) / (roi.float().sum(dim=-1) + eps))


def prec_score(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)
    return _prec_score(tp, None, fp, None, eps)


def _prec_score(tp, tn, fp, fn, eps=1e-12):
    return (tp / (tp + fp + eps))


def recall_score(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)
    return _recall_score(tp, None, None, fn)


def _recall_score(tp, tn, fp, fn, eps=1e-12):
    return (tp / (tp + fn + eps))


def f1_score(y_true, y_pred, roi=None, eps=1e-12):
    # Dice's index : 2*TP/(2*TP + FP + FN)
    return fbeta_score(y_true, y_pred, 1, roi, eps)


def _f1_score(tp, tn, fp, fn, eps=1e-12):
    return _fbeta_score(tp, None, fp, fn, 1, eps)


def fbeta_score(y_true, y_pred, beta, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)

    return _fbeta_score(tp, None, fp, fn, beta, eps)


def _fbeta_score(tp, tn, fp, fn, beta, eps=1e-12):
    beta2 = beta**2
    return ((1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp + eps))


def false_pos_rate(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    fp = _fp(y_true, y_pred, roi)
    tn = _tn(y_true, y_pred, roi)

    return _false_pos_rate(None, tn, fp, None, eps)


def _false_pos_rate(tp, tn, fp, fn, eps=1e-12):
    return (fp / (fp + tn + eps))


def false_neg_rate(y_true, y_pred, roi=None, eps=1e-12):
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    fn = _fn(y_true, y_pred, roi)
    tp = _tp(y_true, y_pred, roi)

    return _false_neg_rate(tp, None, None, fn, eps)


def _false_neg_rate(tp, tn, fp, fn, eps=1e-12):
    return (fn / (fn + tp + eps))


def true_pos_rate(y_true, y_pred, roi=None, eps=1e-12):
    # or sensitivity
    return 1 - false_neg_rate(y_true, y_pred, roi, eps)


def _true_pos_rate(tp, tn, fp, fn, eps=1e-12):
    return 1 - _true_neg_rate(None, tn, fp, fn, eps)


def true_neg_rate(y_true, y_pred, roi=None, eps=1e-12):
    # or specificity
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    fp = _fp(y_true, y_pred, roi)
    tn = _tn(y_true, y_pred, roi)

    return _true_neg_rate(None, tn, fp, None, eps)


def _true_neg_rate(tp, tn, fp, fn, eps=1e-12):
    return (tn / (fp + tn + eps))


def IoU_score(y_true, y_pred, roi=None, eps=1e-12):
    # Jaccard's index : TP/(TP + FP + FN)
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)

    return _IoU_score(tp, None, fp, fn, eps)


def _IoU_score(tp, tn, fp, fn, eps=1e-12):
    return (tp / (tp + fp + fn + eps))


def total_error(y_true, y_pred, roi=None, eps=1e-12):
    # Total error: (FN + FP)/(TP + FP + TN + FN)
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    y_true, y_pred, roi = _sanitize(y_true, y_pred, roi)

    tp = _tp(y_true, y_pred, roi)
    tn = _tn(y_true, y_pred, roi)
    fp = _fp(y_true, y_pred, roi)
    fn = _fn(y_true, y_pred, roi)

    return _total_error(tp, tn, fp, fn, eps)


def _total_error(tp, tn, fp, fn, eps=1e-12):
    return ((fn + fp) / (tp + fp + tn + fn + eps))
