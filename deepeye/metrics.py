from sklearn import metrics

import torch
import numpy as np


def f2_score(y_pred, y_true, threshold=0.5):
    return fbeta_score(y_pred, y_true, 2, threshold)


def fbeta_score(y_pred, y_true, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.gt(torch.sigmoid(y_pred.float()), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision * recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))
