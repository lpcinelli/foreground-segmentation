from sklearn import metrics

import torch
import numpy as np


def acc_score():
    pass


def prec_score():
    pass


def recall_score():
    pass


def f1_score(y_pred, y_true, threshold=0.5):
    # Dice's index : 2*TP/(2*TP + FP + FN)
    return fbeta_score(y_pred, y_true, 1, threshold)


def fbeta_score(y_pred, y_true, roi=None, beta=1, threshold=0.5, eps=1e-12):
    roi = roi or torch.ones_like(y_pred)
    beta2 = beta**2

    y_pred = (torch.sigmoid(y_pred.float()) > threshold).float() * roi
    y_true = y_true.float() * roi

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive / (y_pred.sum(dim=1) + eps)
    recall = true_positive / (y_true.sum(dim=1) + eps)

    return torch.sum((1 + beta2) * (precision * recall) /
                     (precision * beta2 + recall + eps)) / roi.sum()


def false_pos_rate():
    pass


def false_neg_rate():
    pass


def true_pos_rate():
    return recall_score()


def true_neg_rate():
    pass


def IoU_score():
    # Jaccard's index : TP/(TP + FP + FN)
    pass


def total_error():
    # Total error: (FN + FP)/(TP + FP + TN + FN)
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    pass