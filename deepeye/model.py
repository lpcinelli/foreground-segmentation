from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .callbacks import Callback, Progbar
from .utils.generic_utils import AverageMeter


class Model(object):
    def __init__(self, arch, criterion=None, optimizer=None, threshold=0.5):
        if not isinstance(arch, nn.Module):
            raise ValueError('Arch should be an instance of torch.nn.Module')

        self.arch = arch
        self.set_criterion(criterion)
        self.set_optimizer(optimizer)
        self.set_threshold(threshold)

        self.cuda = False
        if next(self.arch.parameters()).is_cuda:
            self.cuda = True

    def set_criterion(self, criterion):
        if not isinstance(criterion, nn.Module) and criterion:
            raise ValueError('Criterion should be an instance of '
                             'torch.nn.Module')

        self._criterion = criterion

    def set_optimizer(self, optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer) and optimizer:
            raise ValueError('Optimizer should be an instance of '
                             'torch.optim.Optimizer')
        self._optimizer = optimizer

    def set_threshold(self, threshold):
        if not isinstance(threshold, float) or threshold > 1 or threshold < 0:
            raise ValueError('Threshold should be a float in the '
                             'interval [0,1]')
        if hasattr(self._criterion, 'threshold'):
            self._threshold = self._criterion.threshold
        else:
            self._threshold = threshold

    @property
    def optimizer(self):
        if not self._optimizer:
            raise ValueError('optimizer was not set')
        return self._optimizer

    @property
    def criterion(self):
        if not self._criterion:
            raise ValueError('criterion was not set')
        return self._criterion

    def fit_loader(self,
                   loader,
                   epochs,
                   val_loader=None,
                   metrics={},
                   callback=None,
                   start_epoch=0):
        if not isinstance(loader, torch.utils.data.DataLoader):
            raise ValueError('loader should be a instance of '
                             'torch.utils.data.DataLoader')

        if val_loader and not isinstance(val_loader,
                                         torch.utils.data.DataLoader):
            raise ValueError('val_loader should be a instance of '
                             'torch.utils.data.DataLoader')

        if not isinstance(callback, Callback):
            raise ValueError('callback should be a instance of Callback')

        names = [
            '{{}}_{}'.format(name.replace('_', '-'))
            for name in ['loss'] + list(metrics.keys())
        ]
        metrics_name = [n.format('train') for n in names]
        if val_loader:
            metrics_name += [n.format('val') for n in names]

        callback.set_params(
            arch=self.arch, optimizer=self.optimizer, criterion=self.criterion)

        callback.on_begin(start_epoch, epochs, metrics_name)

        for epoch in range(start_epoch, epochs):

            callback.on_epoch_begin(epoch)

            # train for one epoch
            train_metrics = self._step_loader(
                loader, callback, metrics=metrics, mode='train')

            if val_loader:
                # evaluate
                val_metrics = self._step_loader(
                    val_loader, callback, metrics=metrics, mode='val')
                train_metrics.update(val_metrics)

            callback.on_epoch_end(train_metrics)

        callback.on_end()

    def eval_loader(self, loader, metrics={}, callback=None):
        if not isinstance(loader, torch.utils.data.DataLoader):
            raise ValueError('loader should be a instance of '
                             'torch.utils.data.DataLoader')

        callback = callback or Progbar(print_freq=len(loader) - 1)

        callback.set_params(arch=self.arch, criterion=self.criterion)

        names = [
            '{}'.format(name.replace('_', '-'))
            for name in ['loss'] + list(metrics.keys())
        ]

        callback.on_begin(metrics_name=names)
        callback.on_epoch_begin(0)
        metrics = self._step_loader(
            loader, callback, metrics=metrics, mode='test')
        callback.on_epoch_end(metrics)
        callback.on_end()

        return metrics

    def predict_loader(self, loader, callback=None):
        if not isinstance(loader, torch.utils.data.DataLoader):
            raise ValueError('loader should be a instance of '
                             'torch.utils.data.DataLoader')

        callback = callback or Progbar(print_freq=len(loader) - 1)

        callback.set_params(arch=self.arch)

        callback.on_begin()
        callback.on_epoch_begin(0)
        outputs = self._step_loader(loader, callback, mode='predict')
        callback.on_epoch_end({})
        callback.on_end()

        return outputs

    def _step_loader(self,
                     loader,
                     callback,
                     metrics={},
                     mode='train',
                     async=True):

        meters = OrderedDict()
        if mode == 'predict':
            outputs = np.zeros((len(loader.dataset), len(
                loader.dataset.classes)))
            seen = 0
        else:
            meters['{}_loss'.format(mode)] = AverageMeter()
            for name in metrics:
                meters['{}_{}'.format(mode, name)] = AverageMeter()

        if mode == 'train':
            # switch to train mode
            self.arch.train()
            volatile = False
        else:
            self.arch.eval()
            volatile = True

        callback.on_step_begin(len(loader), mode=mode)

        for batch, (input, target, roi) in enumerate(loader):
            batch_size = input.size(0)
            nb_pixels = np.prod(input.size())

            callback.on_batch_begin(batch, batch_size)

            input_var = torch.autograd.Variable(input, volatile=volatile)

            if mode != 'predict':
                if self.cuda:
                    target = target.cuda(async=async)
                    roi = roi.cuda(async=async)
                target_var = torch.autograd.Variable(target, volatile=volatile)
                roi_var = torch.autograd.Variable(roi, volatile=volatile)
            # Compute output
            output = self.arch(input_var)
            if mode == 'predict':
                outputs[seen:seen + batch_size, ...] =\
                    output.data.cpu().numpy()
                seen += batch_size
            else:
                loss = self.criterion(output, target_var, roi=roi_var)

                # Updating meters
                meters['{}_loss'.format(mode)].update(loss.data[0], batch_size)
                output = (torch.sigmoid(output) > self._threshold)
                for name, metric in metrics.items():
                    meters['{}_{}'.format(mode, name)].update(
                        metric(target.byte(), output, roi=roi.byte()),
                        nb_pixels)

                if mode == 'train':
                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            callback.on_batch_end(meters)

        callback.on_step_end()

        return meters or outputs
