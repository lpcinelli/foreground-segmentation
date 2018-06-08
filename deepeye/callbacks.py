import os
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch

from inflection import titleize, humanize

from .utils.generic_utils import AverageMeter


class Callback(object):
    def __init__(self):
        pass

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_step_begin(self, size, mode=''):
        pass

    def on_batch_begin(self, batch, batch_size):
        pass

    def on_batch_end(self, metrics):
        pass

    def on_step_end(self):
        pass

    def on_epoch_end(self, metrics):
        pass

    def on_end(self):
        pass

    def set_params(self, arch=None, optimizer=None, criterion=None):
        self.arch = arch
        self.optimizer = optimizer
        self.criterion = criterion


class Compose(Callback):
    def __init__(self, callbacks=[]):
        if len(callbacks) and not all(
            [isinstance(c, Callback) for c in callbacks]):
            raise ValueError('All callbacks must be an instance of Callback')

        self.callbacks = callbacks

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        for callback in self.callbacks:
            callback.on_begin(start_epoch, end_epoch, metrics_name)

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_step_begin(self, size, mode=''):
        for callback in self.callbacks:
            callback.on_step_begin(size, mode=mode)

    def on_step_end(self):
        for callback in self.callbacks:
            callback.on_step_end()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(metrics)

    def on_batch_begin(self, batch, batch_size):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, batch_size)

    def on_batch_end(self, metrics):
        for callback in self.callbacks:
            callback.on_batch_end(metrics)

    def set_params(self, arch=None, optimizer=None, criterion=None):
        for callback in self.callbacks:
            callback.set_params(arch, optimizer, criterion)


class Progbar(Callback):
    def __init__(self, print_freq=0):
        self.print_freq = print_freq

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        self.end_epoch = end_epoch
        self.data_time = AverageMeter()

    def on_step_begin(self, size, mode=''):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.end = time.time()
        self.mode = mode
        self.size = size

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_batch_begin(self, batch, batch_size):
        self.batch = batch
        self.data_time.update(time.time() - self.end)

    def on_batch_end(self, metrics):
        # measure elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        if self.batch % self.print_freq == 0:
            msg = []
            if self.mode.startswith('train'):
                msg += [
                    'Epoch: [{0}][{1}/{2}]  '.format(self.epoch, self.batch,
                                                     self.size)
                ]
            else:
                msg += [
                    '{0}: [{1}/{2}]  '.format(
                        titleize(self.mode), self.batch, self.size)
                ]
            msg += ['Time {0.val:.3f} ({0.avg:.3f})  '.format(self.batch_time)]
            msg += ['Data {0.val:.3f} ({0.avg:.3f})  '.format(self.data_time)]

            # Add metrics alongsise with the loss
            msg += [('{0} {1.val:.3f} ({1.avg:.3f})  ' if isinstance(
                meter, AverageMeter) else '{0} {1.val:.3f} ').format(
                    titleize(
                        humanize(name.rsplit('_')[1].replace('-score', ''))),
                    meter) for name, meter in metrics.items()
                    if not name.startswith('_')]

            print(''.join(msg))
            self.metrics = metrics

    def on_step_end(self):
        msg = []

        if self.mode.startswith('train'):
            msg += [
                'Epoch: [{0}][{1}/{2}]  '.format(self.epoch, self.batch,
                                                 self.size)
            ]
        else:
            msg += [
                '{0}: [{1}/{2}]  '.format(
                    titleize(self.mode), self.batch, self.size)
            ]
        msg += ['Time {0.sum:.3f}  '.format(self.batch_time)]
        msg += ['Data {0.sum:.3f}  '.format(self.data_time)]

        # Add metrics alongsise with the loss
        msg += [
            '{0} {1.avg:.3f}  '.format(
                titleize(humanize(name.rsplit('_')[1].replace('-score', ''))),
                meter) for name, meter in self.metrics.items()
            if not name.startswith('_')
        ]

        print(''.join(msg))


class ModelCheckpoint(Callback):
    def __init__(self,
                 filepath,
                 monitor,
                 mode='min',
                 save_best=True,
                 history=OrderedDict()):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best = save_best
        self.history = history

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
            op = np.min
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
            op = np.max
        else:
            raise ValueError('mode not recognized.')

        if len(history):
            self.best = op(self.history[monitor])

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        if not len(self.history):
            self.history['epochs'] = np.arange(end_epoch)
            for name in metrics_name:
                self.history[name] = []
        print(self.history.keys())

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self, metrics):
        # Keep track of values
        for name, meter in metrics.items():
            self.history[name].append(meter.avg)

        is_best = self.monitor_op(metrics[self.monitor].avg, self.best)
        self.best = metrics[self.monitor].avg if is_best else self.best

        state = {
            'epoch': self.epoch + 1,
            'arch': self.arch,
            'state_dict': self.arch.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'criterion': self.criterion.state_dict(),
            'best_{}'.format(self.monitor): self.best
        }
        state['history'] = self.history

        filepath = self.filepath.format(epoch=self.epoch, **metrics)

        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath,
                            '{}-best{}'.format(*os.path.splitext(filepath)))


class LearningRateScheduler(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        self.lr = self.optimizer.param_groups[0]['lr']

    def on_epoch_begin(self, epoch):
        new_lr = self.scheduler(self.lr, epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class Visdom(Callback):
    def __init__(self,
                 server='http://localhost',
                 port=8097,
                 env='main',
                 history=OrderedDict()):
        from visdom import Visdom
        self.viz = Visdom(server=server, port=port, env=env)
        self.history = history

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        self.modes, self.metrics = zip(
            *[metric.split('_', 1) for metric in metrics_name])

        self.metrics = list(OrderedDict.fromkeys(self.metrics))
        self.modes = list(OrderedDict.fromkeys(self.modes))
        self.modes = self.modes or ['']

        self.viz_windows = {m: None for m in self.metrics}

        self.opts = {
            m: dict(
                title=titleize(humanize(m)),
                ylabel=titleize(humanize(m)),
                xlabel='Epoch',
                legend=[titleize(mode) for mode in self.modes])
            for m in self.metrics
        }

        if not len(self.history):
            for name in metrics_name:
                self.history[name] = np.zeros(end_epoch)
        self.history['epochs'] = np.arange(1, end_epoch + 1)

        if start_epoch != 0:
            for m in self.metrics:
                self.viz_windows[m] = self.viz.line(
                    X=np.column_stack([
                        self.history['epochs'][0:start_epoch]
                        for _ in self.modes
                    ]),
                    Y=np.column_stack([
                        self.history['{}_{}'.format(mode, m)][0:start_epoch]
                        for mode in self.modes
                    ]),
                    opts=self.opts[m])

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self, metrics):
        # Keep track of values
        for name, meter in metrics.items():
            self.history[name][self.epoch] = meter.avg

        for m in self.metrics:
            if self.viz_windows[m] is None:
                self.viz_windows[m] = self.viz.line(
                    X=np.column_stack([
                        self.history['epochs'][0:self.epoch + 1]
                        for _ in self.modes
                    ]),
                    Y=np.column_stack([
                        self.history['{}_{}'.format(mode, m)][0:self.epoch + 1]
                        for mode in self.modes
                    ]),
                    opts=self.opts[m])
            else:
                self.viz.line(
                    X=np.column_stack([
                        self.history['epochs'][0:self.epoch + 1]
                        for _ in self.modes
                    ]),
                    Y=np.column_stack([
                        self.history['{}_{}'.format(mode, m)][0:self.epoch + 1]
                        for mode in self.modes
                    ]),
                    win=self.viz_windows[m],
                    update='replace')
