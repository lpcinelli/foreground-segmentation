import argparse
import errno
import getpass
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from inflection import humanize, titleize

import deepeye.archs as archs
import deepeye.datasets as datasets
import torchvision.transforms as transforms
from deepeye import callbacks, losses, metrics
from deepeye.model import Model
from deepeye.transforms import ToTensor
from deepeye.utils import arg_utils

arch_names = sorted(name for name in archs.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(archs.__dict__[name]))

losses_names = sorted(name for name in losses.__dict__
                      if name.islower() and not name.startswith("__")
                      and callable(losses.__dict__[name]))

optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}


def adjust_learning_rate(lr, epoch, factor=10, every=30):
    """Sets the learning rate to the initial LR decayed by factor 10 every
    30 epochs
    """
    return lr * (1 / factor**(epoch // every))


def _common(args, training=False):

    if 'augmentation' not in args:
        args.augmentation = False
    if 'shrink_negatives' not in args:
        args.shrink_negatives = False

    dataset = datasets.CDNetDataset(
        args.manifest,
        args.img_dir,
        training=training,
        augmentation=args.augmentation,
        shrink_data=args.shrink_negatives,
        input_shape=tuple(map(int, args.shape.split(','))))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=args.cuda)

    # create model
    print("=> creating model '{}'".format(args.arch))
    print("==> args: {}".format(args.arch_params))
    arch = archs.__dict__[args.arch](
        input_shape=dataset.input_shape,
        num_classes=1,
        **arg_utils.parse_kwparams(args.arch_params))

    print(arch)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        if args.cuda:
            arch.features = torch.nn.DataParallel(arch.features)
            arch.cuda()
    else:
        if args.cuda:
            arch = torch.nn.DataParallel(arch)
            arch = arch.cuda()

    # define loss function (criterion) and optimizer
    criterion = losses.__dict__[args.loss]()
    if args.cuda:
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    checkpoint = {}
    if args.load:
        if os.path.isfile(args.load):
            print("=> loading checkpoint '{}'".format(args.load))
            checkpoint = torch.load(args.load)
            args.start_epoch = checkpoint['epoch']
            arch.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.load, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load))

    # Load trainer
    model = Model(arch, criterion=criterion)

    # This can accelarate your code
    if args.cuda:
        cudnn.benchmark = True

    # Data loading code
    print(args)

    return loader, checkpoint, model


def train(args):
    try:

        os.makedirs(os.path.split(os.path.abspath(args.save))[0])
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    train_loader, checkpoint, model = _common(args, training=True)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.arch.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    else:
        print('=> Optimizer parameter momentum ignored')
        optimizer = optimizers[args.optim](
            model.arch.parameters(), args.lr, weight_decay=args.weight_decay)
    history = {}
    if args.load and checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']

    model.set_optimizer(optimizer)

    val_loader = None
    monitor = 'train_f1'
    if args.val_manifest:
        val_set = datasets.CDNetDataset(
            args.val_manifest, args.img_dir, training=False)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=args.cuda)
        monitor = 'val_f1'

    callback_list = [
        callbacks.Progbar(print_freq=args.print_freq),
        callbacks.ModelCheckpoint(
            args.save, monitor, mode='max', history=history.copy()),
        callbacks.LearningRateScheduler(
            partial(
                adjust_learning_rate,
                factor=args.lr_factor,
                every=args.lr_span)),
    ]

    if args.visdom:
        callback_list += [
            callbacks.Visdom(env=args.env, history=history.copy())
        ]

    model.fit_loader(
        train_loader,
        args.epochs,
        val_loader=val_loader,
        metrics={
            'f1': metrics._f1_score,
            'recall': metrics._recall_score,
            'prec': metrics._prec_score,
            'FNR': metrics._false_neg_rate,
            'TPR': metrics._true_pos_rate,
            'IoU': metrics._IoU_score,
            'total-error': metrics._total_error
        },
        callback=callbacks.Compose(callback_list),
        start_epoch=args.start_epoch)


def eval(args):
    loader, _, model = _common(args, training=False)

    outputs = model.eval_loader(
        loader,
        metrics={
            'f1': metrics._f1_score,
            'recall': metrics._recall_score,
            'prec': metrics._prec_score,
            'FNR': metrics._false_neg_rate,
            'TPR': metrics._true_pos_rate,
            'IoU': metrics._IoU_score,
            'total-error': metrics._total_error
        })

    msg = ['==> ']
    msg += [
        '{0} {1.avg:.3f}\t'.format(titleize(humanize(name)), meter)
        for name, meter in outputs.items()
    ]
    print(''.join(msg))


def predict(args):
    end = time.time()

    loader, _, model = _common(args, training=False)

    outputs = model.predict_loader(loader)

    # Transforming into string
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    outputs = 1.0 * (sigmoid(outputs) >= args.threshold)
    outputs = loader.dataset.binarizer.inverse_transform(outputs)
    tags = [' '.join(output) for output in outputs]

    # Saving data
    img_name = [
        os.path.splitext(os.path.basename(img_name))[0]
        for img_name, _ in loader.dataset.data
    ]

    print('=> writing results to {}'.format(args.save))
    # Saving file
    outdata = pd.DataFrame({'image_name': img_name, 'tags': tags})
    outdata.to_csv(args.save, header=True, index=False)
    print('=> results saved. Time {:.3f} s'.format(time.time() - end))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch NN')
    # Dataset
    parser.add_argument(
        '--img-dir',
        metavar='DIR',
        default='data/datafiles',
        help='path to dataset')
    parser.add_argument(
        '--manifest',
        type=str,
        metavar='MANIFEST',
        help='path to .csv',
        required=True)
    # Loader
    parser.add_argument(
        '-b',
        '--batch-size',
        default=32,
        type=int,
        metavar='N',
        help='mini-batch size (default: 32)')
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)')
    parser.add_argument(
        '-e',
        '--exceptions',
        default=[],
        nargs='+',
        help='classes to remove from model (default: None)')
    # Architecture
    parser.add_argument(
        '--arch',
        '-a',
        metavar='ARCH',
        default='toynet',
        choices=arch_names,
        help='model architecture: ' + ' | '.join(arch_names) +
        ' (default: toynet)')
    parser.add_argument(
        '--arch-params',
        metavar='PARAMS',
        default=[],
        nargs='+',
        type=str,
        help='model architecture params')
    parser.add_argument(
        '--shape',
        metavar='C,H,W',
        default=','.join(map(str, datasets.DEFAULT_SHAPE)),
        type=str,
        help='nb of channels, height and width of input image')
    # Loss
    parser.add_argument(
        '--loss',
        '--criterion',
        default='bce',
        type=str,
        choices=losses_names,
        help='losses: ' + ' | '.join(losses_names) + ' (default: bce)')
    # Optimizer
    parser.add_argument(
        '--optim',
        '--solver',
        default='adam',
        type=str,
        choices=optimizers.keys(),
        help='optimizers: ' + ' | '.join(sorted(optimizers.keys())) +
        ' (default: adam)')
    # Other params
    parser.add_argument(
        '--print-freq',
        '-p',
        default=100,
        type=int,
        metavar='N',
        help='print frequency (default: 100)')
    parser.add_argument(
        '--no-cuda', dest='cuda', action='store_false', help='use GPU')
    parser.add_argument(
        '--load',
        default=None,
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)')

    subparsers = parser.add_subparsers()

    # Train parser
    tr_parser = subparsers.add_parser('train', help='Pytorch training')

    tr_parser.add_argument(
        '--augmentation',
        '--aug',
        action='store_true',
        # nargs='*',
        help='specify which data augmetantion methods to use')
    tr_parser.add_argument(
        '--shrink-negatives',
        '--shrink',
        action='store_true',
        help='specify if negative only imgs should be removed from training')
    tr_parser.add_argument(
        '--val_manifest',
        '--val',
        type=str,
        metavar='VAL',
        help='path to val.csv')
    tr_parser.add_argument(
        '--epochs',
        default=90,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    tr_parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)')

    # Hyperparameters
    tr_parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.1,
        type=float,
        metavar='LR',
        help='initial learning rate')
    tr_parser.add_argument(
        '--lr-factor',
        '--learning-decay',
        default=2,
        type=float,
        metavar='LRF',
        help='learning rate decay factor')
    tr_parser.add_argument(
        '--lr-span',
        '--lr-time',
        default=10,
        type=float,
        metavar='LRS',
        help='time span for each learning rate step')
    tr_parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M', help='momentum')
    tr_parser.add_argument(
        '--weight-decay',
        '--wd',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)')

    # Visdom configuration
    tr_parser.add_argument('--visdom', action='store_true', help='use visdom')
    tr_parser.add_argument(
        '--env',
        type=str,
        default=getpass.getuser(),
        help='visdom environment '
        '(default:  {})'.format(getpass.getuser()))
    tr_parser.add_argument(
        '--save',
        type=str,
        default='models/checkpoint.pth.tar',
        help='name of the saved model')
    tr_parser.set_defaults(func=train)

    # Eval parser
    eval_parser = subparsers.add_parser('eval', help='Pytorch evaluation')
    eval_parser.set_defaults(func=eval)

    # Predict parser
    predict_parser = subparsers.add_parser(
        'predict', help='Pytorch prediction')
    predict_parser.add_argument(
        '--threshold',
        '--thrs',
        default=0.5,
        type=float,
        metavar='T',
        help='threshold (default: 0.5)')
    predict_parser.add_argument(
        '--save',
        type=str,
        default='submission.csv',
        help='name of the saved model')

    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)
