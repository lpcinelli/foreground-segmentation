# this file cross-validate the hyper-parameters
# it calls th main.lua

import argparse
import os
import numpy as np
import subprocess

parser = argparse.ArgumentParser(description='Cross validates hyperparameters of the resnet model')

# parameters for setting the learning rate
parser.add_argument('--lr_min', type=float, help='LOWER bound of the LEARNING RATE search space', nargs=1)
parser.add_argument('--lr_max', type=float, help='UPPER bound of the LEARNING RATE search space', nargs=1)
parser.add_argument('--lt_decay', type=int, help='number of epochs the exp. decay scheduler reduces the lr to 10%', nargs=1)

# parameters for setting the regularization (weight decay)
parser.add_argument('--reg_min', type=float, help='LOWER bound of the REGULARIZATION STRENGTH search space', nargs=1)
parser.add_argument('--reg_max', type=float, help='UPPER bound of the REGULARIZATION STRENGTH search space', nargs=1)

# parameters for generic training settings
parser.add_argument('--optim', type=str, help='The solver to use: SGD | Adam', nargs=1)
parser.add_argument('--epoch', type=int, help='Number of epochs the net will be trained', nargs=1)
parser.add_argument('--batch', type=int, help='Mini-batch size', nargs=1)
parser.add_argument('--cuda', type=int, help='Which cuda to use', nargs=1)
parser.add_argument('--trials', type=int, help='Number of trials of this experiment', nargs=1)

# network depth
parser.add_argument('--depth', type=int, help='determines the depth of the network: 22 | 34 | 46 | 58 | 112 | 1204', nargs=1)

#dir to where save the results and models
parser.add_argument('--save', type=str, help='dtermines the the dir in which to save the results', nargs=1)

args = parser.parse_args()


print(args.trials)

for trial in xrange(args.trials[0]): 
    print('trial nb.' + str(trial))
   
    sampled_learning_rate = 10**np.random.uniform(np.log10(args.lr_min)[0],np.log10(args.lr_max[0]))
    sampled_reg = 10**np.random.uniform(np.log10(args.reg_min)[0],np.log10(args.reg_max[0]))

    command = 'CUDA_VISIBLE_DEVICES=' + str(args.cuda[0])
    command = command + ' th main.lua -dataset cdnet -data ~/Documents/cdnet2014/deep-subtraction-split/ -nGPU 1 -nEpochs ' + str(args.epoch[0])
    command = command + ' -depth ' + str(args.depth[0])
    command = command + ' -batchSize ' + str(args.batch[0])
    command = command + ' -LR ' + str(sampled_learning_rate)
    command = command + ' -weightDecay ' + str(sampled_reg)
    # command = command + ' -LR_decay_step ' + str(args.lt_decay[0])
    # command = command + ' -model_init_LR ' + str(2*sampled_learning_rate)
    command = command + ' -save ' + args.save[0]
    command = command + ' -optimizer ' + args.optim[0]

    print(command)
#retval = subprocess.call('CUDA_VISIBLE_DEVICES=2 th main.lua -dataset cdnet -data ~/Documents/cdnet2014/sanity-check-split/ -nGPU 1 -nEpochs 15 -depth 34 -batchSize 20 -LR 0.1', shell=True)

    retval = subprocess.call(command, shell=True)
#for trial in xrange()
