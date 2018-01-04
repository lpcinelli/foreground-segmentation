''''Some helper functions for PyTorch
'''

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

def get_mean_and_std(dataset, axis=(0, 2, 3), batch_size=256, num_workers=2):
    '''
    Computes the mean and standard deviation with one pass.
    Uses Chan's pairwise algorithm.

    @param dataset Input data.
    @param axis Axis to be marginalized.
    @param batch_size Batch size.
    @param num_workers NUmber of parallel works.

    @return Returns mean and standard deviation.
    '''

    # Loading data
    data = DataLoader(dataset, batch_size=batch_size,
                      num_workers=num_workers)

    # Setting initial values
    fll_cn = 0
    fll_mu = 0.0
    fll_sd = 0.0
    shape = None

    # Computing mean and std using Chan's algorithm
    for i, (images, _) in enumerate(data):

        # Testing for tensors
        if torch.is_tensor(images):
            images = images.numpy()

        # Testing sample
        if i == 0:

            # Initializing
            shape = np.array(images.shape)
            shape = shape[np.setdiff1d(np.arange(images.ndim), axis)]
            fll_cn = images.size/np.prod(shape)
            fll_mu = np.mean(images, axis=axis)
            fll_sd = np.std(images, axis=axis)

        else:

            # Computing current images statistics
            cur_cn = images.size/np.prod(shape)
            cur_mu = np.mean(images, axis=axis)
            cur_sd = np.std(images, axis=axis)

            # Using parallel algorithm
            delta = fll_mu - cur_mu
            cur_ss = cur_sd*(cur_cn-1)
            fll_ss = fll_sd*(fll_cn-1)
            fll_mu = (fll_mu*fll_cn +cur_mu*cur_cn)/(fll_cn+cur_cn)
            fll_ss += cur_ss+(delta**2)*(cur_cn*fll_cn/(fll_cn+cur_cn))
            fll_cn += cur_cn
            fll_sd = fll_ss/(fll_cn-1)

    # Return
    return fll_mu, fll_sd
