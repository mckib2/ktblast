'''Example using JEMRIS data.'''

import numpy as np
from scipy.io import loadmat

from ktblast import ktblast
from utils import kt_undersample_2d

if __name__ == '__main__':

    # Get aliased data
    xf_aliased = loadmat('data/xf_aliased.mat')['xf_aliased']
    sx, sy, st = xf_aliased.shape[:]

    R = 2
    dkydt = 1
    kt_tile = kt_undersample_2d(R, dkydt)
    kt_grid = np.tile(kt_tile[None, ...], (sx, int(sy/R), int(st/R)))

    # Check kt_grid
    kt_grid_mat = loadmat('data/kt_grid.mat')['kt_grid']
    assert np.allclose(kt_grid_mat, kt_grid)

    # Load prior
    xf_prior = loadmat('data/xf_prior.mat')['xf_prior']
    sf = 1

    # Run k-t BLAST reconstruction -- we'll need noise statistics
    lin = np.squeeze(xf_aliased[0, :, 1])
    psi = np.cov(lin)
    recon = ktblast(xf_aliased, xf_prior*sf, kt_grid, R, psi)
    recon = np.fft.ifft(recon, axis=-1)

    from mr_utils import view
    view(np.fft.fftshift(recon, axes=(0, 1)))
