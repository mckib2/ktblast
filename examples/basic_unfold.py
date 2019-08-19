'''Basic demo of UNFOLD method.'''

import numpy as np

from ktblast import unfold

if __name__ == '__main__':

    # Load phantom
    kspace = np.load('data/phantom.npy')

    # Shear grid, R=2
    kspace_u = np.zeros(kspace.shape, dtype=kspace.dtype)
    kspace_u[0::2, :, 0::2] = kspace[0::2, :, 0::2]
    kspace_u[1::2, :, 1::2] = kspace[1::2, :, 1::2]

    # Run UNFOLD algorithm
    recon = unfold(kspace_u)
