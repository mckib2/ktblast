'''Basic demo of UNFOLD method.'''

import numpy as np
from phantominator import dynamic

from ktblast import unfold

if __name__ == '__main__':

    # Load phantom
    ph = dynamic(256, 40)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Shear grid, R=2
    kspace_u = np.zeros(kspace.shape, dtype=kspace.dtype)
    kspace_u[0::2, :, 0::2] = kspace[0::2, :, 0::2]
    kspace_u[1::2, :, 1::2] = kspace[1::2, :, 1::2]

    # Run UNFOLD algorithm
    recon = unfold(kspace_u)
