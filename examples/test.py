'''Simple k-t accel demo.'''

import numpy as np

from mr_utils import view

if __name__ == '__main__':

    kspace = np.load('data/phantom.npy')
    imspace = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sx, sy, st = kspace.shape[:]
    view(imspace)

    # Undersample
    kspace_u = kspace.copy()
    kspace_u[0::2, :, 0::2] = 0
    kspace_u[1::2, :, 1::2] = 0
    view(kspace_u, fft=True)

    # Get xy-f space
    xyf_u = np.fft.ifftshift(np.fft.fft(kspace_u, axis=-1), axes=-1)
    xyf_u = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        xyf_u, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    view(xyf_u)

    # Filter xy-f space
    mask = np.zeros(xyf_u.shape, dtype=bool)
    sx4 = int(sx/4)
    sy4 = int(sy/4)
    mask[sx4:-sx4, sy4:-sy4, :] = True
    xyf_recon = xyf_u*mask

    # Make some minor adjustments...
    st2 = int(st/2)
    xyf_recon[..., 0] = 0
    xyf_recon[..., st2] = xyf_u[..., st2]

    # Put into xy-t space
    imspace_recon = np.fft.ifft(np.fft.ifftshift(
        xyf_recon, axes=-1), axis=-1)

    view(imspace)
    # view(imspace - 2*imspace_recon)
