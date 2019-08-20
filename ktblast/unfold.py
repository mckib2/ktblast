'''Python implementation of the UNFOLD algorithm.'''

import numpy as np

from utils import kt2xf

def unfold(kspace, time_axis=-1):
    '''UNaliasing by Fourier‐encoding Overlaps Using temporaL Dim.

    Parameters
    ----------
    kspace : array_like
        Undersampled data in k-t space.  Missing data points should
        be exactly 0.
    time_axis : int, optional
        Dimension that holds the time data.

    Returns
    -------
    recon : array_like
        Reconstructed image space data (in x-t space).

    Notes
    -----
    Implements the algorithm described in [1]_.

    References
    ----------
    .. [1] Tsao, Jeffrey. "On the UNFOLD method." Magnetic Resonance
           in Medicine: An Official Journal of the International
           Society for Magnetic Resonance in Medicine 47.1 (2002):
           202-207.
    '''

    # Move time axis to the back of the bus
    kspace = np.moveaxis(kspace, time_axis, -1)
    sx, sy, st = kspace.shape[:]
    sx4, sy4 = int(sx/4), int(sy/4)
    st2 = int(st/2)

    # 1) zero‐fill the unacquired phase‐encode lines.
    # This is the assumed form for the input kspace data.

    # 2) apply the inverse Fourier transform along the spatial and
    # temporal directions
    xf_u = kt2xf(kspace, shift=True)

    # 3) set all signals outside the support region in x‐f‐space to
    # zero (this is equivalent to the filtering in UNFOLD)
    mask = np.zeros((sx, sy), dtype=bool)
    mask[sx4:-sx4, sy4:-sy4] = True
    xf = xf_u*mask[..., None]

    # Make some minor adjustments due to k-t demo written by Taehoon
    # Shin, found at:
    #     http://ee-classes.usc.edu/ee591/matlab.html
    # It doesn't work if you don't do these, but I'm not sure where
    # they come from...
    xf[..., 0] = 0
    xf[..., st2] = xf_u[..., st2]

    # 4) apply the Fourier transform along the temporal direction to
    # obtain images in x‐t‐space.
    return np.moveaxis(np.fft.ifft(np.fft.ifftshift(
        xf, axes=-1), axis=-1), -1, time_axis)
