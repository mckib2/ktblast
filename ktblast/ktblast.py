'''Python implmentation of k-t BLAST algorithm.'''

import numpy as np
from skimage.filters import threshold_otsu

from utils import kt2xf

def ktblast(kspace, calib, psi=0.01, R=None, time_axis=-1):
    '''Cartesian k-t BLAST.

    Parameters
    ----------
    kspace : array_like
        Undersampled k-space data.  Nonsampled points should be
        exactly 0.  Assumes a sheared lattice sampling grid.
    calib : array_like
        Prior info, usually low-res estimate.
    psi : float, optional
        Noise variance.
    R : int, optional
        Undersampling factor.  Estimated from PSF if not provided.
        If this function gets R wrong, bad things will happen.  Check
        to make sure and provide it if necessary.
    time_axis : int, optional
        Dimension that holds time/frequency data.

    Returns
    -------
    recon : array_like
        Reconstructed x-t space.

    Raises
    ------
    AssertionError
        PSF of k-t grid finds more or less than R aliased copies.
        Only raises if R provided.

    Notes
    -----
    Implements the k-t BLAST algorithm as first described in [1]_.
    The Wiener filter expression is given explicitly in [2]_ (see
    equation 1).

    References
    ----------
    .. [1] Tsao, Jeffrey, Peter Boesiger, and Klaas P. Pruessmann.
           "k‐t BLAST and k‐t SENSE: dynamic MRI with high frame rate
           exploiting spatiotemporal correlations." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           50.5 (2003): 1031-1042.
    .. [2] Sigfridsson, Andreas, et al. "Improving temporal fidelity
           in k-t BLAST MRI reconstruction." International Conference
           on Medical Image Computing and Computer-Assisted
           Intervention. Springer, Berlin, Heidelberg, 2007.
    '''

    # Move time axis to end
    kspace = np.moveaxis(kspace, time_axis, -1)
    calib = np.moveaxis(calib, time_axis, -1)

    # Put everything into x-f space
    xf_aliased = kt2xf(kspace, shift=True)
    xf_prior = kt2xf(calib, shift=True)

    # Make sure psi is real (np.cov() can return complex numbers...)
    psi = np.abs(psi)

    # Get sizes
    cx, cy, ct = xf_prior.shape[:]

    # Get PSF of the sampling grid -- don't fftshift because the
    # coordinate system needs to assume center is (0, 0, 0)
    PSF = np.abs(kt2xf(np.abs(kspace) > 0, shift=False))

    # Get indices of locations of aliased copies, should only be R
    # of these
    if R is not None:
        thresh = np.sort(PSF.flatten())[-1*R]
        PSF[PSF < thresh] = 0
        idx = np.where(PSF > 0)
        assert np.stack(idx).shape[1] == R, (
            'PSF should define R copies!')
    else:
        thresh = threshold_otsu(PSF)
        idx = np.where(PSF > thresh)
        R = len(idx[0])
        print('Based on PSF, R is found to be: %d' % R)

    # calculate filter (Equation 1 in [2]) -- first get denominator
    axf_prior2 = np.abs(xf_prior)**2
    filter_denom = np.zeros((cx, cy, ct))
    for ii in range(R):
        filter_denom += np.roll(
            axf_prior2, (idx[0][ii], idx[1][ii], idx[2][ii]))
    filter_denom += psi

    # now divide numerator
    xf_filter = axf_prior2/filter_denom

    # multiply aliased data by filter, rescale, move time axis back
    return np.moveaxis(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(
        xf_aliased*xf_filter*R,
        axes=-1), axis=-1), axes=-1), -1, time_axis)


if __name__ == '__main__':
    pass
