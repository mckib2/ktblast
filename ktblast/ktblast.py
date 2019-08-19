'''Python implmentation of k-t BLAST algorithm.'''

import numpy as np

from utils import kt2xf

def ktblast(
        xf_aliased, xf_prior, kt_grid, R, psi=0.01, time_axis=-1):
    '''k-t BLAST.

    Parameters
    ----------

    Returns
    -------

    Raises
    ------
    AssertionError
        PSF of k-t grid finds more or less than R aliased copies.
    '''

    # Move time axis to end
    xf_aliased = np.moveaxis(xf_aliased, time_axis, -1)
    xf_prior = np.moveaxis(xf_prior, time_axis, -1)

    # Get sizes
    cx, cy, ct = xf_prior.shape[:]

    # Make sure psi is real (np.cov() can return complex numbers...)
    psi = np.abs(psi)

    # Get PSF of the sampling grid
    PSF = kt2xf(kt_grid)

    # Get indices of locations of aliased copies, should only be R
    # of these
    idx = np.where(PSF > 1e-5)
    assert np.stack(idx).shape[1] == R, 'PSF should define R copies!'

    # calculate filter -- first get the denominator
    axf_prior2 = np.abs(xf_prior)**2
    filter_denom = np.zeros((cx, cy, ct))
    for ii in range(R):
        filter_denom += np.roll(
            axf_prior2, (idx[0][ii], idx[1][ii], idx[2][ii]))
    filter_denom += psi

    # now divide numerator
    xf_filter = axf_prior2/filter_denom

    # multiply aliased data by filter, rescale, move time axis back
    return np.moveaxis(xf_aliased*xf_filter*R, -1, time_axis)


if __name__ == '__main__':
    pass
