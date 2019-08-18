'''Based on JEMRIS script.'''

import numpy as np
import matplotlib.pyplot as plt

def ktBLAST(xf_data, training_data, kt_grid, R, psi=0.01):

    xs, ys, ts = xf_data.shape[:]

    PSF = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(kt_grid, axes=(0, 1)),
        axes=(0, 1)), axes=(0, 1)), axis=-1), axes=-1)
    xShift, yShift, fShift = np.unravel_index(
        np.argwhere(PSF > 1e-5), PSF.shape)

    ## calculate filter
    # first get the denominator
    filter_denom = np.zeros((xs, ys, ts), dtype=psi.dtype)
    for ii in range(R):
        filter_denom += np.roll(np.abs(training_data)**2, (
            xShift.flatten()[ii],
            yShift.flatten()[ii],
            fShift.flatten()[ii]))
    filter_denom += psi

    # now divide numerator
    xf_filter = (np.abs(training_data)**2)/filter_denom

    # multiply aliased data by filter
    recon = xf_data*xf_filter

    # rescale
    recon *= R

    return(recon, xf_filter)

if __name__ == '__main__':

    kspace = np.load('data/phantom.npy')
    imspace = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sx, sy, st = kspace.shape[:]

    # Undersample
    kspace_u = kspace.copy()
    kspace_u[0::2, :, 0::2] = 0
    kspace_u[1::2, :, 1::2] = 0

    # Get xy-f space
    xyt_u = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace_u, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    xyf_u = np.fft.ifftshift(np.fft.fft(xyt_u, axis=-1), axes=-1)

    # noise statistics
    lin = np.squeeze(xyf_u[0, :, 1])
    psi = np.cov(lin)

    # low resolution prior
    mask = np.zeros(xyf_u.shape, dtype=bool)
    sx4 = int(sx/4)
    sy4 = int(sy/4)
    mask[sx4:-sx4, sy4:-sy4, :] = True
    xyf_recon = xyf_u*mask
    sf = 1 # scaling factor for prior

    xf_ktblast, ktblast_filter = ktBLAST(
        xyf_u, xyf_recon*sf, kspace_u == 0, 2, psi)

    imspace_recon = np.fft.ifft(np.fft.ifftshift(
        xf_ktblast, axes=-1), axis=-1)

    print(imspace_recon.shape)
    plt.imshow(np.abs(imspace_recon[..., 1]))
    plt.show()
