'''Python k-t BLAST implementation.'''

import numpy as np
from skimage.util import pad
from tqdm import trange

def ktblast(kspace, calib, calib_win=None, freq_win=None,
            safety_margin=2, time_axis=-1):
    '''k-t BLAST.

    Parameters
    ----------
    kspace : array_like
        Undersampled k-space time frames to be reconstructed.
        Unsampled pixels should be exactly zero.
    calib : array_like
        Training stage data (calibration).  Time frames containing
        only the center of k-space.
    calib_win : array_like, optional
        2D window to apply to calibration k-space data before
        zero-padding and inverse Fourier transformation.
    freq_win : array_like, optional
        1D window to apply to x-f data to attenuate high temporal
        frequencies.
    safety_margin : float, optional
        Factor to multiply x-f data by.  Higher safety margins result
        in more image features being reconstructed at the expense of
        noise increase.  Default is 2 as suggested in [1]_.
    time_axis : int, optional
        Dimension corresponding to time.

    Notes
    -----
    Implements k-t BLAST algorithm described in [1]_.

    References
    ----------
    .. [1] Tsao, Jeffrey, Peter Boesiger, and Klaas P. Pruessmann.
           "k‐t BLAST and k‐t SENSE: dynamic MRI with high frame rate
           exploiting spatiotemporal correlations." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           50.5 (2003): 1031-1042.
    '''

    # Move time dimension to the back so we know where it is
    kspace = np.moveaxis(kspace, time_axis, -1)
    calib = np.moveaxis(calib, time_axis, -1)

    # Get sizes of matrices
    kx, ky, kt = kspace.shape[:]
    cx, cy, ct = calib.shape[:]

    # Get windows with which to filter zero-padded calib data and
    # temporal frequencies
    if calib_win is None:
        calib_win = 1
    else:
        assert calib_win.ndim == 2, 'calib_win must be 2D!'
        calib_win = calib_win[..., None]
    if freq_win is None:
        freq_win = 1
    else:
        assert freq_win.ndim == 1, 'freq_win must be 1D!'
        freq_win = freq_win[None, :]

    # We need a baseline estimate, so let's temporally average the
    # kspace and then inverse FFT
    kspace_avg = np.sum(kspace, axis=-1)
    kspace_avg /= np.sum(np.abs(kspace) > 0, axis=-1)
    imspace_avg = np.sqrt(kx*ky)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(kspace_avg)))

    # Subtract temporally averaged k-space from individual time frames
    resid = kspace - kspace_avg[..., None]

    # Inverse FFT the residual time frames, do in loop for memory
    # considerations...
    for ii in range(kt):
        resid[..., ii] = np.sqrt(kx*ky)*np.fft.fftshift(np.fft.ifft2(
            np.fft.ifftshift(resid[..., ii])))

    # Assume coil noise covariance is identity
    Cn = np.eye(kx)

    # Prepare the calibration data:
    # In-plane inverse Fourier transform of calibration data
    # Zero-padding: adds zeros around calibration data to match size
    # of kspace.  If difference in size between calib and kspace is
    # odd, we can't evenly pad calib, so we arbitrarily choose to
    # throw the leftovers on the left-hand side of the tuple in pad().
    axes = (0, 1)
    px, py = (kx - cx), (ky - cy)
    px2, py2 = int(px/2), int(py/2)
    adjx, adjy = np.mod(px, 2), np.mod(py, 2)
    fac = np.sqrt(kx*ky)
    lowres = fac*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        pad( #pylint: disable=E1102
            calib*calib_win,
            ((px2+adjx, px2), (py2+adjy, py2), (0, 0)),
            mode='constant'),
        axes=axes), axes=axes), axes=axes)

    # For each column...
    kt2 = int(kt/2)
    ct2 = int(ct/2)
    recon = np.zeros((kx, ky, kt), dtype=kspace.dtype)
    for ii in trange(ky, leave=False, desc='k-t BLAST'):
        # For DC baseline estimate, use the image column as f=0
        xf_base = np.zeros((kx, kt), dtype=kspace.dtype)
        xf_base[:, kt2] = imspace_avg[:, ii]

        # For the difference data, construct the x-t array and
        # inverse Fourier transform along t to get x-f array
        xf_resid = np.sqrt(kt)*np.fft.fftshift(np.fft.ifft(
            np.fft.ifftshift(resid[:, ii, :], axes=-1),
            axis=-1), axes=-1)

        # Now we need the same for the calibration data, set f=0 to 0
        xf_calib = np.sqrt(ct)*np.fft.fftshift(np.fft.ifft(
            np.fft.ifftshift(lowres[:, ii, :], axes=-1),
            axis=-1), axes=-1)
        xf_calib[:, ct2] = 0

        # Filter in f to attenuate high temporal frequencies and
        # safety margin
        xf_calib *= freq_win*safety_margin

        # Squared magnitude gives estimated squared deviation.
        # I don't know how this is actually supposed to work...
        # M2 = np.abs(xf_calib)**2
        M2 = np.abs(xf_calib @ xf_calib.conj().T)
        print(M2.shape)
        # M2 = M2[:, :kx]
        # M2 = np.diag(M2[:, ct2])

        # Don't worry about coil sensitivities for now k-t BLAST
        # recon[:, ii, :] = xf_base + M2 @ np.linalg.inv(
        #     M2 + Cn) @ (xf_resid - xf_base)
        # S = np.ones((kx, kx))
        # recon[:, ii, :] = xf_base + M2 @ S.T @ np.linalg.inv(
        #     S @ M2 @ S.T + Cn) @ (xf_resid - S @ xf_base)
        recon[:, ii, :] = M2 @ np.linalg.pinv(
            M2 + np.sum(
                np.abs(xf_resid)**2, axis=(0, 1))**2 + Cn) @ xf_resid
        # recon[: ii, :] = M2 @ np.linalg.pinv(M2 + .01*Cn) @ xf_resid
        # recon[:, ii, :] = xf_base

    # Fourier transform across frequency to get time back
    recon = 1/np.sqrt(kt)*np.fft.ifftshift(np.fft.fft(np.fft.fftshift(
        recon, axes=-1), axis=-1), axes=-1)

    # I don't think it worked -- we get the temporal average back I
    # think...

    # Move time_axis back to where the user had it
    # recon -= imspace_avg[..., None]
    return np.moveaxis(recon, -1, time_axis)

if __name__ == '__main__':
    pass
