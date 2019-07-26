'''Basic usage of k-t BLAST implementation.'''

import numpy as np
from phantominator import shepp_logan

from ktblast import ktblast

if __name__ == '__main__':

    N = 64
    ph = shepp_logan(N)
    ph = (ph + 1j*ph)*N
    kt = 100
    ct = 64

    # Bring into k-space with the desired number of time frames
    axes = (0, 1)
    _kspace = 1/N*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=axes), axes=axes), axes=axes)
    kspace = np.tile(_kspace[..., None], (1, 1, kt))

    # Simple intensity variation in time
    f = 2
    tt = np.linspace(0, 2*np.pi, kt+ct+1)[1:]
    kspace *= np.sin(tt[ct:]*f)[None, None, :]

    # crop 20 lines from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = np.tile(_kspace[ctr-pd:ctr+pd, :, None], (1, 1, ct))
    calib *= np.sin(tt[:ct]*f)[None, None, :]
    print(calib.shape)

    # Undersample kspace: R=4
    mask = np.zeros(kspace.shape, dtype=bool)
    mask[0::4, :, 0::4] = True
    mask[1::4, :, 1::4] = True
    mask[2::4, :, 2::4] = True
    mask[3::4, :, 3::4] = True
    kspace *= mask

    # from mr_utils import view
    # view(np.abs(kspace[:, 0, :]) > 0)
    # assert False

    # Run k-t BLAST
    calib_win = np.hanning(calib.shape[0])[:, None] # PE direction
    freq_win = np.hanning(ct)
    # calib_win = None
    # freq_win = None
    recon = ktblast(
        kspace, calib, calib_win=calib_win, freq_win=freq_win)

    ctr = int(N/2)
    from mr_utils import view
    view(recon, fft_axes=(0, 1), movie_axis=2)

    import matplotlib.pyplot as plt
    plt.plot(np.abs(recon[ctr, ctr, :]))
    plt.plot(np.abs(ph[ctr, ctr, None]*np.sin(tt[ct:]*f)))
    plt.show()
