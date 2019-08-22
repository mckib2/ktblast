'''Basic usage of k-t BLAST implementation.'''

import numpy as np
from phantominator import dynamic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ktblast import ktblast
from utils import kt2xf

if __name__ == '__main__':

    N = 128
    nt = 40
    ph = dynamic(N, nt)
    ax = (0, 1)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # crop 20 lines from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = np.zeros(kspace.shape, dtype=kspace.dtype)
    calib[ctr-pd:ctr+pd, ...] = kspace[ctr-pd:ctr+pd, ...].copy()

    # Undersample kspace:
    mask = np.zeros(kspace.shape, dtype=bool)
    mask[0::2, :, 0::2] = True
    mask[1::2, :, 1::2] = True
    kspace *= mask

    # Get noise statistics from non-moving area
    lin = kt2xf(kspace, shift=True)[0, :, 1]
    psi = np.cov(lin)

    # Run k-t BLAST
    recon = ktblast(kspace, calib, psi=psi)

    # Some code to look at the animation
    fig = plt.figure()
    ax = plt.imshow(np.abs(recon[..., 0]), cmap='gray')

    def init():
        '''Initialize ax data.'''
        ax.set_array(np.abs(recon[..., 0]))
        return(ax,)

    def animate(frame):
        '''Update frame.'''
        ax.set_array(np.abs(recon[..., frame]))
        return(ax,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=ph.shape[-1],
        interval=40, blit=True)
    plt.show()

    # import matplotlib.pyplot as plt
    # plt.imshow(res)

    # N = 64
    # ph = shepp_logan(N)
    # ph = (ph + 1j*ph)*N
    # kt = 100
    # ct = 64
    #
    # # Bring into k-space with the desired number of time frames
    # axes = (0, 1)
    # _kspace = 1/N*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
    #     ph, axes=axes), axes=axes), axes=axes)
    # kspace = np.tile(_kspace[..., None], (1, 1, kt))
    #
    # # Simple intensity variation in time
    # f = 2
    # tt = np.linspace(0, 2*np.pi, kt+ct+1)[1:]
    # kspace *= np.sin(tt[ct:]*f)[None, None, :]
    #
    # # crop 20 lines from the center of k-space for calibration
    # pd = 10
    # ctr = int(N/2)
    # calib = np.tile(_kspace[ctr-pd:ctr+pd, :, None], (1, 1, ct))
    # calib *= np.sin(tt[:ct]*f)[None, None, :]
    # print(calib.shape)
    #
    # # Undersample kspace: R=4
    # mask = np.zeros(kspace.shape, dtype=bool)
    # mask[0::4, :, 0::4] = True
    # mask[1::4, :, 1::4] = True
    # mask[2::4, :, 2::4] = True
    # mask[3::4, :, 3::4] = True
    # kspace *= mask
    #
    # # from mr_utils import view
    # # view(np.abs(kspace[:, 0, :]) > 0)
    # # assert False
    #
    # # Run k-t BLAST
    # calib_win = np.hanning(calib.shape[0])[:, None] # PE direction
    # freq_win = np.hanning(ct)
    # # calib_win = None
    # # freq_win = None
    # recon = ktblast(
    #     kspace, calib, calib_win=calib_win, freq_win=freq_win)
    #
    # ctr = int(N/2)
    # from mr_utils import view
    # view(recon, fft_axes=(0, 1), movie_axis=2)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(np.abs(recon[ctr, ctr, :]))
    # plt.plot(np.abs(ph[ctr, ctr, None]*np.sin(tt[ct:]*f)))
    # plt.show()
