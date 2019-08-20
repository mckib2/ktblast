'''Basic demo of UNFOLD method.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    t0 = time()
    recon = unfold(kspace_u)
    print('Done with recon in %g sec' % (time() - t0))

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
