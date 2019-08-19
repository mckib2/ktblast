'''k-t space to x-f space.'''

import numpy as np

def kt2xf(kt, coil_axis=-1):
    '''k-t space to x-f space.

    Parameters
    ----------
    kt : array_like
        k-t space data.
    time_axis : int, optional
        Dimension that holds time data.

    Returns
    -------
    xf : array_like
        Corresponding x-f space data.
    '''

    # Do the transformin' (also move coil axis to and fro)
    return np.moveaxis(np.fft.fft(np.fft.ifft2(
        np.moveaxis(kt, coil_axis, -1),
        axes=(0, 1)), axis=-1), -1, coil_axis)
