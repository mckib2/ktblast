'''ESMRMB k-t methods tutorial.'''

import numpy as np

def kt_undersample_2d(R, dkydt=1):
    '''Simple undersampling routine for x-y-t data.

    Parameters
    ----------
    R : int
        Undersampling factor in y.
    dkydt : int, optional
        Lattice gradient.

    Returns
    -------
    kt_pat : array_like
        2D undersampling pattern.

    Notes
    -----
    Will only undersample in y, but allows changes to dkydt. Makes
    small-cell pattern.
    '''

    # define basic sampling tile
    kt_pat = np.zeros((R, R), dtype=bool)

    # set one ky sample to 1, all others 0
    kt_pat[0, :] = True

    # shift this sample through time
    for t in range(1, R):
        kt_pat[:, t] = np.roll(kt_pat[:, 0], (dkydt*t, 0))

    return kt_pat
