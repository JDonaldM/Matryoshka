import numpy as np

def multipole(P_n, b, f):
    '''
    Calculates the galaxy power spectrum multipole given a P_n matrix 
    that corresponds to the desired multipole.

    Args:
        P_n (list) : List of arrays ``[P11, Ploop, Pct]``. The arrays
         should have shape ``(3, nk)``, ``(12, nk)``, and ``(6, nk)``
         respectively.
        b (array) : Array of bias parameters and counter terms.
        f (float) : Growth rate at the same redshift as ``P_n``.

    Returns:
        The galaxy multipole.
    '''

    # The block of code is a slightly modified version of 
    # the code in cell 21 of the example PyBird notebook 
    # run_pybird.ipynb
    b1, b2, b3, b4, b5, b6, b7 = b
    b11 = np.array([ b1**2, 2.*b1*f, f**2 ])
    bct = np.array([ 2.*b1*b5, 2.*b1*b6, 2.*b1*b7, 2.*f*b5, 2.*f*b6, 2.*f*b7 ])
    bloop = np.array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
    lin = np.einsum('b,bx->x', b11, P_n[0])
    loop = np.einsum('b,bx->x', bloop, P_n[1]) 
    counterterm = np.einsum('b,bx->x', bct, P_n[2])
    return lin + loop + counterterm