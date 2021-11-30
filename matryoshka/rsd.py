from re import X
import numpy as np
from scipy.special import eval_legendre
from scipy.integrate import trapz, quad
from scipy.interpolate import interp1d

def RSD(Pk, kbins, mu_bins, beta, poles=None, fog=True, sigma=None,
        qperp=None, qpar=None):
    '''
    Calculate power spectrum in redshift space from the real space
    galaxy power spectrum.

    Args:
        Pk (array) : Halo model prediction of the power spectrum.
        kbins (array) : k-bins associated to Pk.
        mu_bins (array) : The mu bins to be used.
        beta (float) : Ratio of the growth rate and galaxy bias.
        poles (array) : Multipoles to be calculated. If None the 2d power spectrum
         will be returned. Default is None.
         Default is None. Required if Pk is 1d.
        fog (bool) : Use finger of god damping.
        sigma (float) : Free parameter of fog model.
        qperp (float) : AP parameter. See eq. 16 in arXiv:2003.07956v2
        qpar (float) : AP parameter. See eq. 16 in arXiv:2003.07956v2

    Returns:
        Redshift space power spectrum. If poles is None will have shape (nk, nmu).
        If poles is not None will have shape (nl, nk).
    '''

    # Create 2d grids from the input k and mu vectors/
    kgrid, mugrid = np.meshgrid(kbins, mu_bins, indexing='ij')

    # If AP paramerers are not None apply the AP effect.
    if (qperp is not None) and (qpar is not None):
        
        F = qpar / qperp

        # Calaculate rescaled k and mu.
        # eq. 18 in arXiv:2003.07956v2
        kp = kgrid / qperp * (1 + mugrid**2 * (F**-2 - 1))**0.5
        mup = mugrid / F * (1 + mugrid**2 * (F**-2 - 1))**-0.5 

        # Evaluate the input 1d power spectrum at the rescaled kbins.
        Pk = interp1d(kbins, Pk, kind='cubic', bounds_error=False, fill_value='extrapolate')(kp)

        # Aplly Kaiser factor.
        _Pkmu = Pk*(1+beta*mup**2)**2

        if fog:
            # Apply simple damping term.
            _Pkmu = _Pkmu*np.exp(-(kp*mup*sigma)**2/2)

    else:

        _Pkmu = np.outer(Pk, eval_legendre(0, mu_bins))
        _Pkmu = (1+beta*mu_bins**2)**2*_Pkmu
        if fog:
            _Pkmu = _Pkmu*np.exp(-(np.outer(kbins, mu_bins)*sigma)**2/2)

    # If no multipoles list is passed just return the 2d power.
    if poles is None:
        return _Pkmu

    else:
        Pell = []
        for pole in poles:
            if (qperp is not None) and (qpar is not None):
                Interg = _Pkmu*eval_legendre(pole, mugrid)
                Pell.append((2*pole+1)/(2*qperp**2 * qpar)  * trapz(Interg, x=mugrid, axis=-1))
            else:
                Interg = _Pkmu*eval_legendre(pole, mu_bins)
                Pell.append((2*pole+1)/2  * trapz(Interg, x=mu_bins, axis=-1))
    return np.stack(Pell)

def reconstruct_from_multipoles(poles, mu_bins):
    '''
    Funtion for reconstructing the 2d power spectrum from multipoles.

    Args:
        poles (list) : List of arrays containg the multipoles to be used
         for the reconstruction.
        mu_bins (array) : The mu bins to be used.

    Returns:
        The reconstructed 2d power spectrum.
    '''
    power = 0
    for i, pole in enumerate(poles):
        power += np.outer(pole, eval_legendre(i*2, mu_bins))
    return power


def multipole(Pk, l, mu_bins):
    '''
    Calculate the redshift-space multipoles from the 2d power spectrum.

    Args:
        Pk (array) : Halo model prediction of the power spectrum.
        l (int) : Multipole order.
        mu_bins (array) : mu bins associated to Pk.

    Returns:
        lth order multipole.
    '''


    I = Pk*eval_legendre(l, mu_bins)
    return (2*l+1)/2 * trapz(I, x=mu_bins, axis=1)

def AP(lPk, mu_bins, kbins, qperp, qpar):
    '''
    Function to include the AP effect into multipoles. The function reconstructs
    the 2d power before including AP.

    Args:
        lPk (array) : Array containing the multipoles. Should have shape (nl, nk).
        mu_bins (array) : The mu bins used to reconstruct the 2d power and recompute
         the multipoles.
        kbins (array) : The k-bins associated to lPk.
        qperp (float) : AP parameter. See eq. 16 in arXiv:2003.07956v2
        qpar (float) : AP parameter. See eq. 16 in arXiv:2003.07956v2

    Returns:
        Array containing the multipoles with AP included. Will have the same shape
        as lPk.
    '''


    kgrid, mugrid = np.meshgrid(kbins, mu_bins, indexing='ij')

    F = qpar / qperp
    kp = kgrid / qperp * (1 + mugrid**2 * (F**-2 - 1))**0.5
    mup = mugrid / F * (1 + mugrid**2 * (F**-2 - 1))**-0.5

    Pkint = interp1d(kbins, lPk, kind='cubic', bounds_error=False, fill_value='extrapolate', axis=-1)(kp)
    print(Pkint.shape)

    _Pkmu = 0
    for i in range(lPk.shape[0]):
        l=i*2
        _Pkmu += Pkint[i]*eval_legendre(l, mup)

    Pell = []
    for pole in range(2):
        l=pole*2
        print(l)
        Interg = _Pkmu*eval_legendre(l, mugrid)
        Pell.append((2*l+1)/(2*qperp**2 * qpar)  * trapz(Interg, x=mugrid, axis=-1))
    return np.stack(Pell)

def Hubble(Om, z):
    """ LCDM AP parameter auxiliary function
    (from pybird) """
    return ((Om) * (1 + z)**3. + (1 - Om))**0.5

#TODO: Write version of this that expects muliple values of Om.
def DA(Om, z):
    """ LCDM AP parameter auxiliary function
    (from pybird) """
    r = quad(lambda x: 1. / Hubble(Om, x), 0, z)[0]
    return r / (1 + z)

def DA_vec(Om, z):
    z_eval = np.linspace(0,z,200)
    z_grid, Om_grid = np.meshgrid(z_eval, Om, indexing='ij')
    r = trapz(1. / Hubble(Om_grid, z_grid), x=z_eval, axis=0)
    return r/(1+z)
