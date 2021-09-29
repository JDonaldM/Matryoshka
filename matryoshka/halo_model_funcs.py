import numpy as np
from numba import jit
#from hmf.density_field.filters import TopHat
import scipy.special as sp
import astropy.units as units
from scipy import constants as consts
#from hmf.density_field.halofit import halofit
from .halofit import halofit
from scipy.integrate import simps


def unnormed_P(k, T, ns):
    '''
    Function to calculate the un-normalised primordial power spectrum.
    (hacked halomod).

    Args:
        k (array) : k-bins of the transfer function(s).
        T (array) : Transfer functions(s).
        ns (array) : Value(s) of the spectral index.

    Returns:
        Array of the un-normalised power spectrum/spectra.
    '''
    return k.reshape(1, -1)**ns.reshape(-1, 1)*T**2


def norm(k, unnormed_P, sigma8):
    '''
    Calculates the normalisation for the primordial power spectrum based on a
    given value of sigma_8 (hacked halomod).

    Args:
        k (array) : k-bins of the un-normalised power spectrum/spectra.
        unnormed_P (array) : Un-normalised power spectrum/spectra.
        sigma8 (array) : Value(s) of sigma_8

    Returns:
        Array of the normalisation for the power spectra.
    '''
    filt = TopHatrep(k, unnormed_P)
    return (sigma8/filt.sigma(8)).reshape(-1, 1)


def power0_v2(k, T, sigma8, ns):
    '''
    Calculate the normalised linear power spectrum at redshift 0.
    (hacked halomod)

    Args:
        k (array) : k-bins of the transfer function(s).
        T (array) : Transfer functions(s).
        sigma8 (array) : Value(s) of sigma_8
        ns (array) : Value(s) of the spectral index.

    Returns:
        Array of the normalised power spectrum/spectra at redshift 0.
    '''
    p = unnormed_P(k, T, ns)
    return p*norm(k, p, sigma8)**2


@jit(nopython=True)
def _p(K, bs, bc, asi, ac, c):
    '''
    Component of ukm see arXiv:astro-ph/0006319 eq. 11. (from halomod)
    '''
    return np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) + np.cos(K) * (ac - bc)


@jit(nopython=True)
def _r_s(m, mean_dens, c, delta_halo=200.0):
    '''
    Calculates the scale radius from the halo mass.
    (hacked halomod)
    '''
    return ((3 * m / (4 * np.pi * delta_halo * mean_dens)) ** (1.0 / 3.0))/c


def u(k, m, c, mean_dens, delta_halo=200.0):
    '''
    NFW in fourier space
    (hacked halomod)
    '''
    r_s = _r_s(m, mean_dens, c, delta_halo)
    K = np.outer(k, r_s)

    bs, bc = sp.sici(K)
    asi, ac = sp.sici((1 + c) * K)
    p = _p(K, bs, bc, asi, ac, c)

    u = p / (np.log(1.0 + c) - c / (1.0 + c))

    return np.squeeze(u)


@jit(nopython=True)
def mass_int(m, I, mass_axis=1):
    '''
    Approximates the mass integral for various quanties used in the halomodel.

    Args:
        m (array) : Array containing the masses over which to do the integral.
        I (array) : The integrand.
        mass_axis (int) : Index of the integrad the corresponds to the mass
         axis.

    Returns:
        The result of the mass integration.
    '''
    f_i = I[:, 1:]
    f_j = I[:, :-1]
    delta_m = np.diff(m)
    return ((f_i + f_j)/2 * delta_m).sum(axis=mass_axis)


def ngal(m, dndm, tot_occ):
    '''
    Calculates the mean number density of galaxies.

    Args:
        m (array) : Array containing the masses.
        dndm (array) : Arracy containing the halo mass function corresponding to
         the masses.
        tot_occ (array) : The expected halo occupation corresponding to the
         masses.

    Returns:
        The mean galaxy occupation.
    '''
    I = dndm*tot_occ
    return mass_int(m, I, 1)


def TinkerBias(sigma, delta_halo=200.0, delta_c=1.686):
    '''
    (halomod hack)

    Args:
        sigma (array) : Array containing the mass varaince.
        delta_halo (float) : The delta used in the halo mass defenition (default
         is 200.0).
        delta_c (float) : Critical density for collapse (default is 1.686)

    Returns:
        Halo bias as a function of mass.
    '''
    nu = (delta_c / sigma)
    y = np.log10(delta_halo)
    A = 1.0 + 0.24 * y * np.exp(-((4 / y) ** 4))
    a = 0.44 * y - 0.88
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4 / y) ** 4))
    B = 0.183
    c = 2.4
    b = 1.5
    return (
        1 - A * nu ** a / (nu ** a + delta_c ** a) + B * nu ** b + C * nu ** c
    )


def cen_Z09(M, logM_cut, sigma):
    '''
    Args:
        M (array) : Array of masses.
        logM_cut (float) : Minimum mass for halo to host a central galaxy.
        sigma (float) : Smoothing factor for central step function.

    Returns:
        The expected central occupation corresponding to the input masses.
    '''
    return 0.5*sp.erfc((np.log(10**logM_cut)-np.log(M))/(np.sqrt(2)*sigma))


def sat_Z09(M, logM1, alpha, kappa, logM_cut):
    '''
    Args:
        M (array) : Array of masses.
        logM1 (float) : Typical mass of halo to host a satellite.
        alpha (float) : Exponent of power law that defines how the expected
         number of sattelites grows with mass.
        kappa (float) : The product kappa*logM_cut defines the mimum mass for a
         halo to host a sattelite.
        logM_cut (float) : Minimum mass for halo to host a central galaxy.

    Returns:
        The expected sattelite occupation corresponding to the inputs masses.
    '''
    mean = np.zeros_like(M)
    idx_nonzero = np.where(M - kappa*10**logM_cut > 0)[0]
    mean[idx_nonzero] = (
        (M[idx_nonzero]-kappa*10**logM_cut)/(10**logM1))**alpha
    return mean


def power_1h_ss(ukm, dndm, m, cen_occ, sat_occ, mean_tracer_den):
    '''
    Returns the sat-sat contribution to the 1halo term of the galaxy power spectrum.
    (hacked halomod). Imposes central condition

    Args:
        ukm (array) : The halo profile in Fourier space.
        dndm (array) : The halo mass function.
        m (array) : Masses corresponding to the halo mass function and halo
         profile.
        cen_occ (array) : The expected central occupation corresponding to m.
        sat_occ (array) : The expected central occupation corresponding to m.
        mean_tracer_den (float) : The mean galaxy number density.

    Returns:
        Contribution to the 1h-term from sat-sat pairs.
    '''
    I = ukm**2 * dndm * sat_occ**2 * cen_occ
    return mass_int(m, I, 1)/mean_tracer_den**2


def power_1h_cs(ukm, dndm, m, cen_occ, sat_occ, mean_tracer_den):
    '''
    Returns the cen-sat contribution to the 1halo term of the galaxy power spectrum.
    (hacked halomod). Imposes central condition

    Args:
        ukm (array) : The halo profile in Fourier space.
        dndm (array) : The halo mass function.
        m (array) : Masses corresponding to the halo mass function and halo
         profile.
        cen_occ (array) : The expected central occupation corresponding to m.
        sat_occ (array) : The expected central occupation corresponding to m.
        mean_tracer_den (float) : The mean galaxy number density.

    Returns:
        Contribution to the 1h-term from cen-sat pairs.
    '''
    cs_pairs = cen_occ*sat_occ
    I = dndm * 2 * cs_pairs * ukm
    return mass_int(m, I, 1)/mean_tracer_den ** 2


def power_2h(ukm, dndm, m, total_occ, mean_tracer_den, Plin, halo_bias):
    '''
    Returns the 2halo term of the galaxy power spectrum

    Args:
        ukm (array) : The halo profile in Fourier space.
        dndm (array) : The halo mass function.
        m (array) : Masses corresponding to the halo mass function and halo
         profile.
        tot_occ (array) : The expected occupation corresponding to m.
        mean_tracer_den (float) : The mean galaxy number density.
        Plin (array) : The linear power spectrum.
        halo_bias (array) : The halo bias corresponding to m.

    Returns:
        2h-term.
    '''
    I = ukm * dndm * halo_bias * total_occ / mean_tracer_den
    return mass_int(m, I, 1)**2*Plin


def mean_density0_v2(h, Om0):
    '''
    Calculate the mean density at redshift 0.

    Args:
        h (array) : Value(s) of h, of shape (n,).
        Om0 (array) : Value(s) of Om0, of shape (n,).

    Returns:
        Array of mean densities of shape (n,).
    '''

    critdens_const = 3. / (8. * np.pi * consts.G * 1000)
    H0units_to_invs = (units.km / (units.s * units.Mpc)).to(1.0 / units.s)
    rho0_unit_conv = (units.gram/units.cm**3).to(units.Msun / units.Mpc ** 3)

    H0_s = 100*h * H0units_to_invs
    cd0value = critdens_const * H0_s ** 2

    return Om0 * cd0value / h ** 2 * rho0_unit_conv


def Tinkerfsigma(sigma, redshift):
    '''
    Tinker10 fitting function used to calculate the halo mass function from the
    mass variance (halomod hack).

    Args:
        sigma (array) : Array containing the mass variance of shape (n,).
        redshift (float) : Value of the redshift.

    Returns:
        Array containing fsigma of shape (n,).
    '''
    A_0 = 1.858659e-01
    a_0 = 1.466904
    b_0 = 2.571104
    c_0 = 1.193958
    A_exp = 0.14
    a_exp = 0.06

    A = A_0 * (1 + redshift) ** (-A_exp)
    a = a_0 * (1 + redshift) ** (-a_exp)
    alpha = 10 ** (-((0.75 / np.log10(200 / 75.0)) ** 1.2))
    b = b_0 * (1 + redshift) ** (-alpha)
    c = c_0

    return (A * ((sigma / b) ** (-a) + 1) * np.exp(-c / sigma ** 2))


def hmf(sigma, dlnsdlnm, mean_dens, m, growth, redshift):
    '''
    Tinker08 halo mass function (halomod hack).

    Args:
        sigma (array) : Array containing the mass variance of shape (n,).
        dlnsdlnm (array) : Array containing the logarithmic derivative of the
         mass varaince of shape (n,).
        m (array) : Array containing the masses of shape (n,).
        growth (float) : Value of the growth function.
        redshift (float) : Value of the redshift.

    Returns:
        Array containing the halo mass function of shape (n,).
    '''
    fs = Tinkerfsigma(np.sqrt(sigma**2*growth**2), redshift)
    return fs * mean_dens * np.abs(dlnsdlnm) / m ** 2


def delta_k(k, power):
    '''
    From halomod.

    Args:
        k (array) : Array of k-bins of shape (n,).
        power (array) : Array containing the power spectrum corresponding to k
         of shape (n,).

    Returns:
        Array containing the dimensionless power spectrum of shape (n,).
    '''
    return k ** 3 * power / (2 * np.pi ** 2)


def nonlinear_power(k, power, sigma8, redshift, cosmo):
    '''
    Args:
        k (array) : Array of k-bins of shape (n,).
        power (array) : Array containing the power spectrum corresponding to k
         of shape (n,).
        sigma8 (float) : Value of sigma_8.
        redshift (float) : Value of redshift.
        cosmo (astropy FlatwCDM) : Astropy FlatwCDM cosmology object.

    Returns:
        Array containing the nonliear power spectrum of shape (n,).
    '''
    dk = delta_k(k, power)
    dk_nl = halofit(k, dk, sigma8, redshift, cosmo, True)

    return k ** -3 * dk_nl * (2 * np.pi ** 2)


def halomodel_power(k, m, transfer, sigma, dlns, cosmo, sigma8, ns,
                    HOD, conc, growth, redshift, nonlinear=False,
                    split_1h_2h=False):
    '''
    Covience function for calculating the galaxy power spectrum.

    Args:
        k (array) : Array of k-bins of shape (nk,).
        m (array) : Array of masses of shape (nm,).
        transfer (array) : Array containing the transfer function corresponding
         to k of shape (nk,).
        sigma (array) : Array containing the mass varaince corresponding to m of
         shape (nm,)
        dlns (array) : Array containing the logarithmic derivative of the mass
         variance of shape (nm,).
        cosmo (astropy FlatwCDM) : Astropy FlatwCDM cosmology object.
        sigma8 (float) : Value of sigma_8.
        ns (float) : Value of the spectral index.
        HOD (array) : Array containing the HOD parameters of shape (5,)
        conc (array) : Array containing the halo concentration corresponding to
         m of shape (nm,).
        growth (float) : Value of the growth function corresponding to the
         redshift.
        redshift (float) : redshift.
        nonlinear (bool) : If True nonlinearities included via HALOFIT.
        split_1h_2h (bool) : If True contributions from the 1-halo and 2-halo
         terms are returned seperately.

    Returns:
        The galaxy power spectrum of shape (nk,).
    '''
    # Calculate mask for the mass.
    tm = m >= HOD[0] - 5 * HOD[1]

    # Calculate the expected occupation from the HOD params.
    N_c = cen_Z09(m[tm], HOD[0], HOD[1])
    N_s = sat_Z09(m[tm], HOD[2], HOD[4], HOD[3], HOD[0])
    N_tot = N_c*(1+N_s)

    halo_bias = TinkerBias(np.sqrt(sigma**2*growth**2))
    mean_dens = mean_density0_v2(h=cosmo.H0/100., Om0=cosmo.Om0)
    hmf_matry = hmf(sigma, dlns, mean_dens, m, growth, redshift)

    u_m = u(k, m[tm], conc[tm], mean_dens, 200)

    n_t = ngal(m[tm].reshape(1, -1), hmf_matry[tm].reshape(1, -1),
               (N_tot).reshape(1, -1))[0]
    pm_matry = power0_v2(k, transfer, sigma8=sigma8, ns=ns)[0]
    if nonlinear:
        pm_matry = nonlinear_power(
            k, pm_matry*growth**2, sigma8, redshift, cosmo)
    else:
        pm_matry = pm_matry*growth**2
    P1h_ss = power_1h_ss(u_m, hmf_matry[tm], m[tm], N_c, N_s, n_t)
    P1h_cs = power_1h_cs(u_m, hmf_matry[tm], m[tm], N_c, N_s, n_t)
    P2h = power_2h(u_m, hmf_matry[tm], m[tm],
                   N_tot, n_t, pm_matry, halo_bias[tm])
    if split_1h_2h:
        return P1h_cs+P1h_ss, P2h, ns
    else:
        return P2h+P1h_cs+P1h_ss, n_t


def Gcov(P_k, k, dk, n, V):
    '''
    Calculate a Gaussian diagonal covariance.

    Args:
        P_k (array) : Array containing the power spectrum of shape (n,).
        k (array) : k-bins corresponding to the power spectrum of shape (n,).
        dk (float) : Width of the k-bins.
        n (float) : Number density.
        V (float) : Volume.

    Returns:
        Array containing Gaussian covariance of shape (n,).
    '''
    N_k = (4*np.pi*k**2*dk)/((2*np.pi)/V**(1./3.))**3
    B = (P_k**2 + (2*P_k)/n + 1/n**2)
    return 2/N_k*B


def Duffy08cmz(m, redshift):
    '''
    Replacement Class for ``halomod.concentration.Duffy08`` (hacked halomod)

    Assumes a spherical overdensity mass defention.

    Args:
        m (array) : Array containing masses at whcih to calculate the
         concentration.
        redshift (float) : Redshift at which to calculate the concentration.

    Returns:
        An array containing the halo concentrations.
    '''

    a = 11.93
    b = -0.09
    c = 0.99
    ms = 2e12

    return a / (1 + redshift) ** c * (m / ms) ** b


class TopHatrep:
    '''
    Replacement Class for ``hmf.density_field.filters.TopHat`` (hacked halomod)

    Args:
        k (array) : Array containing wave-numbers associated to ``power``.
        power (array) : Un-normalised power spectrum at redshift 0.
    '''

    def __init__(self, k, power):
        self.k = k
        self.power = power

    def sigma(self, r, order=0, rk=None):
        '''
        Mass variance.
        '''
        if rk is None:
            rk = np.outer(r, self.k)

        dlnk = np.log(self.k[1] / self.k[0])

        # we multiply by k because our steps are in logk.
        rest = self.power * self.k ** (3 + order * 2)
        integ = rest * self.k_space(rk) ** 2
        sigma = (0.5 / np.pi ** 2) * simps(integ, dx=dlnk, axis=-1)
        return np.sqrt(sigma)

    def k_space(self, kr):
        '''
        Top-hat window function in Fourier space.
        '''
        return np.where(kr > 1.4e-6, (3 / kr ** 3) * (np.sin(kr) - kr * np.cos(kr)), 1)
