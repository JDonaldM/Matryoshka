import numpy as np
from numba import jit
from hmf.density_field.filters import TopHat
import scipy.special as sp
import astropy.units as units
from scipy import constants as consts
from hmf.density_field.halofit import halofit


def unnormed_P(k, T, ns):
    '''
    Returns the un-normalised primordial power spectrum.
     (From halomod)
    '''
    return k.reshape(1, -1)**ns.reshape(-1, 1)*T**2


def norm(k, unnormed_P, sigma8):
    '''
    Calculates the normalisation for the primordial power spectrum based on a given value of sigma_8
     (From halomod)
    '''
    filt = TopHat(k, unnormed_P)
    return (sigma8/filt.sigma(8)).reshape(-1, 1)


def power0(unnormed_P, norm):
    '''
    Returns the linear power spectrum at redshift 0.
     (From halomod)
    '''
    return unnormed_P*norm**2


def power0_v2(k, T, sigma8, ns, growth_factor=1.):
    '''
    Returns the linear power spectrum at redshift 0.
     (From halomod)
    '''
    p = unnormed_P(k, T, ns)
    return p*norm(k, p, sigma8)**2*growth_factor**2


@jit(nopython=True)
def _p(K, bs, bc, asi, ac, c):
    '''
    Component of ukm
     see arXiv:astro-ph/0006319 eq. 11.
     (hacked halomod)
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
    Approximates the mass integral for various quanties used in the halomodel
    '''
    f_i = I[:, 1:]
    f_j = I[:, :-1]
    delta_m = np.diff(m)
    return ((f_i + f_j)/2 * delta_m).sum(axis=mass_axis)


def ngal(m, dndm, tot_occ):
    '''
    Calculates the mean number density of galaxies.
    '''
    I = dndm*tot_occ
    return mass_int(m, I, 1)


def TinkerBias(sigma, delta_halo=200.0, delta_c=1.686):
    '''
    (halomod hack)
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


def cen_occ(M, M_min):
    '''
    Mean central galaxy occupation.
    '''
    return (M >= 10**M_min).astype(float)


def cen_Z09(M, logM_cut, sigma):
    return 0.5*sp.erfc((np.log(10**logM_cut)-np.log(M))/(np.sqrt(2)*sigma))


def sat_occ(M, M1, alpha):
    '''
    Mean satellite galaxy occupation.
    '''
    M1 = 10**M1
    return (M/M1)**alpha


def sat_Z09(M, logM1, alpha, kappa, logM_cut):
    mean = np.zeros_like(M)
    idx_nonzero = np.where(M - kappa*10**logM_cut > 0)[0]
    mean[idx_nonzero] = (
        (M[idx_nonzero]-kappa*10**logM_cut)/(10**logM1))**alpha
    return mean


def power_1h_ss(ukm, dndm, m, cen_occ, sat_occ, mean_tracer_den):
    '''
    Returns the sat-sat contribution to the 1halo term of the galaxy power spectrum
     (hacked halomod)
     Imposes central condition
    '''
    I = ukm**2 * dndm * sat_occ**2 * cen_occ
    return mass_int(m, I, 1)/mean_tracer_den**2


def power_1h_cs(ukm, dndm, m, cen_occ, sat_occ, mean_tracer_den):
    '''
    Returns the cen-sat contribution to the 1halo term of the galaxy power spectrum
     (hacked halomod)
     Imposes central condition
    '''
    cs_pairs = cen_occ*sat_occ
    I = dndm * 2 * cs_pairs * ukm
    return mass_int(m, I, 1)/mean_tracer_den ** 2


def power_2h(ukm, dndm, m, total_occ, mean_tracer_den, Plin, halo_bias):
    '''
    Returns the 2halo term of the galaxy power spectrum
    '''
    I = ukm * dndm * halo_bias * total_occ / mean_tracer_den
    return mass_int(m, I, 1)**2*Plin


def mean_density0(cosmo):
    """
    Mean density of universe at z=0, [Msun h^2 / Mpc**3]
     (halomod hack)
    """
    return (
        (cosmo.Om0 * cosmo.critical_density0 / cosmo.h ** 2)
        .to(units.Msun / units.Mpc ** 3)
        .value
    )


def mean_density0_v2(h, Om0):

    critdens_const = 3. / (8. * np.pi * consts.G * 1000)
    H0units_to_invs = (units.km / (units.s * units.Mpc)).to(1.0 / units.s)
    rho0_unit_conv = (units.gram/units.cm**3).to(units.Msun / units.Mpc ** 3)

    H0_s = 100*h * H0units_to_invs
    cd0value = critdens_const * H0_s ** 2

    return Om0 * cd0value / h ** 2 * rho0_unit_conv


def Tinkerfsigma(sigma, redshift):
    '''
    Tinker fitting function used to calculate the halo mass function from the
     mass variance.
     (halomod hack)
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
    halomod hack
    '''
    fs = Tinkerfsigma(np.sqrt(sigma**2*growth**2), redshift)
    return fs * mean_dens * np.abs(dlnsdlnm) / m ** 2


def delta_k(k, power):
    '''
    From halomod.
    '''
    return k ** 3 * power / (2 * np.pi ** 2)


def nonlinear_power(k, power, sigma8, redshift, cosmo):
    '''
    From halomod.
    '''
    dk = delta_k(k, power)
    dk_nl = halofit(k, dk, sigma8, redshift, cosmo, True)

    return k ** -3 * dk_nl * (2 * np.pi ** 2)


def halomodel_power(k, m, transfer, sigma, dlns, cosmo, sigma8, ns,
                    HOD, conc, growth, redshift, nonlinear=False):
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
    return P2h+P1h_cs+P1h_ss, n_t


def Gcov(P_k, k, dk, n, V):
    N_k = (4*np.pi*k**2*dk)/((2*np.pi)/V**(1./3.))**3
    B = (P_k**2 + (2*P_k)/n + 1/n**2)
    return 2/N_k*B
