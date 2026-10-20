'''
gaiamockplus: extensions to gaiamock (El-Badry et al. 2024) for modeling the Gaia astrometry of exoplanet systems.

Everything in gaiamock is re-exported, so `import gaiamockplus as gaiamock` gives access to both the original
gaiamock functions and the additions below. New functionality lives in this file (rather than in gaiamock.py)
so that updates from the upstream gaiamock repository can be merged without conflicts.

Conventions follow gaiamock: times in days (Tp relative to the reference epoch of the data release), angles in
radians, masses in Msun, astrometry in mas. Thiele-Innes elements describe the photocenter (here, the host star's
reflex) orbit, so w is the argument of periastron of the star's orbit, which differs by pi from that of the planet.
'''
import os
import ctypes
import numpy as np
import gaiamock
from gaiamock import *

# Each planet is a dict with exactly these keys:
#     period: orbital period in days
#     Tp: periastron time in days, relative to the reference epoch of the data release
#     ecc: eccentricity, 0 <= ecc < 1
#     Omega: longitude of the ascending node ("big Omega") in radians
#     inc: inclination in radians, defined so that 0 or pi is face-on, and pi/2 is edge-on
#     w: argument of periastron ("little omega") of the star's reflex orbit, in radians
#     m_planet: planet mass in Msun
PLANET_KEYS = ('period', 'Tp', 'ecc', 'Omega', 'inc', 'w', 'm_planet')

def _get_rng(rng):
    '''
    Return the source of random numbers: a numpy Generator if one is passed, otherwise the global np.random
    state (as used throughout gaiamock, so that results match gaiamock for the same np.random.seed()).
    '''
    return np.random if rng is None else rng

def validate_planets(planets, require_Omega=True):
    '''
    Check a list of planet dicts (see PLANET_KEYS) and return a copy with all values converted to floats.
    Raises ValueError with an informative message for missing/unknown keys or unphysical values.
    require_Omega: if False, 'Omega' may be omitted (it does not affect radial velocities or reflex amplitudes).
    '''
    if isinstance(planets, dict):
        raise ValueError('planets must be a list of dicts (one per planet), not a single dict. Use [planet] for one planet.')
    required = PLANET_KEYS if require_Omega else tuple(k for k in PLANET_KEYS if k != 'Omega')
    checked = []
    for i, planet in enumerate(planets):
        if not isinstance(planet, dict):
            raise ValueError('planet %d is not a dict.' % i)
        if 'omega' in planet:
            raise ValueError("planet %d has key 'omega', which is ambiguous. Use 'Omega' for the longitude of the ascending node and 'w' for the argument of periastron." % i)
        missing = [k for k in required if k not in planet]
        unknown = [k for k in planet if k not in PLANET_KEYS]
        if missing or unknown:
            raise ValueError('planet %d: missing keys %s, unknown keys %s. Required keys: %s' % (i, missing, unknown, list(required)))
        p = {k: float(planet[k]) for k in PLANET_KEYS if k in planet}
        if not np.all(np.isfinite(list(p.values()))):
            raise ValueError('planet %d has non-finite values: %s' % (i, p))
        if p['period'] <= 0:
            raise ValueError('planet %d: period must be positive.' % i)
        if not (0 <= p['ecc'] < 1):
            raise ValueError('planet %d: ecc must satisfy 0 <= ecc < 1.' % i)
        if p['m_planet'] < 0:
            raise ValueError('planet %d: m_planet must be non-negative.' % i)
        checked.append(p)
    return checked

def get_scan_metadata(ra, dec, data_release, reject_10_percent=True, rng=None):
    '''
    Get the Gaia scan times, scan angles, and parallax factors for a sky position, as used by all of the
    predict_astrometry_* functions. Requires the healpix_scans/ directory (see README).
    ra, dec: coordinates in degrees
    data_release: 'dr3', 'dr4', or 'dr5'
    reject_10_percent: if True, randomly drop ~10% of FOV transits, as in gaiamock
    rng: optional numpy Generator; if None, the global np.random state is used
    returns t_ast_yr (years relative to the reference epoch), psi (radians), plx_factor
    '''
    t = gaiamock.get_gost_one_position(ra, dec, data_release=data_release)
    if reject_10_percent:
        t = t[_get_rng(rng).uniform(0, 1, len(t)) > 0.1]
    psi, plx_factor, jds = gaiamock.fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    t_ast_yr = gaiamock.rescale_times_astrometry(jd=jds, data_release=data_release)
    return t_ast_yr, psi, plx_factor

def add_epoch_noise(ast_true, phot_g_mean_mag, rng=None):
    '''
    Add gaiamock's per-FOV-transit noise model to noiseless along-scan astrometry: the per-CCD uncertainty
    divided by sqrt(8), plus unmodeled noise with sigma ~ U(0, 0.04) mas for sources brighter than G = 13.
    ast_true: noiseless along-scan positions in mas
    phot_g_mean_mag: G-band magnitude
    rng: optional numpy Generator; if None, the global np.random state is used
    returns ast_obs, ast_err (mas)
    '''
    rng = _get_rng(rng)
    N_ccd_avg = 8
    epoch_err_per_transit = gaiamock.al_uncertainty_per_ccd_interp(G=phot_g_mean_mag)/np.sqrt(N_ccd_avg)

    if phot_g_mean_mag < 13:
        extra_noise = rng.uniform(0, 0.04)
    else:
        extra_noise = 0

    ast_obs = ast_true + epoch_err_per_transit*rng.standard_normal(len(ast_true)) # modeled noise
    ast_obs += extra_noise*rng.standard_normal(len(ast_true)) # unmodeled noise
    return ast_obs, epoch_err_per_transit*np.ones(len(ast_obs))

def get_thiele_innes(a0_mas, Omega, inc, w):
    '''
    Convert Campbell elements to Thiele-Innes elements, using the same convention as gaiamock.
    a0_mas: photocenter semimajor axis in mas
    Omega, inc, w: longitude of ascending node, inclination, and argument of periastron in radians
    returns A, B, F, G in mas
    '''
    A = a0_mas*( np.cos(w)*np.cos(Omega) - np.sin(w)*np.sin(Omega)*np.cos(inc) )
    B = a0_mas*( np.cos(w)*np.sin(Omega) + np.sin(w)*np.cos(Omega)*np.cos(inc) )
    F = -a0_mas*( np.sin(w)*np.cos(Omega) + np.cos(w)*np.sin(Omega)*np.cos(inc) )
    G = -a0_mas*( np.sin(w)*np.sin(Omega) - np.cos(w)*np.cos(Omega)*np.cos(inc) )
    return A, B, F, G

def keplerian_offsets(t_ast_yr, period, Tp, ecc, A, B, F, G, c_funcs):
    '''
    Sky-plane displacement of the photocenter due to one Keplerian orbit (no parallax or proper motion).
    t_ast_yr: times in years relative to the reference epoch
    period, Tp: in days; ecc: eccentricity; A, B, F, G: Thiele-Innes elements in mas
    returns delta_ra_cosdec, delta_dec in mas
    '''
    EE = gaiamock.solve_kepler_eqn_on_array(M = 2*np.pi/period * (np.asarray(t_ast_yr, dtype=float)*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE)
    x, y = B*X + G*Y, A*X + F*Y
    return x, y

def keplerian_al_signal(t_ast_yr, psi, period, Tp, ecc, A, B, F, G, c_funcs):
    '''
    Along-scan displacement due to one Keplerian orbit, i.e. keplerian_offsets() projected onto the scan
    direction psi (radians). Returns an array in mas.
    '''
    x, y = keplerian_offsets(t_ast_yr = t_ast_yr, period = period, Tp = Tp, ecc = ecc, A = A, B = B, F = F, G = G, c_funcs = c_funcs)
    return y*np.cos(psi) + x*np.sin(psi)

def _reflex_a0(m_star, planets, parallax, jacobi):
    '''Reflex semimajor axes (mas) for an already-validated list of planets; see get_planet_a0_mas().'''
    if len(planets) == 0:
        return np.zeros(0)
    periods = np.array([p['period'] for p in planets])
    masses = np.array([p['m_planet'] for p in planets])

    central_mass = np.full(len(planets), float(m_star))
    if jacobi:
        order = np.argsort(periods, kind='stable') # ties keep the order of the list
        central_mass[order] = m_star + np.cumsum(masses[order]) - masses[order]
    return gaiamock.get_a0_mas(period = periods, m1 = central_mass, m2 = masses, parallax = parallax, f = 0)

def get_planet_a0_mas(m_star, planets, parallax, jacobi=False):
    '''
    Calculate the semimajor axis of the star's reflex orbit (in mas) due to each planet.
    m_star: stellar mass in Msun
    planets: list of planet dicts (see PLANET_KEYS; Omega is not needed)
    parallax: 1/distance, in mas
    jacobi: if False (default), each planet orbits the star as an independent two-body system, with
        a^3 = G(m_star + m_planet)P^2/(4 pi^2) and a0 = a*m_planet/(m_star + m_planet).
        If True, use Jacobi coordinates: each planet orbits the barycenter of the star and all planets with
        shorter periods, so that m_star is replaced by m_star + (masses of inner planets). This is more accurate
        for hierarchical systems with massive planets; the difference is of order m_planet/m_star.
    returns an array with one a0 (mas) per planet, in the same order as planets
    '''
    planets = validate_planets(planets, require_Omega=False)
    return _reflex_a0(m_star = m_star, planets = planets, parallax = parallax, jacobi = jacobi)

def _K_kms(m_star, planets, jacobi):
    '''RV semi-amplitudes (km/s) for an already-validated list of planets; see get_planet_K_kms().'''
    if len(planets) == 0:
        return np.zeros(0)
    AU = 1.4959787e+11 # same constant as gaiamock.get_a0_mas()
    a0_au = _reflex_a0(m_star = m_star, planets = planets, parallax = 1.0, jacobi = jacobi) # with parallax = 1 mas, a0 in mas = a0 in AU
    period = np.array([p['period'] for p in planets])
    ecc = np.array([p['ecc'] for p in planets])
    inc = np.array([p['inc'] for p in planets])
    return 2*np.pi*a0_au*AU*np.sin(inc)/(period*86400*np.sqrt(1 - ecc**2))/1e3

def get_planet_K_kms(m_star, planets, jacobi=False):
    '''
    Calculate the semi-amplitude of the star's radial velocity (in km/s) due to each planet. It is computed from
    the reflex orbit as K = 2 pi a0 sin(inc)/(P sqrt(1 - ecc^2)), so that the RV and astrometric amplitudes are
    consistent by construction (including when jacobi = True). This equals the usual
    K = (2 pi G/P)^(1/3) m_planet sin(inc)/((m_star + m_planet)^(2/3) sqrt(1 - ecc^2)).
    m_star: stellar mass in Msun
    planets: list of planet dicts (see PLANET_KEYS; Omega is not needed)
    jacobi: see get_planet_a0_mas()
    returns an array with one K (km/s) per planet, in the same order as planets
    '''
    planets = validate_planets(planets, require_Omega=False)
    return _K_kms(m_star = m_star, planets = planets, jacobi = jacobi)

def keplerian_rv_signal(t_rvs_yr, period, Tp, ecc, w, K, c_funcs):
    '''
    Radial velocity of the star due to one Keplerian orbit: K [cos(nu + w) + ecc cos(w)], with positive values
    meaning motion away from the observer.
    t_rvs_yr: times in years relative to the reference epoch (the same epoch used for the astrometry)
    period, Tp: in days; ecc: eccentricity; w: argument of periastron of the star's orbit, in radians
    K: semi-amplitude in km/s
    returns an array in km/s
    '''
    return gaiamock.predict_radial_velocities(t_rvs_day = np.asarray(t_rvs_yr, dtype=float)*365.25, period = period, Tp = Tp, ecc = ecc, w = w, K = K, gamma = 0, c_funcs = c_funcs)

def _gamma_per_epoch(gamma, rv_dataset, n_rvs):
    '''
    Expand the systemic velocities to one value per RV epoch.
    gamma: a single systemic velocity, or one per dataset (km/s)
    rv_dataset: None (all RVs belong to dataset 0), or an integer array with the dataset index of each RV
    '''
    gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
    if gamma.ndim != 1 or len(gamma) == 0 or not np.all(np.isfinite(gamma)):
        raise ValueError('gamma must be a finite number or a 1D array with one value per RV dataset.')
    if rv_dataset is None:
        if len(gamma) != 1:
            raise ValueError('gamma has %d values, so rv_dataset must give the dataset index (0 to %d) of each RV.' % (len(gamma), len(gamma) - 1))
        return np.full(n_rvs, gamma[0])
    rv_dataset = np.asarray(rv_dataset)
    if rv_dataset.shape != (n_rvs,):
        raise ValueError('rv_dataset must have one entry per RV epoch (%d), but has shape %s.' % (n_rvs, rv_dataset.shape))
    if n_rvs == 0:
        return np.zeros(0)
    if not np.issubdtype(rv_dataset.dtype, np.integer):
        raise ValueError('rv_dataset must contain integer dataset indices.')
    if rv_dataset.min() < 0 or rv_dataset.max() >= len(gamma):
        raise ValueError('rv_dataset values must be between 0 and %d (one gamma per dataset).' % (len(gamma) - 1))
    return gamma[rv_dataset]

def astrometry_planets_model(t_ast_yr, psi, plx_factor, ra_off, dec_off, parallax, pmra, pmdec, m_star, planets, c_funcs, jacobi=False, return_components=False):
    '''
    Predict the (noiseless) along-scan astrometry of a star hosting any number of planets, for modeling/fitting
    epoch astrometry. The star's reflex motion is the sum of one Keplerian orbit per planet.
    t_ast_yr: times in years relative to the reference epoch of the data release
    psi: scan angles in radians
    plx_factor: along-scan parallax factors
    ra_off, dec_off: offsets of the barycenter from the reference position at the reference epoch, in mas
        (ra_off includes the cos(dec) factor). These are free parameters when fitting real epoch astrometry.
    parallax (mas), pmra, pmdec (mas/yr): barycenter parallax and proper motion
    m_star: stellar mass in Msun
    planets: list of dicts, one per planet, with keys period, Tp, ecc, Omega, inc, w, m_planet (see PLANET_KEYS)
    c_funcs: from read_in_C_functions() (if None, it is read in)
    jacobi: if True, use Jacobi coordinates for the planet orbits (see get_planet_a0_mas())
    return_components: if True, also return an array of shape (N_planets, N_epochs) with each planet's signal
    returns the along-scan positions in mas (plus components if return_components is True)
    '''
    planets = validate_planets(planets)
    if c_funcs is None:
        c_funcs = gaiamock.read_in_C_functions()
    t_ast_yr, psi, plx_factor = (np.asarray(arr, dtype=float) for arr in (t_ast_yr, psi, plx_factor))

    a0_mas = _reflex_a0(m_star = m_star, planets = planets, parallax = parallax, jacobi = jacobi)
    components = np.zeros((len(planets), len(t_ast_yr)))
    for i, planet in enumerate(planets):
        A, B, F, G = get_thiele_innes(a0_mas = a0_mas[i], Omega = planet['Omega'], inc = planet['inc'], w = planet['w'])
        components[i] = keplerian_al_signal(t_ast_yr = t_ast_yr, psi = psi, period = planet['period'], Tp = planet['Tp'], ecc = planet['ecc'], A = A, B = B, F = F, G = G, c_funcs = c_funcs)

    Lambda_pred = (ra_off + pmra*t_ast_yr)*np.sin(psi) + (dec_off + pmdec*t_ast_yr)*np.cos(psi) + parallax*plx_factor # barycenter motion
    for component in components:
        Lambda_pred = Lambda_pred + component # reflex motion due to each planet

    if return_components:
        return Lambda_pred, components
    return Lambda_pred

def rv_planets_model(t_rvs_yr, m_star, planets, gamma, c_funcs, rv_dataset=None, jacobi=False, return_components=False):
    '''
    Predict the (noiseless) radial velocity of a star hosting any number of planets.
    t_rvs_yr: times in years relative to the reference epoch (the same epoch used for the astrometry,
        e.g. t_rvs_yr = (BJD - 2457936.875)/365.25 for DR4; see gaiamock.rescale_times_astrometry())
    m_star: stellar mass in Msun
    planets: list of planet dicts (see PLANET_KEYS; Omega does not affect RVs and may be omitted)
    gamma: systemic velocity in km/s: a single value, or one value per RV dataset (e.g. per instrument)
    c_funcs: from read_in_C_functions() (if None, it is read in)
    rv_dataset: integer array with the dataset index (0 to len(gamma) - 1) of each RV. Needed only if
        gamma has more than one value.
    jacobi: if True, use Jacobi coordinates for the planet orbits (see get_planet_a0_mas())
    return_components: if True, also return an array of shape (N_planets, N_rvs) with each planet's signal
    returns the RVs in km/s (plus components if return_components is True)
    '''
    planets = validate_planets(planets, require_Omega=False)
    if c_funcs is None:
        c_funcs = gaiamock.read_in_C_functions()
    t_rvs_yr = np.asarray(t_rvs_yr, dtype=float)

    rv_pred = _gamma_per_epoch(gamma = gamma, rv_dataset = rv_dataset, n_rvs = len(t_rvs_yr))
    K = _K_kms(m_star = m_star, planets = planets, jacobi = jacobi)
    components = np.zeros((len(planets), len(t_rvs_yr)))
    for i, planet in enumerate(planets):
        components[i] = keplerian_rv_signal(t_rvs_yr = t_rvs_yr, period = planet['period'], Tp = planet['Tp'], ecc = planet['ecc'], w = planet['w'], K = K[i], c_funcs = c_funcs)
    for component in components:
        rv_pred = rv_pred + component

    if return_components:
        return rv_pred, components
    return rv_pred

def astrometry_and_rv_planets_model(t_ast_yr, psi, plx_factor, t_rvs_yr, ra_off, dec_off, parallax, pmra, pmdec, m_star, planets, gamma, c_funcs, rv_dataset=None, jacobi=False):
    '''
    Predict the (noiseless) along-scan astrometry and radial velocities of a star hosting any number of planets.
    The astrometry and RVs can be sampled at different times, but t_ast_yr and t_rvs_yr must be relative to
    the same reference epoch. Arguments are as in astrometry_planets_model() and rv_planets_model(); the same
    w (argument of periastron of the star's orbit) and Omega are used for both.
    returns Lambda_pred (mas), rv_pred (km/s)
    '''
    if c_funcs is None:
        c_funcs = gaiamock.read_in_C_functions()
    Lambda_pred = astrometry_planets_model(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ra_off = ra_off, dec_off = dec_off, parallax = parallax, pmra = pmra, pmdec = pmdec, m_star = m_star, planets = planets, c_funcs = c_funcs, jacobi = jacobi)
    rv_pred = rv_planets_model(t_rvs_yr = t_rvs_yr, m_star = m_star, planets = planets, gamma = gamma, c_funcs = c_funcs, rv_dataset = rv_dataset, jacobi = jacobi)
    return Lambda_pred, rv_pred

def predict_astrometry_planets(ra, dec, parallax, pmra, pmdec, m_star, planets, phot_g_mean_mag, data_release, c_funcs, jacobi=False, reject_10_percent=True, rng=None, return_components=False):
    '''
    This function predicts the epoch-level astrometry of a star hosting any number of planets, as it would be
    observed by Gaia. The star's reflex motion is the sum of one Keplerian orbit per planet (planet-planet
    interactions are neglected), and the planets are assumed to contribute no light. The noiseless part is
    computed by astrometry_planets_model(), with the source at the reference position (ra_off = dec_off = 0).
    ra and dec (degrees): the coordinates of the source at the reference time (which is different for dr3/dr4/dr5)
    parallax (mas): the true parallax (i.e., 1/d)
    pmra, pmdec: true proper motions in mas/yr
    m_star: stellar mass in Msun
    planets: list of dicts, one per planet, with keys period, Tp, ecc, Omega, inc, w, m_planet (see PLANET_KEYS).
        An empty list gives a single star.
    phot_g_mean_mag: G-band magnitude of the star
    data_release: 'dr3', 'dr4', or 'dr5'
    c_funcs: from read_in_C_functions() (if None, it is read in)
    jacobi: if True, use Jacobi coordinates for the planet orbits (see get_planet_a0_mas())
    reject_10_percent: if True, randomly drop ~10% of FOV transits, as in gaiamock
    rng: optional numpy Generator for reproducibility; if None, the global np.random state is used, and the
        output is identical to gaiamock's predict_astrometry_* functions for the same np.random.seed()
    return_components: if True, also return an array of shape (N_planets, N_epochs) with the noiseless
        along-scan signal of each planet

    returns t_ast_yr, psi, plx_factor, ast_obs, ast_err (the same format as other gaiamock functions),
        plus components if return_components is True
    '''
    planets = validate_planets(planets)
    if c_funcs is None:
        c_funcs = gaiamock.read_in_C_functions()

    t_ast_yr, psi, plx_factor = get_scan_metadata(ra = ra, dec = dec, data_release = data_release, reject_10_percent = reject_10_percent, rng = rng)
    Lambda_pred, components = astrometry_planets_model(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ra_off = 0, dec_off = 0, parallax = parallax, pmra = pmra, pmdec = pmdec, m_star = m_star, planets = planets, c_funcs = c_funcs, jacobi = jacobi, return_components = True)
    ast_obs, ast_err = add_epoch_noise(ast_true = Lambda_pred, phot_g_mean_mag = phot_g_mean_mag, rng = rng)

    if return_components:
        return t_ast_yr, psi, plx_factor, ast_obs, ast_err, components
    return t_ast_yr, psi, plx_factor, ast_obs, ast_err


# ================================================================ fitting astrometry with N planets

MAX_PLANETS_FIT = 10 # must match GMP_MAX_PLANETS in kepler_solve_planets.c
_C_FUNCS_PLUS = None

def read_in_C_functions_plus():
    '''
    Read in the compiled functions from kepler_solve_planets.so (compile kepler_solve_planets.c first; see the top of that file).
    These are separate from gaiamock's read_in_C_functions().
    '''
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kepler_solve_planets.so')
    if not os.path.exists(path):
        raise ValueError('You need to compile kepler_solve_planets.c! (see the top of that file)')
    lib = ctypes.CDLL(path)
    dbl = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    intp = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    c_int, c_dbl = ctypes.c_int, ctypes.c_double
    data = [c_int, dbl, dbl, dbl, dbl, dbl]
    lib.gmp_astrometry_chi2.argtypes = data + [c_int, dbl, dbl, dbl, c_dbl, c_dbl, c_int, dbl, dbl]
    lib.gmp_astrometry_periodogram.argtypes = data + [c_int, dbl, dbl, dbl, c_int, dbl, c_dbl, c_dbl, c_int, dbl]
    lib.gmp_astrometry_scan.argtypes = data + [c_int, dbl, dbl, dbl, c_int, dbl, c_int, dbl, c_int, c_dbl, c_dbl, c_int, dbl]
    lib.gmp_astrometry_polish.argtypes = data + [c_int, dbl, dbl, dbl, intp, dbl, dbl, c_dbl, c_dbl, c_dbl, c_dbl, c_int, c_int, c_dbl, c_dbl, c_int, c_int, dbl]
    lib.gmp_astrometry_multistart.argtypes = data + [c_int, dbl, dbl, dbl, dbl, dbl, c_dbl, c_dbl, c_dbl, c_dbl, c_int, c_int, c_int, c_int, c_int, c_int, c_int, dbl, intp]
    lib.gmp_astrometry_residuals.argtypes = data + [c_int, dbl, dbl]
    for name in ('gmp_astrometry_chi2', 'gmp_astrometry_periodogram', 'gmp_astrometry_scan', 'gmp_astrometry_polish', 'gmp_astrometry_multistart', 'gmp_astrometry_residuals'):
        getattr(lib, name).restype = None
    return lib

def _get_c_funcs_plus(c_funcs_plus):
    '''Return the given compiled library, or read it in once per process.'''
    global _C_FUNCS_PLUS
    if c_funcs_plus is not None:
        return c_funcs_plus
    if _C_FUNCS_PLUS is None:
        _C_FUNCS_PLUS = read_in_C_functions_plus()
    return _C_FUNCS_PLUS

def _astrometry_data(t_ast_yr, psi, plx_factor, ast_obs, ast_err):
    '''Check epoch astrometry arrays and return them as contiguous float64 arrays (the C functions' input format).'''
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (t_ast_yr, psi, plx_factor, ast_obs, ast_err)]
    n = len(arrays[0])
    if any(a.ndim != 1 or len(a) != n for a in arrays):
        raise ValueError('t_ast_yr, psi, plx_factor, ast_obs and ast_err must be 1D arrays of the same length.')
    if not all(np.all(np.isfinite(a)) for a in arrays) or np.any(arrays[4] <= 0):
        raise ValueError('epoch astrometry must be finite, with ast_err > 0.')
    return arrays

def _prior_args(parallax_prior, eccentricity_prior):
    '''Convert the prior options to the C arguments (plx_mu, plx_sig, ecc_a, ecc_b); sig/a/b <= 0 means no prior.'''
    if parallax_prior is None:
        plx_mu, plx_sig = 0.0, -1.0
    else:
        plx_mu, plx_sig = (float(v) for v in parallax_prior)
        if not plx_sig > 0:
            raise ValueError('parallax_prior must be (mean, sigma) with sigma > 0, or None.')
    if eccentricity_prior is None:
        ecc_a, ecc_b = -1.0, -1.0
    else:
        ecc_a, ecc_b = (float(v) for v in eccentricity_prior)
        if not (ecc_a > 0 and ecc_b > 0):
            raise ValueError('eccentricity_prior must be (a, b) with a, b > 0 for a Beta(a, b) prior, or None.')
    return plx_mu, plx_sig, ecc_a, ecc_b

def _nonlinear_arrays(periods, phi_ps, eccs):
    arrays = [np.ascontiguousarray(np.atleast_1d(a), dtype=np.float64) for a in (periods, phi_ps, eccs)]
    if not (len(arrays[0]) == len(arrays[1]) == len(arrays[2])):
        raise ValueError('periods, phi_ps and eccs must have the same length (one per planet).')
    if len(arrays[0]) > MAX_PLANETS_FIT:
        raise ValueError('at most %d planets are supported.' % MAX_PLANETS_FIT)
    if np.any(arrays[0] <= 0) or np.any(arrays[2] < 0) or np.any(arrays[2] >= 1):
        raise ValueError('periods must be positive and 0 <= ecc < 1.')
    return arrays

def get_astrometric_chi2_planets(t_ast_yr, psi, plx_factor, ast_obs, ast_err, periods, phi_ps, eccs, parallax_prior=None, reject_outlier=False, c_funcs_plus=None):
    '''
    Solve the linear astrometric parameters for fixed nonlinear orbital parameters of any number of planets
    (the N-planet generalization of gaiamock.get_astrometric_chi2()).
    t_ast_yr, psi, plx_factor, ast_obs, ast_err: epoch astrometry (per-transit or per-CCD measurements)
    periods (days), phi_ps (= 2 pi Tp/period, radians), eccs: arrays with one value per planet (can be empty)
    parallax_prior: None (default) or (mean, sigma) in mas, for a Gaussian parallax prior
    reject_outlier: if True, the epoch with the largest residual is dropped (default False)
    c_funcs_plus: from read_in_C_functions_plus() (read in automatically if None)
    returns:
        chi2: data chi2, plus the parallax-prior term (mu_plx - mean)^2/sigma^2 if a prior is used
        mu: linear parameters (ra_off, pmra, dec_off, pmdec, parallax, then B, G, A, F for each planet)
    '''
    lib = _get_c_funcs_plus(c_funcs_plus)
    data = _astrometry_data(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    P, phi, e = _nonlinear_arrays(periods, phi_ps, eccs)
    plx_mu, plx_sig, _, _ = _prior_args(parallax_prior, None)
    out, mu = np.zeros(3), np.zeros(5 + 4*len(P))
    lib.gmp_astrometry_chi2(len(data[0]), *data, len(P), P, phi, e, plx_mu, plx_sig, int(bool(reject_outlier)), out, mu)
    return out[0] + out[1], mu

# Settings of the per-planet search, which is gaiamock's "periodogram_multistart" optimizer (see the ASTFIT_* settings
# in kepler_solve_astrometry.c): number of periods in the main and short-period periodogram grids, number of phases in
# the eccentricity scan, number of distinct periods scanned, number of starts polished, and whether the Nelder-Mead
# polish uses gaiamock's "full" (tighter) tolerances.
OPTIMIZER_SETTINGS = {
    'periodogram_multistart': dict(n_period=180, n_short_period=560, n_ecc_phase=36, max_periods_keep=20, max_starts=48, full_nm=0),
    'periodogram_multistart_full': dict(n_period=260, n_short_period=800, n_ecc_phase=48, max_periods_keep=24, max_starts=64, full_nm=1),
}

class _AstrometryProblem:
    '''Epoch astrometry plus fitting options, with thin wrappers around the compiled functions.'''
    def __init__(self, t_ast_yr, psi, plx_factor, ast_obs, ast_err, parallax_prior, eccentricity_prior, reject_outlier, P_min, P_max, e_max, c_funcs_plus):
        self.lib = _get_c_funcs_plus(c_funcs_plus)
        self.data = _astrometry_data(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
        self.n = len(self.data[0])
        self.plx_mu, self.plx_sig, self.ecc_a, self.ecc_b = _prior_args(parallax_prior, eccentricity_prior)
        self.reject = int(bool(reject_outlier))
        if not (0 < P_min < P_max) or not (0 < e_max < 1):
            raise ValueError('require 0 < P_min < P_max and 0 < e_max < 1.')
        # bounds on (P, phi_p, ecc), as in gaiamock (which also uses e >= 1e-6 with an eccentricity prior)
        self.L = np.array([float(P_min), 0., 1e-6 if eccentricity_prior is not None else 0.])
        self.U = np.array([float(P_max), 2*np.pi, float(e_max)])

    def chi2(self, P, phi, e):
        '''data chi2 + parallax-prior term, linear parameters, and index of the rejected epoch (-1 if none)'''
        P, phi, e = _nonlinear_arrays(P, phi, e)
        out, mu = np.zeros(3), np.zeros(5 + 4*len(P))
        self.lib.gmp_astrometry_chi2(self.n, *self.data, len(P), P, phi, e, self.plx_mu, self.plx_sig, self.reject, out, mu)
        return out[0] + out[1], mu, int(out[2])

    def multistart(self, P, phi, e, optimizer):
        '''
        gaiamock's multistart search for one more planet, with planets (P, phi, e) held fixed. Returns the polished
        starts as a list of (P, phi, e, objective) for the new planet, best first.
        '''
        P, phi, e = _nonlinear_arrays(P, phi, e)
        s = OPTIMIZER_SETTINGS[optimizer]
        out, n_out = np.zeros(4*s['max_starts']), np.zeros(1, dtype=np.int32)
        self.lib.gmp_astrometry_multistart(self.n, *self.data, len(P), P, phi, e, self.L, self.U, self.plx_mu, self.plx_sig, self.ecc_a, self.ecc_b,
                                           self.reject, s['n_period'], s['n_short_period'], s['n_ecc_phase'], s['max_periods_keep'],
                                           s['max_starts'], s['full_nm'], out, n_out)
        return [tuple(row) for row in out[:4*n_out[0]].reshape(-1, 4)]

    def polish(self, P, phi, e, free, max_iter=400, ftol=1e-6, xtol=1e-7, max_restarts=9, smooth=True):
        '''
        Nelder-Mead polish of the free planets (by default with the tolerances of gaiamock's "full" optimizer, restarted
        until the objective stops improving), in (log P, e cos(phi), e sin(phi)) if smooth (default; cannot get stuck
        at e = 0 or at the phi = 0/2 pi boundary) or in gaiamock's (log P, phi, e). Returns P, phi, e (polished copies),
        objective (chi2 + all priors), and chi2 (data + parallax-prior term).
        '''
        P, phi, e = (np.array(a, dtype=np.float64) for a in _nonlinear_arrays(P, phi, e))
        P, phi, e = np.clip(P, self.L[0], self.U[0]), np.clip(phi, self.L[1], self.U[1]), np.clip(e, self.L[2], self.U[2])
        out = np.zeros(4)
        self.lib.gmp_astrometry_polish(self.n, *self.data, len(P), P, phi, e, np.ascontiguousarray(free, dtype=np.int32), self.L, self.U,
                                       self.plx_mu, self.plx_sig, self.ecc_a, self.ecc_b, self.reject, max_iter, ftol, xtol, max_restarts, int(bool(smooth)), out)
        return P, phi, e, out[0], out[1] + out[2]

    def residuals(self, theta, n_pl):
        out = np.zeros(self.n)
        self.lib.gmp_astrometry_residuals(self.n, *self.data, n_pl, np.ascontiguousarray(theta, dtype=np.float64), out)
        return out

def _same_periods(P1, P2, tol=0.005):
    '''True if two solutions have the same (sorted) periods to within a fractional tolerance.'''
    return len(P1) == len(P2) and np.all(np.abs(np.log(np.sort(P1)/np.sort(P2))) < tol)

def _dedupe(solutions, n_max):
    '''Sort solutions (P, phi, e, objective, chi2) by objective and keep the best n_max with distinct sets of periods.'''
    kept = []
    for sol in sorted(solutions, key=lambda s: s[3]):
        if not any(_same_periods(sol[0], k[0]) for k in kept):
            kept.append(sol)
            if len(kept) == n_max:
                break
    return kept

def _search_one_planet(prob, P, phi, e, settings):
    '''
    Add one planet to the planets (P, phi, e): run gaiamock's multistart search for the new planet with the existing
    planets held fixed, then (if there are existing planets) polish all planets jointly, starting from the best
    settings['n_joint'] polished starts (which can share a period but differ in eccentricity or phase) and the best
    start at each of the settings['n_joint'] best distinct periods. For a single planet the best solution is exactly
    that of gaiamock's optimizer. Returns a list of solutions (P, phi, e, objective, chi2) with distinct periods, best first.
    '''
    starts = prob.multistart(P, phi, e, settings['optimizer']) # distinct in (P, phi, e) and sorted, as in gaiamock
    sols = [(np.append(P, Pn), np.append(phi, phin), np.append(e, en), obj, None) for Pn, phin, en, obj in starts]
    best_per_period = _dedupe(sols, settings['n_joint'])
    sols = sols[:settings['n_joint']] + [s for s in best_per_period if not any(s is t for t in sols[:settings['n_joint']])]
    out = []
    for sol in sols:
        if len(P) > 0:
            sol = prob.polish(sol[0], sol[1], sol[2], np.ones(len(P) + 1, dtype=np.int32))
        else:
            sol = sol[:4] + (prob.chi2(*sol[:3])[0],)
        out.append(sol)
    return _dedupe(out, settings['n_joint'])

def _refine_planets(prob, sol, settings):
    '''
    One round of "backfitting": re-search each planet with the other planets held fixed (followed by a joint polish),
    keeping any solution that improves the objective. This helps when an early planet was fit while a later one was
    still unmodeled, which biases it.
    '''
    best = sol
    for i in range(len(sol[0])):
        others = [k for k in range(len(sol[0])) if k != i]
        res = _search_one_planet(prob, best[0][others], best[1][others], best[2][others], settings)[0]
        if res[3] < best[3] - 1e-6:
            best = res
    return best

def _derived_planet_quantities(theta, cov, i, m_star):
    '''
    Tp, a0, inc, Omega, w (and m_planet if m_star is given) of planet i, and their uncertainties propagated from the
    covariance matrix of theta (ra_off, dec_off, parallax, pmra, pmdec, then P, e, phi_p, A, B, F, G per planet) with a
    central-difference Jacobian. Differences of Omega (defined modulo pi) and w (modulo 2 pi) are wrapped, so orbits
    near those wrap points get correct uncertainties.
    '''
    sig = np.sqrt(np.abs(np.diag(cov)))
    def derived(th):
        Pi, ei, phii, A, B, F, G = th[5 + 7*i: 12 + 7*i]
        a0, Omega, w, inc = gaiamock.get_Campbell_elements(A, B, F, G)
        out = [phii*Pi/(2*np.pi), a0, inc, Omega, w]
        if m_star is not None:
            out.append(get_dark_companion_mass(a0_mas=a0, period=Pi, parallax=th[2], m1=m_star))
        return np.array(out)
    with np.errstate(invalid='ignore', divide='ignore'): # e.g. no real mass for a negative parallax
        g0 = derived(theta)
        Jg = np.zeros((len(g0), len(theta)))
        for j in range(len(theta)):
            h = 1e-4*sig[j] if (np.isfinite(sig[j]) and sig[j] > 0) else 1e-8
            up, down = theta.copy(), theta.copy()
            up[j] += h; down[j] -= h
            dg = derived(up) - derived(down)
            # Omega is returned in [0, pi): when it wraps, get_Campbell_elements shifts Omega by pi and w by pi together
            # (the same orbit), so remove the same multiple of pi from both before wrapping w modulo 2 pi
            k_pi = np.round(dg[3]/np.pi)
            dg[3] -= k_pi*np.pi
            dg[4] = np.angle(np.exp(1j*(dg[4] - k_pi*np.pi)))
            Jg[:, j] = dg/(2*h)
    return g0, np.sqrt(np.abs(np.diag(Jg @ cov @ Jg.T)))

def _astrometry_fit_summary(prob, P, phi, e, m_star, binned):
    '''Linear parameters, uncertainties (from the Jacobian, inflated by the goodness of fit as in gaiamock), and derived quantities.'''
    order = np.argsort(P)
    P, phi, e = P[order], phi[order], e[order]
    K = len(P)
    chi2_total, mu, worst = prob.chi2(P, phi, e)
    theta = [mu[0], mu[2], mu[4], mu[1], mu[3]]
    names = ['ra_off', 'dec_off', 'parallax', 'pmra', 'pmdec']
    for i in range(K):
        B, G, A, F = mu[5 + 4*i: 9 + 4*i]
        theta += [P[i], e[i], phi[i], A, B, F, G]
        names += ['%s_%d' % (k, i) for k in ('period', 'ecc', 'phi_p', 'A', 'B', 'F', 'G')]
    theta = np.array(theta)
    keep = np.ones(prob.n, dtype=bool)
    if worst >= 0:
        keep[worst] = False

    def resid(th):
        return prob.residuals(th, K)[keep]

    # Jacobian of the normalized residuals (central differences, step 1e-8 as in gaiamock)
    J = np.zeros((keep.sum(), len(theta)))
    for j in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[j] += 1e-8; down[j] -= 1e-8
        J[:, j] = (resid(up) - resid(down))/2e-8
    JTJ = J.T @ J
    if prob.plx_sig > 0:
        JTJ[2, 2] += 1/prob.plx_sig**2
    if prob.ecc_a > 0:
        for i in range(K):
            ei = np.clip(e[i], 1e-6, 1 - 1e-6)
            H_e = (prob.ecc_a - 1)/ei**2 + (prob.ecc_b - 1)/(1 - ei)**2
            if H_e > 0:
                JTJ[5 + 7*i + 1, 5 + 7*i + 1] += H_e
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.full_like(JTJ, np.nan)

    chi2_data = np.sum(resid(theta)**2)
    n_used, n_par = int(keep.sum()), len(theta)
    nu = n_used - n_par
    if binned:
        chi2_red = chi2_data/nu
        chi2_red_unbinned = gaiamock.predict_reduced_chi2_unbinned_data(chi2_red_binned=chi2_red, n_param=n_par, N_points=n_used, Nbin=8)
        nu_unbinned = 8*n_used - n_par
        cc = np.sqrt(chi2_red_unbinned/(1 - 2/(9*nu_unbinned))**3)
        F2 = gaiamock.predict_F2_unbinned_data(chi2_red_binned=chi2_red, n_param=n_par, N_points=n_used, Nbin=8)
        uwe = np.sqrt(chi2_red_unbinned)
    else:
        chi2_red = chi2_data/nu
        cc = np.sqrt(chi2_red/(1 - 2/(9*nu))**3)
        F2 = np.sqrt(9*nu/2)*(chi2_red**(1/3) + 2/(9*nu) - 1)
        uwe = np.sqrt(chi2_red)
    cov = cc**2*cov
    sig = np.sqrt(np.abs(np.diag(cov)))

    result = dict(n_planets=K, chi2=chi2_total, chi2_data=chi2_data, n_obs=prob.n, n_used=n_used, n_params=n_par, nu=nu, F2=F2, uwe=uwe,
                  uncertainty_scale=cc, outlier_index=worst, theta=theta, theta_names=names, cov=cov)
    for j, name in enumerate(names[:5]):
        result[name], result['sig_' + name] = theta[j], sig[j]

    planets, model_planets = [], []
    for i in range(K):
        g0, sig_g = _derived_planet_quantities(theta, cov, i, m_star)
        planet = {}
        for j, key in enumerate(('period', 'ecc', 'phi_p', 'A', 'B', 'F', 'G')):
            planet[key], planet['sig_' + key] = theta[5 + 7*i + j], sig[5 + 7*i + j]
        for j, key in enumerate(('Tp', 'a0_mas', 'inc', 'Omega', 'w') + (('m_planet',) if m_star is not None else ())):
            planet[key], planet['sig_' + key] = g0[j], sig_g[j]
        planet['inc_deg'], planet['sig_inc_deg'] = np.degrees(planet['inc']), np.degrees(planet['sig_inc'])
        planets.append(planet)
        if m_star is not None:
            model_planets.append({k: planet[k] for k in PLANET_KEYS})
    result['planets'] = planets
    result['model_planets'] = model_planets if m_star is not None else None
    return result

DEFAULT_DELTA_CHI2_MIN = 50.0

def fit_astrometry_planets(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_planets=None, max_planets=3, delta_chi2_min=None, m_star=None,
                           P_min=10., P_max=1e4, e_max=0.99, parallax_prior=None, eccentricity_prior=None, reject_outlier=False, binned=True,
                           initial_planets=None, optimizer='periodogram_multistart', n_joint=3, refine=True, n_beam=1, verbose=False, c_funcs_plus=None):
    '''
    Fit epoch astrometry with a model of a star hosting several planets (dark companions), adding one planet at a time.

    Each planet is found with gaiamock's own search (its default "periodogram_multistart" optimizer, ported to
    kepler_solve_planets.c), generalized so that the planets already found are included in every chi2 evaluation:
    a circular-orbit periodogram, an eccentricity/phase scan of the best periods, and Nelder-Mead polish of the best
    starts. For one planet this is exactly gaiamock's fit_orbital_solution_nonlinear(). With more than one planet,
    the best few results are then polished jointly with all the other planets. The linear parameters (ra_off, dec_off,
    parallax, proper motion, and Thiele-Innes elements) are always solved exactly.

    t_ast_yr, psi, plx_factor, ast_obs, ast_err: epoch astrometry (as returned by the predict_astrometry_* functions).
        Per-transit (default, as in gaiamock) or per-CCD measurements can be used: the best fit and delta chi2 values
        are the same either way; set binned=False for per-CCD data so that F2, uwe and the uncertainty inflation are
        computed correctly.
    n_planets: number of planets to fit (known-N mode). If None (default), search blindly: keep adding planets while
        the chi2 improves by more than delta_chi2_min, up to max_planets.
    max_planets: largest number of planets considered in blind-search mode (default 3); fewer are tried if there are
        not enough measurements (a model with N planets has 5 + 7N parameters, and must leave at least 1 degree of freedom).
    delta_chi2_min: detection threshold on the chi2 improvement from adding a planet, used in blind-search mode
        (default DEFAULT_DELTA_CHI2_MIN).
    m_star: stellar mass in Msun. If given, planet masses (and their uncertainties) are computed, and model_planets
        are returned in the format of astrometry_planets_model(). Default None.
    P_min, P_max: period search range in days (default 10 to 1e4, as in gaiamock).
    e_max: largest eccentricity allowed (default 0.99, as in gaiamock).
    parallax_prior: None (default) or (mean, sigma) in mas: Gaussian parallax prior, included in the chi2.
    eccentricity_prior: None (default) or (a, b): Beta(a, b) prior on each eccentricity, included in the objective
        that is minimized (but not in the chi2 used for detection).
    reject_outlier: if True, the single epoch with the largest residual is ignored in every likelihood evaluation,
        as in gaiamock's reject_outlier option (default False).
    binned: True (default) if each measurement is the average of 8 CCD measurements (one FOV transit), as in gaiamock.
    initial_planets: optional list of starting guesses, each a dict with keys 'period', 'Tp' (days, relative to the
        reference epoch) and 'ecc'. If given, no search is done: the planets are polished jointly from these guesses.
    optimizer: 'periodogram_multistart' (default; gaiamock's default settings) or 'periodogram_multistart_full'
        (gaiamock's denser grids, more starts, and tighter polish), used for each planet.
    n_joint: with more than one planet, the number of the best (distinct) results of each planet's search that are
        polished jointly with all planets (default 3).
    refine: if True (default), after each new planet is added, re-search each planet with the others held fixed
        (helps when an early planet was biased by a not-yet-modeled one).
    n_beam: number of solutions carried forward to the next planet's search (default 1). Larger values make the search
        more robust to near-ties between a true and a spurious solution, at proportionally higher cost.
    c_funcs_plus: from read_in_C_functions_plus() (read in automatically if None).

    returns a dict with:
        n_planets, planets (list of dicts, sorted by period, each with period, ecc, phi_p, Tp, A, B, F, G, a0_mas,
            inc (radians), inc_deg, Omega, w, [m_planet], their uncertainties sig_*, and delta_chi2: the increase in
            chi2 when that planet is removed and the other planets are re-polished),
        ra_off, dec_off, parallax, pmra, pmdec and sig_* for each,
        chi2 (data + parallax-prior term), chi2_data, chi2_history (chi2 for 0, 1, 2, ... planets), n_obs, n_used,
        n_params, nu, F2, uwe, uncertainty_scale, outlier_index, theta, theta_names, cov, model_planets,
        and, in blind-search mode, rejected_candidate: the periods of the best model with one more planet, and its
        chi2 improvement, if that model did not pass the threshold (None otherwise), and max_planets_searched (which
        can be smaller than max_planets if there are few measurements).
    Note: astrometry alone cannot distinguish (Omega, w) from (Omega + pi, w + pi); Omega is returned in [0, pi).
    '''
    prob = _AstrometryProblem(t_ast_yr, psi, plx_factor, ast_obs, ast_err, parallax_prior, eccentricity_prior, reject_outlier, P_min, P_max, e_max, c_funcs_plus)
    if delta_chi2_min is None:
        delta_chi2_min = DEFAULT_DELTA_CHI2_MIN
    if optimizer not in OPTIMIZER_SETTINGS:
        raise ValueError('optimizer must be one of %s.' % list(OPTIMIZER_SETTINGS))
    settings = dict(optimizer=optimizer, n_joint=int(n_joint))
    blind = n_planets is None and initial_planets is None
    n_target = max_planets if blind else (len(initial_planets) if initial_planets is not None else int(n_planets))
    if n_target < 0 or n_target > MAX_PLANETS_FIT:
        raise ValueError('the number of planets must be between 0 and %d.' % MAX_PLANETS_FIT)
    n_usable = prob.n - prob.reject # a model with N planets has 5 + 7N free parameters
    if not blind and n_usable <= 5 + 7*n_target:
        raise ValueError('%d planets (%d parameters) cannot be fit to %d usable measurements.' % (n_target, 5 + 7*n_target, n_usable))
    if blind:
        n_target = min(n_target, max(0, (n_usable - 6)//7))

    P, phi, e = np.zeros(0), np.zeros(0), np.zeros(0)
    chi2_history = [prob.chi2(P, phi, e)[0]]
    objective = chi2_history[0]
    rejected = None
    if initial_planets is not None:
        if len(initial_planets) > 0:
            P0 = np.array([float(p['period']) for p in initial_planets])
            P, phi, e, objective, chi2 = prob.polish(P0, 2*np.pi*np.mod(np.array([float(p['Tp']) for p in initial_planets])/P0, 1),
                                                     np.array([float(p['ecc']) for p in initial_planets]), np.ones(len(P0), dtype=np.int32))
            chi2_history.append(chi2)
    else:
        beam = [(P, phi, e, objective, chi2_history[0])]
        while len(P) < n_target:
            children = [child for parent in beam for child in _search_one_planet(prob, *parent[:3], settings)]
            if refine and len(P) >= 1: # refine every child: a poor-looking child can refine into the best solution
                children = [_refine_planets(prob, child, settings) for child in _dedupe(children, len(children))]
            children = _dedupe(children, n_beam)
            best = children[0]
            gain = chi2_history[-1] - best[4]
            if verbose:
                print('%d planets: P = %s d, delta chi2 = %.1f' % (len(P) + 1, np.array2string(np.sort(best[0]), precision=1), gain))
            if blind and gain < delta_chi2_min:
                rejected = dict(periods=np.sort(best[0]), delta_chi2=gain)
                break
            beam = children
            P, phi, e, objective = best[:4]
            chi2_history.append(best[4])

    result = _astrometry_fit_summary(prob, P, phi, e, m_star, binned)
    # significance of each planet: chi2 increase when it is removed and the other planets are re-polished
    chi2_full = prob.chi2(P, phi, e)[0]
    for k, i in enumerate(np.argsort(P)):
        others = [j for j in range(len(P)) if j != i]
        if others:
            chi2_without = prob.polish(P[others], phi[others], e[others], np.ones(len(others), dtype=np.int32))[4]
        else:
            chi2_without = chi2_history[0]
        result['planets'][k]['delta_chi2'] = chi2_without - chi2_full
    result['chi2_history'] = chi2_history
    result['objective'] = objective
    if blind:
        result['rejected_candidate'] = rejected
        result['max_planets_searched'] = n_target
    return result
