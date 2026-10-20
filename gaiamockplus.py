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

MAX_RV_DATASETS = 20 # must match GMP_MAX_SETS in kepler_solve_planets.c

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
    # every function takes an astrometric dataset (n_ast, t, psi, plx_factor, obs, err, plx_mu, plx_sig, reject) and an
    # RV dataset (n_rv, t_rv, rv, rv_err, rv_set, n_sets); either can be empty (n = 0)
    data = [c_int, dbl, dbl, dbl, dbl, dbl, c_dbl, c_dbl, c_int, c_int, dbl, dbl, dbl, intp, c_int]
    lib.gmp_chi2.argtypes = data + [c_int, dbl, dbl, dbl, dbl, dbl, dbl]
    lib.gmp_periodogram.argtypes = data + [c_int, dbl, dbl, dbl, c_int, dbl, dbl]
    lib.gmp_scan.argtypes = data + [c_int, dbl, dbl, dbl, c_int, dbl, c_int, dbl, c_int, dbl]
    lib.gmp_polish.argtypes = data + [c_int, dbl, dbl, dbl, intp, dbl, dbl, c_dbl, c_dbl, c_int, c_dbl, c_dbl, c_int, c_int, dbl]
    lib.gmp_multistart.argtypes = data + [c_int, dbl, dbl, dbl, dbl, dbl, c_dbl, c_dbl, c_int, c_int, c_int, c_int, c_int, c_int, c_int, dbl, intp]
    lib.gmp_anneal.argtypes = data + [c_int, dbl, dbl, dbl, dbl, dbl, c_dbl, c_dbl, ctypes.c_uint, dbl]
    lib.gmp_astrometry_residuals.argtypes = [c_int, dbl, dbl, dbl, dbl, dbl, c_int, dbl, dbl]
    lib.gmp_rv_residuals.argtypes = [c_int, dbl, dbl, dbl, intp, c_int, c_int, dbl, dbl]
    for name in ('gmp_chi2', 'gmp_periodogram', 'gmp_scan', 'gmp_polish', 'gmp_multistart', 'gmp_anneal', 'gmp_astrometry_residuals', 'gmp_rv_residuals'):
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

def _rv_data(t_rvs_yr, rv, rv_err, rv_dataset):
    '''
    Check RV arrays and return t, rv, rv_err (contiguous float64), the dataset index of each RV (int32) and the number
    of datasets. rv_dataset: None (one dataset) or integer indices 0, 1, ..., n_sets - 1 (each must be used).
    '''
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (t_rvs_yr, rv, rv_err)]
    n = len(arrays[0])
    if any(a.ndim != 1 or len(a) != n for a in arrays) or n == 0:
        raise ValueError('t_rvs_yr, rv and rv_err must be non-empty 1D arrays of the same length.')
    if not all(np.all(np.isfinite(a)) for a in arrays) or np.any(arrays[2] <= 0):
        raise ValueError('RVs must be finite, with rv_err > 0.')
    if rv_dataset is None:
        sets = np.zeros(n, dtype=np.int32)
    else:
        sets = np.asarray(rv_dataset)
        if sets.shape != (n,) or not np.issubdtype(sets.dtype, np.integer):
            raise ValueError('rv_dataset must be an integer array with one entry per RV.')
        sets = np.ascontiguousarray(sets, dtype=np.int32)
    n_sets = int(sets.max()) + 1
    if sets.min() < 0 or n_sets > MAX_RV_DATASETS or len(np.unique(sets)) != n_sets:
        raise ValueError('rv_dataset values must be 0, 1, ..., n_sets - 1 (each used at least once; at most %d datasets).' % MAX_RV_DATASETS)
    return arrays[0], arrays[1], arrays[2], sets, n_sets

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

# Settings of the per-planet search, i.e. gaiamock's optimizers (see the ASTFIT_* settings in kepler_solve_astrometry.c).
# For the multistart optimizers: number of periods in the main and short-period period grids, number of phases in the
# eccentricity scan, number of distinct periods scanned, number of starts polished, whether the Nelder-Mead polish uses
# gaiamock's "full" (tighter) tolerances, and the number of circular-orbit phases scored at each trial period (0: the
# linear circular-orbit periodogram of "periodogram_multistart"; 8: the circular grid of gaiamock's "multistart").
# "annealing" is gaiamock's adaptive simulated annealing.
OPTIMIZER_SETTINGS = {
    'periodogram_multistart': dict(n_period=180, n_short_period=560, n_ecc_phase=36, max_periods_keep=20, max_starts=48, full_nm=0, n_circ_phase=0),
    'periodogram_multistart_full': dict(n_period=260, n_short_period=800, n_ecc_phase=48, max_periods_keep=24, max_starts=64, full_nm=1, n_circ_phase=0),
    'multistart': dict(n_period=180, n_short_period=560, n_ecc_phase=36, max_periods_keep=20, max_starts=48, full_nm=0, n_circ_phase=8),
    'annealing': dict(annealing=True),
}

_EMPTY_DBL = np.zeros(1)
_EMPTY_INT = np.zeros(1, dtype=np.int32)

class _Problem:
    '''
    A fitting problem: epoch astrometry and/or RVs, plus fitting options, with thin wrappers around the compiled
    functions. For fixed (P, phi_p, e) of each planet, the linear parameters of each dataset are solved exactly, and the
    objective is the sum of the chi2 of the datasets (plus prior terms). RV jitter (one value per RV dataset) is added
    in quadrature to rv_err; it is held fixed in each compiled call and set with set_jitter().
    '''
    def __init__(self, ast=None, rv=None, parallax_prior=None, eccentricity_prior=None, reject_outlier=False, P_min=10., P_max=1e4, e_max=0.99, c_funcs_plus=None):
        self.lib = _get_c_funcs_plus(c_funcs_plus)
        if ast is None and rv is None:
            raise ValueError('no data.')
        self.data = _astrometry_data(*ast) if ast is not None else None
        self.n = len(self.data[0]) if ast is not None else 0
        if rv is not None:
            self.t_rv, self.rv, self.rv_err, self.rv_set, self.n_sets = _rv_data(*rv)
            self.n_rv = len(self.rv)
        else:
            self.t_rv = self.rv = self.rv_err = _EMPTY_DBL
            self.rv_set, self.n_sets, self.n_rv = _EMPTY_INT, 0, 0
        self.set_jitter(np.zeros(self.n_sets))
        self.plx_mu, self.plx_sig, self.ecc_a, self.ecc_b = _prior_args(parallax_prior, eccentricity_prior)
        self.reject = int(bool(reject_outlier))
        if self.reject and ast is None:
            raise ValueError('reject_outlier applies to astrometry only.')
        if not (0 < P_min < P_max) or not (0 < e_max < 1):
            raise ValueError('require 0 < P_min < P_max and 0 < e_max < 1.')
        # bounds on (P, phi_p, ecc), as in gaiamock (which also uses e >= 1e-6 with an eccentricity prior)
        self.L = np.array([float(P_min), 0., 1e-6 if eccentricity_prior is not None else 0.])
        self.U = np.array([float(P_max), 2*np.pi, float(e_max)])

    def set_jitter(self, jitter):
        '''Set the RV jitter (km/s, one value per RV dataset), added in quadrature to rv_err.'''
        self.jitter = np.array(jitter, dtype=np.float64)
        if self.n_rv:
            self.rv_err_eff = np.ascontiguousarray(np.sqrt(self.rv_err**2 + self.jitter[self.rv_set]**2))
        else:
            self.rv_err_eff = _EMPTY_DBL

    def _args(self):
        ast = self.data if self.data is not None else [_EMPTY_DBL]*5
        return (self.n, *ast, self.plx_mu, self.plx_sig, self.reject, self.n_rv, self.t_rv, self.rv, self.rv_err_eff, self.rv_set, self.n_sets)

    def chi2(self, P, phi, e):
        '''
        chi2 of all datasets + parallax-prior term, the astrometric linear parameters, the index of the rejected
        astrometric epoch (-1 if none), and the RV linear parameters.
        '''
        P, phi, e = _nonlinear_arrays(P, phi, e)
        out, mu_ast, mu_rv = np.zeros(4), np.zeros(5 + 4*len(P)), np.zeros(self.n_sets + 2*len(P))
        self.lib.gmp_chi2(*self._args(), len(P), P, phi, e, out, mu_ast, mu_rv)
        return out[0] + out[1] + out[2], mu_ast, int(out[3]), mu_rv

    def minus_2_ln_like(self, chi2):
        '''-2 ln(likelihood) up to a constant: chi2 + sum of ln(2 pi sigma^2) of the RVs, which depends on the jitter.'''
        return chi2 + (np.sum(np.log(2*np.pi*self.rv_err_eff**2)) if self.n_rv else 0.0)

    def search(self, P, phi, e, optimizer, seed=0):
        '''
        gaiamock's search for one more planet (with the given optimizer), with planets (P, phi, e) held fixed. Returns
        the results as a list of (P, phi, e, objective) for the new planet, best first: all polished starts of the
        multistart optimizers, or the single result of annealing (which uses the C rand(), seeded with srand(seed) if
        seed > 0, and otherwise continuing the sequence, as gaiamock does).
        '''
        P, phi, e = _nonlinear_arrays(P, phi, e)
        s = OPTIMIZER_SETTINGS[optimizer]
        if s.get('annealing'):
            out = np.zeros(4)
            self.lib.gmp_anneal(*self._args(), len(P), P, phi, e, self.L, self.U, self.ecc_a, self.ecc_b, int(seed), out)
            return [tuple(out)] if np.all(np.isfinite(out)) else []
        out, n_out = np.zeros(4*s['max_starts']), np.zeros(1, dtype=np.int32)
        self.lib.gmp_multistart(*self._args(), len(P), P, phi, e, self.L, self.U, self.ecc_a, self.ecc_b, s['n_period'], s['n_short_period'],
                                s['n_ecc_phase'], s['max_periods_keep'], s['max_starts'], s['full_nm'], s['n_circ_phase'], out, n_out)
        return [tuple(row) for row in out[:4*n_out[0]].reshape(-1, 4)]

    def polish(self, P, phi, e, free, max_iter=400, ftol=1e-6, xtol=1e-7, max_restarts=9, smooth=True):
        '''
        Nelder-Mead polish of the free planets (by default with the tolerances of gaiamock's "full" optimizer, restarted
        until the objective stops improving), in (log P, e cos(phi), e sin(phi)) if smooth (default; cannot get stuck
        at e = 0 or at the phi = 0/2 pi boundary) or in gaiamock's (log P, phi, e). Returns P, phi, e (polished copies),
        objective (chi2 + all priors), and chi2 (all datasets + parallax-prior term).
        '''
        P, phi, e = (np.array(a, dtype=np.float64) for a in _nonlinear_arrays(P, phi, e))
        P, phi, e = np.clip(P, self.L[0], self.U[0]), np.clip(phi, self.L[1], self.U[1]), np.clip(e, self.L[2], self.U[2])
        out = np.zeros(5)
        self.lib.gmp_polish(*self._args(), len(P), P, phi, e, np.ascontiguousarray(free, dtype=np.int32), self.L, self.U,
                            self.ecc_a, self.ecc_b, max_iter, ftol, xtol, max_restarts, int(bool(smooth)), out)
        return P, phi, e, out[0], out[1] + out[2] + out[4]

    def residuals(self, theta, n_pl):
        '''normalized astrometric residuals for theta = (ra_off, dec_off, parallax, pmra, pmdec, then P, e, phi_p, A, B, F, G per planet)'''
        out = np.zeros(self.n)
        self.lib.gmp_astrometry_residuals(self.n, *self.data, n_pl, np.ascontiguousarray(theta, dtype=np.float64), out)
        return out

    def rv_residuals(self, theta, n_pl):
        '''normalized RV residuals (with jitter) for theta = (gamma per dataset, then P, e, phi_p, K cos(w), K sin(w) per planet)'''
        out = np.zeros(self.n_rv)
        self.lib.gmp_rv_residuals(self.n_rv, self.t_rv, self.rv, self.rv_err_eff, self.rv_set, self.n_sets, n_pl, np.ascontiguousarray(theta, dtype=np.float64), out)
        return out

def _AstrometryProblem(t_ast_yr, psi, plx_factor, ast_obs, ast_err, parallax_prior, eccentricity_prior, reject_outlier, P_min, P_max, e_max, c_funcs_plus):
    '''An astrometry-only _Problem.'''
    return _Problem(ast=(t_ast_yr, psi, plx_factor, ast_obs, ast_err), parallax_prior=parallax_prior, eccentricity_prior=eccentricity_prior,
                    reject_outlier=reject_outlier, P_min=P_min, P_max=P_max, e_max=e_max, c_funcs_plus=c_funcs_plus)

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
    prob = _AstrometryProblem(t_ast_yr, psi, plx_factor, ast_obs, ast_err, parallax_prior, None, reject_outlier, 1., 2., 0.5, c_funcs_plus)
    chi2, mu, _, _ = prob.chi2(periods, phi_ps, eccs)
    return chi2, mu

def get_rv_chi2_planets(t_rvs_yr, rv, rv_err, periods, phi_ps, eccs, rv_dataset=None, jitter=None, c_funcs_plus=None):
    '''
    Solve the linear RV parameters for fixed nonlinear orbital parameters of any number of planets (the RV counterpart
    of get_astrometric_chi2_planets()).
    t_rvs_yr: RV times in years relative to the reference epoch; rv, rv_err: RVs and uncertainties in km/s
    periods (days), phi_ps (= 2 pi Tp/period, radians), eccs: arrays with one value per planet (can be empty)
    rv_dataset: None (one dataset) or the dataset (instrument) index 0, 1, ... of each RV
    jitter: None (no jitter) or one jitter per dataset in km/s, added in quadrature to rv_err
    c_funcs_plus: from read_in_C_functions_plus() (read in automatically if None)
    returns:
        chi2: chi2 of the RVs (with the jitter)
        mu: linear parameters (one systemic velocity per dataset, then K cos(w), K sin(w) for each planet)
    '''
    prob = _Problem(rv=(t_rvs_yr, rv, rv_err, rv_dataset), P_min=1., P_max=2., e_max=0.5, c_funcs_plus=c_funcs_plus)
    if jitter is not None:
        prob.set_jitter(np.broadcast_to(np.asarray(jitter, dtype=float), (prob.n_sets,)))
    chi2, _, _, mu = prob.chi2(periods, phi_ps, eccs)
    return chi2, mu

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
    Add one planet to the planets (P, phi, e): run gaiamock's search (settings['optimizer']) for the new planet with
    the existing planets held fixed, then (if there are existing planets) polish all planets jointly, starting from the best
    settings['n_joint'] polished starts (which can share a period but differ in eccentricity or phase) and the best
    start at each of the settings['n_joint'] best distinct periods. For a single planet the best solution is exactly
    that of gaiamock's optimizer. Returns a list of solutions (P, phi, e, objective, chi2) with distinct periods, best first.
    '''
    starts = prob.search(P, phi, e, settings['optimizer']) # distinct in (P, phi, e) and sorted, as in gaiamock
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

def _settle_none(prob, sol):
    '''No extra parameters to adjust (astrometry): the detection statistic is the chi2.'''
    return sol, sol[4]

def _rv_weights_changed(prob, jitter_a, jitter_b, tol=0.05):
    '''whether two jitters give RV errors (rv_err and jitter in quadrature) that differ by more than ~tol (relative)'''
    if prob.n_rv == 0:
        return False
    sig_a = np.sqrt(prob.rv_err**2 + np.asarray(jitter_a)[prob.rv_set]**2)
    sig_b = np.sqrt(prob.rv_err**2 + np.asarray(jitter_b)[prob.rv_set]**2)
    return np.max(np.abs(np.log(sig_a/sig_b))) > tol

def _fit_planets(prob, n_planets, max_planets, delta_chi2_min, initial_planets, optimizer, n_joint, refine, n_beam, verbose,
                 n_base_params, n_params_per_planet, settle=_settle_none):
    '''
    The search shared by all fitting functions: add one planet at a time with gaiamock's search (_search_one_planet),
    optionally refining and keeping a beam of solutions, until n_planets are fit (known-N mode) or the detection
    statistic stops improving by more than delta_chi2_min (blind mode, up to max_planets); or polish initial_planets.
    settle(prob, sol) adjusts any parameters held fixed during the search (e.g. RV jitter) and returns the solution and
    its detection statistic (the chi2, or -2 ln L if the jitter is fit); if settling changes the RV weights, the new
    planet is searched for again with the settled jitter. A model with N planets has
    n_base_params + n_params_per_planet*N free parameters, which must leave at least one degree of freedom.
    Returns P, phi, e (arrays), objective, the statistic for 0, 1, ... planets, the rejected candidate, the number of
    planets searched, and whether this was a blind search.
    '''
    if delta_chi2_min is None:
        delta_chi2_min = DEFAULT_DELTA_CHI2_MIN
    if optimizer not in OPTIMIZER_SETTINGS:
        raise ValueError('optimizer must be one of %s.' % list(OPTIMIZER_SETTINGS))
    settings = dict(optimizer=optimizer, n_joint=int(n_joint))
    blind = n_planets is None and initial_planets is None
    n_target = max_planets if blind else (len(initial_planets) if initial_planets is not None else int(n_planets))
    if n_target < 0 or n_target > MAX_PLANETS_FIT:
        raise ValueError('the number of planets must be between 0 and %d.' % MAX_PLANETS_FIT)
    n_usable = prob.n + prob.n_rv - prob.reject
    if not blind and n_usable <= n_base_params + n_params_per_planet*n_target:
        raise ValueError('%d planets (%d parameters) cannot be fit to %d usable measurements.' % (n_target, n_base_params + n_params_per_planet*n_target, n_usable))
    if blind:
        n_target = min(n_target, max(0, (n_usable - n_base_params - 1)//n_params_per_planet))

    empty = np.zeros(0)
    chi2_0 = prob.chi2(empty, empty, empty)[0]
    sol, stat = settle(prob, (empty, empty, empty, chi2_0, chi2_0))
    history = [stat]
    rejected = None
    if initial_planets is not None:
        if len(initial_planets) > 0:
            P0 = np.array([float(p['period']) for p in initial_planets])
            sol = prob.polish(P0, 2*np.pi*np.mod(np.array([float(p['Tp']) for p in initial_planets])/P0, 1),
                              np.array([float(p['ecc']) for p in initial_planets]), np.ones(len(P0), dtype=np.int32))
            sol, stat = settle(prob, sol)
            history.append(stat)
    else:
        def grow(beam):
            children = [child for parent in beam for child in _search_one_planet(prob, *parent[:3], settings)]
            if refine and len(beam[0][0]) >= 1: # refine every child: a poor-looking child can refine into the best solution
                children = [_refine_planets(prob, child, settings) for child in _dedupe(children, len(children))]
            return _dedupe(children, n_beam)

        beam = [sol]
        while len(sol[0]) < n_target:
            jitter_before = prob.jitter.copy()
            children = grow(beam)
            best, stat = settle(prob, children[0])
            # the search weighted the data with the parent's jitter, which also absorbed the new planet's RV signal; if
            # settling changed the weights, search again with the settled jitter and keep the better solution
            searched_with = jitter_before
            for _ in range(3):
                jitter_best = prob.jitter.copy()
                if not _rv_weights_changed(prob, searched_with, jitter_best):
                    break
                searched_with = jitter_best
                retry = grow(beam)
                retry_best, retry_stat = settle(prob, retry[0])
                if retry_stat < stat - 1e-6:
                    children, best, stat = retry, retry_best, retry_stat
                else:
                    prob.set_jitter(jitter_best)
                    break
            gain = history[-1] - stat
            if verbose:
                print('%d planets: P = %s d, delta = %.1f' % (len(sol[0]) + 1, np.array2string(np.sort(best[0]), precision=1), gain))
            if blind and gain < delta_chi2_min:
                rejected = dict(periods=np.sort(best[0]), delta_chi2=gain)
                prob.set_jitter(jitter_before) # back to the accepted model
                break
            beam = [best] + children[1:]
            sol = best
            history.append(stat)
    return sol, history, rejected, n_target, blind

def _leave_one_out(prob, sol, history, settle):
    '''For each planet (in order of period), the increase of the detection statistic when it is removed and the other
    planets are re-polished (and settled). The problem's adjustable parameters (jitter) are restored afterwards.'''
    P, phi, e = sol[:3]
    jitter = prob.jitter.copy()
    _, stat_full = settle(prob, sol)
    deltas = []
    for i in np.argsort(P):
        others = [j for j in range(len(P)) if j != i]
        if others:
            reduced = prob.polish(P[others], phi[others], e[others], np.ones(len(others), dtype=np.int32))
            stat_without = settle(prob, reduced)[1]
        else:
            stat_without = history[0]
        deltas.append(stat_without - stat_full)
        prob.set_jitter(jitter)
    return deltas

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

def _astrometric_goodness_of_fit(chi2_data, n_used, n_par, binned):
    '''
    F2, uwe, and the factor by which gaiamock inflates the uncertainties, from the astrometric chi2 of n_used measurements
    and n_par parameters. If binned, each measurement is the mean of 8 CCD measurements, and the chi2 is converted to
    that of the unbinned data, as in gaiamock.
    '''
    nu = n_used - n_par
    chi2_red = chi2_data/nu
    if binned:
        chi2_red_unbinned = gaiamock.predict_reduced_chi2_unbinned_data(chi2_red_binned=chi2_red, n_param=n_par, N_points=n_used, Nbin=8)
        nu_unbinned = 8*n_used - n_par
        cc = np.sqrt(chi2_red_unbinned/(1 - 2/(9*nu_unbinned))**3)
        F2 = gaiamock.predict_F2_unbinned_data(chi2_red_binned=chi2_red, n_param=n_par, N_points=n_used, Nbin=8)
        uwe = np.sqrt(chi2_red_unbinned)
    else:
        cc = np.sqrt(chi2_red/(1 - 2/(9*nu))**3)
        F2 = np.sqrt(9*nu/2)*(chi2_red**(1/3) + 2/(9*nu) - 1)
        uwe = np.sqrt(chi2_red)
    return F2, uwe, cc

def _astrometry_fit_summary(prob, P, phi, e, m_star, binned):
    '''Linear parameters, uncertainties (from the Jacobian, inflated by the goodness of fit as in gaiamock), and derived quantities.'''
    order = np.argsort(P)
    P, phi, e = P[order], phi[order], e[order]
    K = len(P)
    chi2_total, mu, worst, _ = prob.chi2(P, phi, e)
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
    F2, uwe, cc = _astrometric_goodness_of_fit(chi2_data, n_used, n_par, binned)
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
    optimizer: gaiamock's optimizer used to search for each planet: 'periodogram_multistart' (default),
        'periodogram_multistart_full' (denser grids, more starts, and tighter polish), 'multistart' (circular-orbit
        grid instead of the linear periodogram; slower, with nearly identical results), or 'annealing' (adaptive
        simulated annealing; slower, and stochastic: it uses the C rand() without reseeding, as gaiamock does).
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
    sol, history, rejected, n_target, blind = _fit_planets(prob, n_planets, max_planets, delta_chi2_min, initial_planets, optimizer, n_joint,
                                                           refine, n_beam, verbose, 5, 7)
    result = _astrometry_fit_summary(prob, *sol[:3], m_star, binned)
    for planet, delta in zip(result['planets'], _leave_one_out(prob, sol, history, _settle_none)):
        planet['delta_chi2'] = delta
    result['chi2_history'] = history
    result['objective'] = sol[3]
    if blind:
        result['rejected_candidate'] = rejected
        result['max_planets_searched'] = n_target
    return result

def _fit_rv_jitter(prob, P, phi, e, max_sweeps=50):
    '''
    Maximize the likelihood over the jitter of each RV dataset at fixed nonlinear orbital parameters (the linear
    parameters are re-solved at every evaluation), by coordinate descent: a bounded 1D minimization in log(jitter) for
    each dataset, plus a check of zero jitter, repeated until -2 ln L stops improving. Leaves the result in prob.
    '''
    from scipy.optimize import minimize_scalar
    jitter = prob.jitter.copy()
    scale = np.ptp(prob.rv) + np.max(prob.rv_err)
    log_lo, log_hi = np.log(1e-6*scale), np.log(10*scale)
    def stat(s, k):
        trial = jitter.copy(); trial[k] = s
        prob.set_jitter(trial)
        return prob.minus_2_ln_like(prob.chi2(P, phi, e)[0])
    previous = np.inf
    for _ in range(max_sweeps):
        for k in range(prob.n_sets):
            res = minimize_scalar(lambda u: stat(np.exp(u), k), bounds=(log_lo, log_hi), method='bounded', options=dict(xatol=1e-7))
            candidates = [(res.fun, np.exp(res.x)), (stat(0.0, k), 0.0), (stat(jitter[k], k), jitter[k])]
            jitter[k] = min(candidates)[1]
        prob.set_jitter(jitter)
        current = prob.minus_2_ln_like(prob.chi2(P, phi, e)[0])
        if previous - current < 1e-8:
            break
        previous = current
    prob.set_jitter(jitter)

def _settle_rv(prob, sol):
    '''Alternate between fitting the RV jitter (orbits fixed) and polishing the orbits (jitter fixed) until -2 ln L converges.'''
    P, phi, e, objective, chi2 = sol
    previous = np.inf
    for _ in range(20):
        _fit_rv_jitter(prob, P, phi, e)
        if len(P) > 0:
            P, phi, e, objective, chi2 = prob.polish(P, phi, e, np.ones(len(P), dtype=np.int32))
        else:
            chi2 = objective = prob.chi2(P, phi, e)[0]
        stat = prob.minus_2_ln_like(chi2)
        if previous - stat < 1e-6:
            break
        previous = stat
    return (P, phi, e, objective, chi2), stat

def _settle_rv_fixed_jitter(prob, sol):
    '''Fixed jitter: the detection statistic is -2 ln L (= chi2 + a constant).'''
    return sol, prob.minus_2_ln_like(sol[4])

def _derived_rv_quantities(theta, cov, i, n_sets, m_star):
    '''
    Tp, K, w (and m sin i if m_star is given) of planet i, and their uncertainties propagated from the covariance
    matrix of theta (gamma per RV dataset, then P, e, phi_p, K cos(w), K sin(w) per planet) with a central-difference
    Jacobian. Differences of w (defined modulo 2 pi) are wrapped. m sin(i) is the companion mass for sin(i) = 1,
    computed exactly as the inverse of get_planet_K_kms() with inc = pi/2.
    '''
    AU = 1.4959787e+11 # as in gaiamock.get_a0_mas()
    sig = np.sqrt(np.abs(np.diag(cov)))
    def derived(th):
        Pi, ei, phii, h, c = th[n_sets + 5*i: n_sets + 5*i + 5]
        K = np.hypot(h, c)
        out = [phii*Pi/(2*np.pi), K, np.mod(np.arctan2(c, h), 2*np.pi)]
        if m_star is not None: # a_star sin(i) in AU, then the companion mass for sin(i) = 1
            a_sini_au = K*1e3*Pi*86400*np.sqrt(1 - ei**2)/(2*np.pi*AU)
            out.append(get_dark_companion_mass(a0_mas=a_sini_au, period=Pi, parallax=1.0, m1=m_star))
        return np.array(out)
    g0 = derived(theta)
    Jg = np.zeros((len(g0), len(theta)))
    for j in range(len(theta)):
        h = 1e-4*sig[j] if (np.isfinite(sig[j]) and sig[j] > 0) else 1e-8
        up, down = theta.copy(), theta.copy()
        up[j] += h; down[j] -= h
        dg = derived(up) - derived(down)
        dg[2] = np.angle(np.exp(1j*dg[2])) # w is defined modulo 2 pi
        Jg[:, j] = dg/(2*h)
    return g0, np.sqrt(np.abs(np.diag(Jg @ cov @ Jg.T)))

def _rv_fit_summary(prob, P, phi, e, m_star):
    '''Linear parameters, uncertainties (from the Jacobian of the RV residuals, at the fitted jitter), and derived quantities.'''
    order = np.argsort(P)
    P, phi, e = P[order], phi[order], e[order]
    K_pl = len(P)
    chi2, _, _, mu = prob.chi2(P, phi, e)
    S = prob.n_sets
    theta = list(mu[:S])
    names = ['gamma_%d' % k for k in range(S)]
    for i in range(K_pl):
        h, c = mu[S + 2*i: S + 2*i + 2]
        theta += [P[i], e[i], phi[i], h, c]
        names += ['%s_%d' % (k, i) for k in ('period', 'ecc', 'phi_p', 'K_cos_w', 'K_sin_w')]
    theta = np.array(theta)

    # Jacobian of the normalized residuals (central differences, step 1e-8 as in gaiamock)
    J = np.zeros((prob.n_rv, len(theta)))
    for j in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[j] += 1e-8; down[j] -= 1e-8
        J[:, j] = (prob.rv_residuals(up, K_pl) - prob.rv_residuals(down, K_pl))/2e-8
    JTJ = J.T @ J
    if prob.ecc_a > 0:
        for i in range(K_pl):
            ei = np.clip(e[i], 1e-6, 1 - 1e-6)
            H_e = (prob.ecc_a - 1)/ei**2 + (prob.ecc_b - 1)/(1 - ei)**2
            if H_e > 0:
                JTJ[S + 5*i + 1, S + 5*i + 1] += H_e
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.full_like(JTJ, np.nan)
    sig = np.sqrt(np.abs(np.diag(cov)))
    n_par = len(theta)
    result = dict(n_planets=K_pl, chi2=chi2, minus_2_ln_like=prob.minus_2_ln_like(chi2), n_rv=prob.n_rv, n_params=n_par,
                  nu=prob.n_rv - n_par, chi2_red=chi2/(prob.n_rv - n_par) if prob.n_rv > n_par else np.nan,
                  gamma=theta[:S].copy(), sig_gamma=sig[:S].copy(), jitter=prob.jitter.copy(), theta=theta, theta_names=names, cov=cov)

    planets, model_planets = [], []
    for i in range(K_pl):
        g0, sig_g = _derived_rv_quantities(theta, cov, i, S, m_star)
        planet = {}
        for j, key in enumerate(('period', 'ecc', 'phi_p', 'K_cos_w', 'K_sin_w')):
            planet[key], planet['sig_' + key] = theta[S + 5*i + j], sig[S + 5*i + j]
        for j, key in enumerate(('Tp', 'K', 'w') + (('m_sini',) if m_star is not None else ())):
            planet[key], planet['sig_' + key] = g0[j], sig_g[j]
        planets.append(planet)
        if m_star is not None:
            model_planets.append(dict(period=planet['period'], Tp=planet['Tp'], ecc=planet['ecc'], inc=np.pi/2, w=planet['w'], m_planet=planet['m_sini']))
    result['planets'] = planets
    result['model_planets'] = model_planets if m_star is not None else None
    return result

def fit_rv_planets(t_rvs_yr, rv, rv_err, rv_dataset=None, n_planets=None, max_planets=3, delta_chi2_min=None, m_star=None,
                   P_min=10., P_max=1e4, e_max=0.99, eccentricity_prior=None, fit_jitter=True, jitter=None, initial_planets=None,
                   optimizer='periodogram_multistart', n_joint=3, refine=True, n_beam=1, verbose=False, c_funcs_plus=None):
    '''
    Fit radial velocities with a model of a star hosting several planets, adding one planet at a time, with the same
    search as fit_astrometry_planets(): gaiamock's "periodogram_multistart" optimizer for each new planet (the
    circular-orbit periodogram step becomes the generalized Lomb-Scargle periodogram with one offset per RV dataset),
    then a joint polish of all planets and (by default) a re-search of each planet with the others held fixed. The
    linear parameters (one systemic velocity per dataset, and K cos(w), K sin(w) for each planet) are always solved
    exactly. The RV model is that of rv_planets_model(): w is the argument of periastron of the star's orbit.

    t_rvs_yr: RV times in years relative to the reference epoch (e.g. (BJD - 2457936.875)/365.25 for DR4)
    rv, rv_err: RVs and their uncertainties in km/s
    rv_dataset: None (one dataset) or the dataset (instrument) index 0, 1, ..., n_sets - 1 of each RV
    n_planets: number of planets to fit (known-N mode). If None (default), search blindly: keep adding planets while
        -2 ln L improves by more than delta_chi2_min, up to max_planets.
    max_planets: largest number of planets considered in blind-search mode (default 3); fewer are tried if there are
        not enough RVs (a model with N planets has n_sets + 5N parameters, plus n_sets jitters if they are fit).
    delta_chi2_min: detection threshold on the improvement of -2 ln L (= chi2 for fixed jitter) from adding a planet,
        used in blind-search mode (default DEFAULT_DELTA_CHI2_MIN = 50; in simulations of noise with 40-200 RVs from
        1-2 instruments over 6 years, with fitted jitter, spurious planets reached up to ~46, and exceeded 40 in ~1%).
    m_star: stellar mass in Msun. If given, the minimum masses m sin(i) (and uncertainties) are computed, and
        model_planets are returned in the format of rv_planets_model() (with inc = pi/2 and m_planet = m sin(i)).
    P_min, P_max: period search range in days (default 10 to 1e4, as in gaiamock). The search grids are gaiamock's,
        which become too coarse for periods much shorter than ~10 days over multi-year baselines.
    e_max: largest eccentricity allowed (default 0.99, as in gaiamock).
    eccentricity_prior: None (default) or (a, b): Beta(a, b) prior on each eccentricity, included in the objective
        that is minimized (but not in the statistic used for detection).
    fit_jitter: if True (default), fit one jitter per RV dataset (added in quadrature to rv_err) by maximum likelihood.
        The jitter is held fixed while searching for each new planet, and then re-fit (alternating with polishing the
        orbits) once the planet is added. If False, the jitter is fixed at its initial value.
    jitter: initial (or, if fit_jitter is False, fixed) jitter in km/s: None (zero), a number, or one value per dataset.
    initial_planets: optional list of starting guesses, each a dict with keys 'period', 'Tp' (days, relative to the
        reference epoch) and 'ecc'. If given, no search is done: the planets are polished jointly from these guesses.
    optimizer, n_joint, refine, n_beam: search settings, as in fit_astrometry_planets().
    c_funcs_plus: from read_in_C_functions_plus() (read in automatically if None).

    returns a dict with:
        n_planets, planets (list of dicts, sorted by period, each with period, ecc, phi_p, Tp, K (km/s), w (radians),
            K_cos_w, K_sin_w, [m_sini (Msun)], their uncertainties sig_*, and delta_chi2: the increase in -2 ln L when
            that planet is removed and the other planets (and the jitter) are re-fit),
        gamma, sig_gamma (one systemic velocity per dataset, km/s), jitter (km/s),
        chi2 (with the jitter), minus_2_ln_like, chi2_history (-2 ln L, up to a constant, for 0, 1, 2, ... planets),
        n_rv, n_params, nu, chi2_red, theta, theta_names, cov, model_planets, and, in blind-search mode,
        rejected_candidate and max_planets_searched (as in fit_astrometry_planets()).
    Notes: uncertainties are computed from the Jacobian at the fitted jitter (which accounts for any excess scatter),
    without the goodness-of-fit inflation that gaiamock applies to astrometry, and do not include the uncertainty of
    the jitter. RVs alone do not constrain inc or Omega.
    '''
    prob = _Problem(rv=(t_rvs_yr, rv, rv_err, rv_dataset), eccentricity_prior=eccentricity_prior, P_min=P_min, P_max=P_max, e_max=e_max,
                    c_funcs_plus=c_funcs_plus)
    if jitter is not None:
        jitter = np.broadcast_to(np.asarray(jitter, dtype=float), (prob.n_sets,))
        if np.any(jitter < 0) or not np.all(np.isfinite(jitter)):
            raise ValueError('jitter must be non-negative.')
        prob.set_jitter(jitter)
    settle = _settle_rv if fit_jitter else _settle_rv_fixed_jitter
    n_base = prob.n_sets*(2 if fit_jitter else 1)
    sol, history, rejected, n_target, blind = _fit_planets(prob, n_planets, max_planets, delta_chi2_min, initial_planets, optimizer, n_joint,
                                                           refine, n_beam, verbose, n_base, 5, settle)
    result = _rv_fit_summary(prob, *sol[:3], m_star)
    for planet, delta in zip(result['planets'], _leave_one_out(prob, sol, history, settle)):
        planet['delta_chi2'] = delta
    result['chi2_history'] = history
    result['objective'] = sol[3]
    if blind:
        result['rejected_candidate'] = rejected
        result['max_planets_searched'] = n_target
    return result

# ================================================================ joint fitting of astrometry and RVs

def _ecc_prior_offset(a, b):
    '''-2 ln Beta(mode; a, b), so that -2 ln Beta(e; a, b) minus this is >= 0 and can enter a least-squares fit as a squared residual.'''
    from scipy.stats import beta
    if a < 1 or b < 1:
        raise ValueError('the joint fit supports eccentricity priors Beta(a, b) with a, b >= 1 only.')
    mode = 0.5 if a + b == 2 else (a - 1)/(a + b - 2)
    return -2*beta.logpdf(mode, a, b)

class _JointModel:
    '''
    Physical model of astrometry + RVs for N planets, with
        theta = (ra_off, dec_off, parallax, pmra, pmdec, then for each planet: P, e, phi_p, Omega, inc, w, a0_mas),
    and the conventions of astrometry_and_rv_planets_model(): Thiele-Innes elements of the star's reflex orbit, and RVs
    K [cos(nu + w) + e cos(w)] with the same w and K = 2 pi (a0/parallax) [AU] sin(inc)/(P sqrt(1 - e^2)), which needs
    no stellar mass. The systemic velocity of each RV dataset is solved analytically (weighted mean of its residuals).
    The RV errors include the jitter of the problem (prob.rv_err_eff); keep selects the astrometric epochs used.
    '''
    AU = 1.4959787e+11 # as in gaiamock.get_a0_mas()

    def __init__(self, prob, c_funcs, keep, ast_scale=1.0):
        self.prob, self.c_funcs, self.keep, self.ast_scale = prob, c_funcs, keep, ast_scale
        self.ecc_offset = _ecc_prior_offset(prob.ecc_a, prob.ecc_b) if prob.ecc_a > 0 else None

    def signals(self, theta):
        '''the astrometric model (mas), and the sum of the planets' RV signals (km/s, without systemic velocities)'''
        t, psi, plx_f = self.prob.data[0], self.prob.data[1], self.prob.data[2]
        ra_off, dec_off, plx, pmra, pmdec = theta[:5]
        ast = (ra_off + pmra*t)*np.sin(psi) + (dec_off + pmdec*t)*np.cos(psi) + plx*plx_f
        rv = np.zeros(self.prob.n_rv)
        for i in range((len(theta) - 5)//7):
            P, e, phi, Omega, inc, w, a0 = theta[5 + 7*i: 12 + 7*i]
            Tp = phi*P/(2*np.pi)
            ast = ast + keplerian_al_signal(t, psi, P, Tp, e, *get_thiele_innes(a0, Omega, inc, w), self.c_funcs)
            K = 2*np.pi*(a0/plx)*self.AU*np.sin(inc)/(P*86400*np.sqrt(1 - e**2))/1e3
            rv = rv + keplerian_rv_signal(self.prob.t_rv, P, Tp, e, w, K, self.c_funcs)
        return ast, rv

    def gamma(self, rv_planets):
        '''systemic velocity of each RV dataset that minimizes the chi2 (weighted mean of the residuals)'''
        wts = 1/self.prob.rv_err_eff**2
        y = self.prob.rv - rv_planets
        return np.array([np.sum((wts*y)[self.prob.rv_set == k])/np.sum(wts[self.prob.rv_set == k]) for k in range(self.prob.n_sets)])

    def residuals(self, theta, gamma=None):
        '''normalized astrometric and RV residuals, plus prior terms as residuals (systemic velocities solved unless given)'''
        ast, rv_pl = self.signals(theta)
        if gamma is None:
            gamma = self.gamma(rv_pl)
        r = [((ast - self.prob.data[3])/self.prob.data[4])[self.keep]/self.ast_scale,
             (rv_pl + gamma[self.prob.rv_set] - self.prob.rv)/self.prob.rv_err_eff]
        if self.prob.plx_sig > 0:
            r.append([(theta[2] - self.prob.plx_mu)/self.prob.plx_sig])
        if self.ecc_offset is not None:
            from scipy.stats import beta
            e = theta[6::7]
            r.append(np.sqrt(np.maximum(-2*beta.logpdf(e, self.prob.ecc_a, self.prob.ecc_b) - self.ecc_offset, 0)))
        return np.concatenate(r)

    def chi2_parts(self, theta):
        '''astrometric chi2 (epochs used), RV chi2 (with jitter), parallax-prior term'''
        ast, rv_pl = self.signals(theta)
        gamma = self.gamma(rv_pl)
        chi2_ast = np.sum((((ast - self.prob.data[3])/self.prob.data[4])[self.keep])**2)
        chi2_rv = np.sum(((rv_pl + gamma[self.prob.rv_set] - self.prob.rv)/self.prob.rv_err_eff)**2)
        prior = (theta[2] - self.prob.plx_mu)**2/self.prob.plx_sig**2 if self.prob.plx_sig > 0 else 0.0
        return chi2_ast, chi2_rv, prior

    def minus_2_ln_like(self, theta):
        '''-2 ln L up to a constant (data + parallax prior; the same convention as _Problem.minus_2_ln_like)'''
        return self.prob.minus_2_ln_like(sum(self.chi2_parts(theta)))

    def fit_jitter(self, theta):
        '''
        Maximize the likelihood over the jitter of each RV dataset with the orbits fixed. With the planets' RV signals fixed,
        each dataset depends only on its own jitter and systemic velocity, so each jitter is a bounded 1D minimization in
        log(jitter) (with the systemic velocity re-solved), plus a check of zero jitter.
        '''
        from scipy.optimize import minimize_scalar
        _, rv_pl = self.signals(theta)
        y = self.prob.rv - rv_pl
        jitter = self.prob.jitter.copy()
        scale = np.ptp(self.prob.rv) + np.max(self.prob.rv_err)
        for k in range(self.prob.n_sets):
            yk, ek = y[self.prob.rv_set == k], self.prob.rv_err[self.prob.rv_set == k]
            def stat(s):
                v = ek**2 + s**2
                g = np.sum(yk/v)/np.sum(1/v)
                return np.sum((yk - g)**2/v + np.log(2*np.pi*v))
            res = minimize_scalar(lambda u: stat(np.exp(u)), bounds=(np.log(1e-6*scale), np.log(10*scale)), method='bounded', options=dict(xatol=1e-7))
            jitter[k] = min([(res.fun, np.exp(res.x)), (stat(0.0), 0.0)])[1]
        self.prob.set_jitter(jitter)

def _joint_start(prob, sol):
    '''
    Physical starting parameters from a linked solution (shared P, phi_p, e; separate astrometric Thiele-Innes and RV
    amplitudes): Campbell elements from the Thiele-Innes elements, with the (Omega, w) -> (Omega + pi, w + pi) branch
    chosen to match the w of the RV solution.
    '''
    P, phi, e = sol[:3]
    _, mu_ast, worst, mu_rv = prob.chi2(P, phi, e)
    theta = [mu_ast[0], mu_ast[2], mu_ast[4], mu_ast[1], mu_ast[3]]
    S = prob.n_sets
    for i in range(len(P)):
        B, G, A, F = mu_ast[5 + 4*i: 9 + 4*i]
        a0, Omega, w, inc = gaiamock.get_Campbell_elements(A, B, F, G)
        w_rv = np.arctan2(mu_rv[S + 2*i + 1], mu_rv[S + 2*i])
        if abs(np.angle(np.exp(1j*(w - w_rv)))) > np.pi/2:
            Omega, w = Omega + np.pi, w + np.pi
        theta += [P[i], e[i], phi[i], Omega, inc, w, a0]
    return np.array(theta), worst

def _refine_joint(prob, sol, c_funcs, fit_jitter, max_cycles=20):
    '''
    Exact joint fit of astrometry and RVs, starting from a linked solution: least squares over the physical parameters
    (jitter fixed), alternating with the jitter fit, until -2 ln L converges. With outlier rejection, the rejected
    astrometric epoch is the worst one of the linked solution, re-checked once against the joint solution.
    Returns theta, the model (holding the epochs used), and -2 ln L.
    '''
    from scipy.optimize import least_squares
    theta, worst = _joint_start(prob, sol)
    n_pl = (len(theta) - 5)//7
    lo = np.full(len(theta), -np.inf); hi = np.full(len(theta), np.inf)
    lo[2] = 1e-6 # the RV amplitude needs a positive parallax
    for i in range(n_pl):
        k = 5 + 7*i
        lo[k], hi[k] = prob.L[0], prob.U[0]               # P
        lo[k + 1], hi[k + 1] = prob.L[2], prob.U[2]       # e
        lo[k + 4], hi[k + 4] = 0., np.pi                  # inc
        lo[k + 6] = 0.                                    # a0
    theta = np.clip(theta, lo, hi)
    for attempt in range(2):
        keep = np.ones(prob.n, dtype=bool)
        if worst >= 0:
            keep[worst] = False
        model = _JointModel(prob, c_funcs, keep)
        previous = np.inf
        for _ in range(max_cycles):
            if n_pl > 0 or prob.n_rv > 0:
                theta = least_squares(model.residuals, theta, bounds=(lo, hi), x_scale='jac', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000).x
            if fit_jitter and prob.n_rv > 0:
                model.fit_jitter(theta)
            stat = model.minus_2_ln_like(theta)
            if previous - stat < 1e-6:
                break
            previous = stat
        if not prob.reject:
            break
        ast, _ = model.signals(theta) # re-check the rejected epoch against the joint solution
        new_worst = int(np.argmax(((ast - prob.data[3])/prob.data[4])**2))
        if new_worst == worst:
            break
        worst = new_worst
    return theta, model, stat

def _joint_fit_summary(model, theta, m_star, binned):
    '''Uncertainties (from the Jacobian of all residuals; astrometric uncertainties inflated as in gaiamock) and derived quantities.'''
    prob = model.prob
    n_pl = (len(theta) - 5)//7
    S = prob.n_sets
    # order the planets by period, and wrap the angles
    order = np.argsort(theta[5::7])
    theta = np.concatenate([theta[:5]] + [theta[5 + 7*i: 12 + 7*i] for i in order])
    for i in range(n_pl):
        k = 5 + 7*i
        theta[k + 2] = np.mod(theta[k + 2], 2*np.pi); theta[k + 3] = np.mod(theta[k + 3], 2*np.pi); theta[k + 5] = np.mod(theta[k + 5], 2*np.pi)
    _, rv_pl = model.signals(theta)
    gamma = model.gamma(rv_pl)
    chi2_ast, chi2_rv, prior = model.chi2_parts(theta)
    n_used = int(model.keep.sum())
    F2, uwe, cc = _astrometric_goodness_of_fit(chi2_ast, n_used, 5 + 7*n_pl, binned)

    # Jacobian of all residuals with respect to (theta, gamma), with the astrometric errors scaled by cc as in gaiamock
    full = np.concatenate([theta, gamma])
    scaled = _JointModel(prob, model.c_funcs, model.keep, ast_scale=cc)
    def resid(x):
        return scaled.residuals(x[:len(theta)], gamma=x[len(theta):])
    J = np.zeros((len(resid(full)), len(full)))
    for j in range(len(full)):
        h = 1e-7*max(abs(full[j]), 1e-2)
        up, down = full.copy(), full.copy()
        up[j] += h; down[j] -= h
        J[:, j] = (resid(up) - resid(down))/(2*h)
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((len(full), len(full)), np.nan)
    sig = np.sqrt(np.abs(np.diag(cov)))
    names = ['ra_off', 'dec_off', 'parallax', 'pmra', 'pmdec'] + ['%s_%d' % (k, i) for i in range(n_pl) for k in ('period', 'ecc', 'phi_p', 'Omega', 'inc', 'w', 'a0_mas')] \
            + ['gamma_%d' % k for k in range(S)]

    result = dict(n_planets=n_pl, minus_2_ln_like=model.minus_2_ln_like(theta), chi2=chi2_ast + chi2_rv + prior, chi2_ast=chi2_ast, chi2_rv=chi2_rv,
                  n_obs=prob.n, n_used=n_used, n_rv=prob.n_rv, F2=F2, uwe=uwe, uncertainty_scale=cc,
                  outlier_index=int(np.where(~model.keep)[0][0]) if (~model.keep).any() else -1,
                  gamma=gamma, sig_gamma=sig[len(theta):], jitter=prob.jitter.copy(), theta=full, theta_names=names, cov=cov)
    for j, name in enumerate(names[:5]):
        result[name], result['sig_' + name] = full[j], sig[j]

    def derived(x, i):
        '''Tp, K (km/s), inc in degrees (and m_planet if m_star is given) of planet i'''
        P, e, phi, Omega, inc, w, a0 = x[5 + 7*i: 12 + 7*i]
        K = 2*np.pi*(a0/x[2])*_JointModel.AU*np.sin(inc)/(P*86400*np.sqrt(1 - e**2))/1e3
        out = [phi*P/(2*np.pi), K, np.degrees(inc)]
        if m_star is not None:
            out.append(get_dark_companion_mass(a0_mas=a0, period=P, parallax=x[2], m1=m_star))
        return np.array(out)

    planets, model_planets = [], []
    for i in range(n_pl):
        g0 = derived(full, i)
        Jg = np.zeros((len(g0), len(full)))
        for j in range(len(full)):
            h = 1e-4*sig[j] if (np.isfinite(sig[j]) and sig[j] > 0) else 1e-8
            up, down = full.copy(), full.copy()
            up[j] += h; down[j] -= h
            Jg[:, j] = (derived(up, i) - derived(down, i))/(2*h)
        sig_g = np.sqrt(np.abs(np.diag(Jg @ cov @ Jg.T)))
        planet = {}
        for j, key in enumerate(('period', 'ecc', 'phi_p', 'Omega', 'inc', 'w', 'a0_mas')):
            planet[key], planet['sig_' + key] = full[5 + 7*i + j], sig[5 + 7*i + j]
        for j, key in enumerate(('Tp', 'K', 'inc_deg') + (('m_planet',) if m_star is not None else ())):
            planet[key], planet['sig_' + key] = g0[j], sig_g[j]
        planets.append(planet)
        if m_star is not None:
            model_planets.append({k: planet[k] for k in PLANET_KEYS})
    result['planets'] = planets
    result['model_planets'] = model_planets if m_star is not None else None
    return result

def fit_astrometry_and_rv_planets(t_ast_yr, psi, plx_factor, ast_obs, ast_err, t_rvs_yr, rv, rv_err, rv_dataset=None, n_planets=None,
                                  max_planets=3, delta_chi2_min=None, m_star=None, P_min=10., P_max=1e4, e_max=0.99, parallax_prior=None,
                                  eccentricity_prior=None, reject_outlier=False, binned=True, fit_jitter=True, jitter=None, initial_planets=None,
                                  optimizer='periodogram_multistart', n_joint=3, refine=True, n_beam=1, verbose=False, c_funcs=None, c_funcs_plus=None):
    '''
    Fit Gaia epoch astrometry and radial velocities jointly with a model of a star hosting several planets, in two steps:
    1. A linked search, with the same search as fit_astrometry_planets() and fit_rv_planets() (gaiamock's
       "periodogram_multistart" optimizer for each new planet, then a joint polish and a re-search of each planet with
       the others held fixed). Each planet's period, phase and eccentricity are shared by the astrometry and the RVs,
       whose amplitudes (Thiele-Innes elements; K cos(w), K sin(w)) are solved separately, so all the search machinery
       applies unchanged.
    2. An exact joint fit, starting from the linked solution: least squares over the physical parameters (ra_off, dec_off,
       parallax, pmra, pmdec, and P, e, phi_p, Omega, inc, w, a0 for each planet), in which the RVs follow from the
       astrometric orbit (the model of astrometry_and_rv_planets_model(), with w the star's argument of periastron and
       K = 2 pi a0 sin(inc)/(P sqrt(1 - e^2)), a0 in AU), alternating with the jitter fit. The systemic velocities are
       solved analytically. With RVs, Omega is determined in [0, 2 pi) (astrometry alone leaves Omega and Omega + pi).

    Arguments are as in fit_astrometry_planets() (astrometry and search settings) and fit_rv_planets() (RVs and jitter):
    t_ast_yr, psi, plx_factor, ast_obs, ast_err: epoch astrometry. t_rvs_yr, rv, rv_err, rv_dataset: RVs (km/s), with
        times relative to the same reference epoch.
    n_planets (known-N mode) or blind search (default) with max_planets and delta_chi2_min (applied to -2 ln L of the
        linked model); initial_planets: starting guesses (dicts with 'period', 'Tp', 'ecc') instead of a search.
    m_star: if given, planet masses and their uncertainties, and model_planets for astrometry_and_rv_planets_model().
    parallax_prior, eccentricity_prior (Beta(a, b) with a, b >= 1), reject_outlier (astrometry), binned, P_min, P_max,
        e_max, fit_jitter, jitter, optimizer, n_joint, refine, n_beam: as in fit_astrometry_planets()/fit_rv_planets().
    c_funcs: gaiamock's compiled functions (read_in_C_functions(); read in automatically if None).

    returns a dict with:
        n_planets, planets (sorted by period; each with period, ecc, phi_p, Tp, Omega, inc, inc_deg, w, a0_mas, K,
            [m_planet], their uncertainties sig_*, and delta_chi2 from the linked model, as in fit_astrometry_planets()),
        ra_off, dec_off, parallax, pmra, pmdec (and sig_*), gamma, sig_gamma, jitter,
        minus_2_ln_like, chi2, chi2_ast, chi2_rv, F2, uwe and uncertainty_scale (of the astrometry, as in gaiamock),
        linked_minus_2_ln_like, consistency_delta (= minus_2_ln_like - linked_minus_2_ln_like >= 0; it is large if the
        astrometric and RV orbits disagree), chi2_history (linked -2 ln L for 0, 1, 2, ... planets), theta, theta_names,
        cov, model_planets, outlier_index, and in blind-search mode rejected_candidate and max_planets_searched.
    Notes: astrometric uncertainties are inflated by gaiamock's goodness-of-fit factor (uncertainty_scale); the
    uncertainty of the jitter is not propagated.
    '''
    prob = _Problem(ast=(t_ast_yr, psi, plx_factor, ast_obs, ast_err), rv=(t_rvs_yr, rv, rv_err, rv_dataset), parallax_prior=parallax_prior,
                    eccentricity_prior=eccentricity_prior, reject_outlier=reject_outlier, P_min=P_min, P_max=P_max, e_max=e_max,
                    c_funcs_plus=c_funcs_plus)
    if eccentricity_prior is not None:
        _ecc_prior_offset(*eccentricity_prior) # check that the prior is supported
    if jitter is not None:
        jitter = np.broadcast_to(np.asarray(jitter, dtype=float), (prob.n_sets,))
        if np.any(jitter < 0) or not np.all(np.isfinite(jitter)):
            raise ValueError('jitter must be non-negative.')
        prob.set_jitter(jitter)
    if c_funcs is None:
        c_funcs = gaiamock.read_in_C_functions()
    settle = _settle_rv if fit_jitter else _settle_rv_fixed_jitter
    n_base = 5 + prob.n_sets*(2 if fit_jitter else 1)
    sol, history, rejected, n_target, blind = _fit_planets(prob, n_planets, max_planets, delta_chi2_min, initial_planets, optimizer, n_joint,
                                                           refine, n_beam, verbose, n_base, 9, settle)
    deltas = _leave_one_out(prob, sol, history, settle)
    linked = settle(prob, sol)[1]
    theta, model, stat = _refine_joint(prob, sol, c_funcs, fit_jitter)
    result = _joint_fit_summary(model, theta, m_star, binned)
    for planet, delta in zip(result['planets'], deltas):
        planet['delta_chi2'] = delta
    result['linked_minus_2_ln_like'] = linked
    result['consistency_delta'] = result['minus_2_ln_like'] - linked
    result['chi2_history'] = history
    if blind:
        result['rejected_candidate'] = rejected
        result['max_planets_searched'] = n_target
    return result
