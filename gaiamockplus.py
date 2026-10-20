'''
gaiamockplus: extensions to gaiamock (El-Badry et al. 2024) for modeling the Gaia astrometry of exoplanet systems.

Everything in gaiamock is re-exported, so `import gaiamockplus as gaiamock` gives access to both the original
gaiamock functions and the additions below. New functionality lives in this file (rather than in gaiamock.py)
so that updates from the upstream gaiamock repository can be merged without conflicts.

Conventions follow gaiamock: times in days (Tp relative to the reference epoch of the data release), angles in
radians, masses in Msun, astrometry in mas. Thiele-Innes elements describe the photocenter (here, the host star's
reflex) orbit, so w is the argument of periastron of the star's orbit, which differs by pi from that of the planet.
'''
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

def validate_planets(planets):
    '''
    Check a list of planet dicts (see PLANET_KEYS) and return a copy with all values converted to floats.
    Raises ValueError with an informative message for missing/unknown keys or unphysical values.
    '''
    if isinstance(planets, dict):
        raise ValueError('planets must be a list of dicts (one per planet), not a single dict. Use [planet] for one planet.')
    checked = []
    for i, planet in enumerate(planets):
        if not isinstance(planet, dict):
            raise ValueError('planet %d is not a dict.' % i)
        if 'omega' in planet:
            raise ValueError("planet %d has key 'omega', which is ambiguous. Use 'Omega' for the longitude of the ascending node and 'w' for the argument of periastron." % i)
        missing = [k for k in PLANET_KEYS if k not in planet]
        unknown = [k for k in planet if k not in PLANET_KEYS]
        if missing or unknown:
            raise ValueError('planet %d: missing keys %s, unknown keys %s. Required keys: %s' % (i, missing, unknown, list(PLANET_KEYS)))
        p = {k: float(planet[k]) for k in PLANET_KEYS}
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

def get_planet_a0_mas(m_star, planets, parallax, jacobi=False):
    '''
    Calculate the semimajor axis of the star's reflex orbit (in mas) due to each planet.
    m_star: stellar mass in Msun
    planets: list of planet dicts (see PLANET_KEYS)
    parallax: 1/distance, in mas
    jacobi: if False (default), each planet orbits the star as an independent two-body system, with
        a^3 = G(m_star + m_planet)P^2/(4 pi^2) and a0 = a*m_planet/(m_star + m_planet).
        If True, use Jacobi coordinates: each planet orbits the barycenter of the star and all planets with
        shorter periods, so that m_star is replaced by m_star + (masses of inner planets). This is more accurate
        for hierarchical systems with massive planets; the difference is of order m_planet/m_star.
    returns an array with one a0 (mas) per planet, in the same order as planets
    '''
    planets = validate_planets(planets)
    if len(planets) == 0:
        return np.zeros(0)
    periods = np.array([p['period'] for p in planets])
    masses = np.array([p['m_planet'] for p in planets])

    central_mass = np.full(len(planets), float(m_star))
    if jacobi:
        order = np.argsort(periods, kind='stable') # ties keep the order of the list
        central_mass[order] = m_star + np.cumsum(masses[order]) - masses[order]
    return gaiamock.get_a0_mas(period = periods, m1 = central_mass, m2 = masses, parallax = parallax, f = 0)

def predict_astrometry_planets(ra, dec, parallax, pmra, pmdec, m_star, planets, phot_g_mean_mag, data_release, c_funcs, jacobi=False, reject_10_percent=True, rng=None, return_components=False):
    '''
    This function predicts the epoch-level astrometry of a star hosting any number of planets, as it would be
    observed by Gaia. The star's reflex motion is the sum of one Keplerian orbit per planet (planet-planet
    interactions are neglected), and the planets are assumed to contribute no light.
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
    a0_mas = get_planet_a0_mas(m_star = m_star, planets = planets, parallax = parallax, jacobi = jacobi)

    components = np.zeros((len(planets), len(t_ast_yr)))
    for i, planet in enumerate(planets):
        A, B, F, G = get_thiele_innes(a0_mas = a0_mas[i], Omega = planet['Omega'], inc = planet['inc'], w = planet['w'])
        components[i] = keplerian_al_signal(t_ast_yr = t_ast_yr, psi = psi, period = planet['period'], Tp = planet['Tp'], ecc = planet['ecc'], A = A, B = B, F = F, G = G, c_funcs = c_funcs)

    Lambda_pred = pmra*t_ast_yr*np.sin(psi) + pmdec*t_ast_yr*np.cos(psi) + parallax*plx_factor # barycenter motion
    for component in components:
        Lambda_pred = Lambda_pred + component # reflex motion due to each planet

    ast_obs, ast_err = add_epoch_noise(ast_true = Lambda_pred, phot_g_mean_mag = phot_g_mean_mag, rng = rng)

    if return_components:
        return t_ast_yr, psi, plx_factor, ast_obs, ast_err, components
    return t_ast_yr, psi, plx_factor, ast_obs, ast_err
