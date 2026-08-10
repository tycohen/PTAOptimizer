import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize

def get_gains(scope, dec, g_zenith):
    """
    Compute elevation-dependent multiplicative receiver gain factor at a
    pulsar's meridian altitude

    Parameters:
    ----------
    scope : telescope.Telescope object
    dec : float
          pulsar declination in fractional degrees
    g_zenith : float or numpy.ndarray
          beam-center, zenith multiplicative receiver gain(s) in K/Jy

    Returns: gain (float or numpy.ndarray, depending on g_zenith type)
    """
    if abs(scope.lat - dec) >= 90. - scope.alt_lim[1]:
        # source never rises
        if isinstance(g_zenith, (list, np.ndarray)): 
            return np.zeros(len(g_zenith))
        elif isinstance(g_zenith, (int, float)):
            return 0.
    else:
        if scope.gainmodel is None:
            gain = g_zenith
        if scope.gainmodel == 'cos':
            gain = g_zenith * np.cos(np.radians(scope.lat - dec)) ** scope.gainexp
        if scope.gainmodel == 'exp':
            gain = g_zenith * np.exp(-((dec - scope.lat) ** 2) / (scope.lat ** 2.))
        return gain

def is_in_beam(ra_beam, dec_beam, ra_src, dec_src, beam_fwhm):
    """
    Returns true if a source with equitorial coords `(ra_src, dec_src)`
    is within `beam_fwhm` of the beam center (ra_beam, dec_beam)
    All variables in degrees
    """
    srcs_coord = SkyCoord(ra_src * u.deg, dec_src * u.deg, frame="icrs")
    beam_coord = SkyCoord(ra_beam * u.deg, dec_beam * u.deg, frame="icrs")
    d = beam_coord.separation(srcs_coord)
    if d.deg <= beam_fwhm:
        in_beam = True
    else:
        in_beam = False
    return in_beam

def uptime(dec, obs_lat, horiz=0., epoch_days=30.):
    """
    Simple calculation of time pulsar spends above horizon per epoch.

    Parameters:
    __________
    dec : float
        Declination of pulsar in decimal degrees
    obs_lat : float
        Observatory latitude in decimal degrees
    horiz : float
        Altitude of horizon in decimal degrees
    Returns:
    _______

    t_max : float 
          maximum source time above horizon in seconds
    """
    if obs_lat > 90. or obs_lat < -90.:
        raise ValueError("Observatory latitude must be between -90 and 90 deg.")
    
    dec = np.radians(dec)
    obs_lat = np.radians(obs_lat)
    horiz = np.radians(horiz)
    if abs(dec - obs_lat) >= np.pi / 2. - horiz:
        # source never rises
        return 0.
    elif abs(dec + obs_lat) >= np.pi / 2. + horiz:
        # source never sets
        return 86400. * epoch_days
    else:
        da = 2. * np.arccos((np.sin(horiz) / (np.cos(obs_lat) * np.cos(dec)))- np.tan(dec) * np.tan(obs_lat))
        return da * epoch_days * 86400. / (2. * np.pi)

def get_tobs(t0, scope, psr_dec, horiz=0., cutoff=1.08e5):
    """
    Very rough estimate of source time above the horizon

    Parameters:
    t0 : float or numpy.ndarray
         time above horizon in seconds for a Dec.=0 source
    scope : telescope.Telescope object
    psr_dec : float 
              pulsar Dec. in fractional degrees
    horiz : float
            lower elevation limit of telescope in fractional degrees
    cutoff : float
             maximum available time in seconds (default 30 hrs)
    """
    if abs(psr_dec - scope.lat) >= 90. - horiz:
        # source never rises
        if isinstance(t0, (list, np.ndarray)): 
            t_obs = np.zeros(len(t0))
        elif isinstance(t0, (int, float)):
            t_obs = 0.
    elif abs(psr_dec + scope.lat) >= 90. + horiz:
        # source never sets
        t_obs = t0 * (2 * np.cos(np.radians(psr_dec)) ** -1) ** scope.timefac
    else:
        t_obs = t0 * (np.cos(np.radians(psr_dec)) ** -1) ** scope.timefac
    return np.clip(t_obs, 0., cutoff)

def shared_optimal_beams(pta, beam_fwhm, instr_key=None):
    """
    Return only the most inclusive beams that contain multiple pulsars
    Brute force search

    Parameters:
    ----------
    pta : pta.PTA object
    beam_fwhm : float
          Full-width at half-maximum of the beam
    instr_key : str
          optional instrument key, excludes pulsars not visible to instrument

    Returns:
    -------
    psr_groups : list of lists of pulsar.Pulsar objects contained within each beam
    """
    import matplotlib.pyplot as plt
    if instr_key is None:
        psrlist = pta.psrlist
    else:
        psrlist = [p for p in pta.psrlist
                   if p.sigmas[instr_key]["sigma_tot"] > 0.]
    pta_psrs = np.array(psrlist)
    groups = []
    for i, thisp in enumerate(pta_psrs):
        thisp_group = [i]
        # start with beam centered on first src
        beam_ra = thisp.ra
        beam_dec = thisp.dec
        beam_width = beam_fwhm # start searching FWHM away from first src
        for j, otherp in enumerate(pta_psrs):
            if i == j:
                continue
            if is_in_beam(beam_ra, beam_dec,
                          otherp.ra, otherp.dec,
                          beam_width):
                thisp_group.append(j)
                # center the beam btwn all sources
                ra_list = [p.ra for p in pta_psrs[thisp_group]]
                dec_list = [p.dec for p in pta_psrs[thisp_group]]
                beam_ra, beam_dec = find_beam_center(ra_list, dec_list)
                beam_width = beam_fwhm / 2. # now only search beam radius from ctr
        thisp_set = set(thisp_group)
        if len(thisp_set) == 1: #no groups, lonely pulsar
            continue
        for g in groups:
            if thisp_set.issubset(set(g)):
                break
        else:
            groups = [g for g in groups if not set(g).issubset(thisp_set)]
            groups.append(thisp_group)
    psr_groups = [pta_psrs[list(idx)] for idx in groups]
    return psr_groups

def find_beam_center(ra_list, dec_list):
    """
    Finds the RA, Dec that minimizes max angular separation 
    to all sources in beam

    Parameters:
    ----------
    ra_list: list or array of Right Ascension values in degrees
    dec_list: list or array of Declination values in degrees

    Returns:
    -------
    ra, dec of centered beam
    """
    # Convert input to SkyCoord
    coords = SkyCoord(ra=ra_list*u.deg, dec=dec_list*u.deg, frame='icrs')

    # Initial guess: mean of unit vectors projected back to sphere
    x = np.cos(coords.ra.radian) * np.cos(coords.dec.radian)
    y = np.sin(coords.ra.radian) * np.cos(coords.dec.radian)
    z = np.sin(coords.dec.radian)
    mean_vec = np.array([np.mean(x), np.mean(y), np.mean(z)])
    mean_vec /= np.linalg.norm(mean_vec)

    init_ra = np.degrees(np.arctan2(mean_vec[1], mean_vec[0])) % 360
    init_dec = np.degrees(np.arcsin(mean_vec[2]))
    initial_guess = [init_ra, init_dec]

    # Objective function: maximum angular separation to all points
    def max_separation(x):
        ra, dec = x
        if not (0 <= ra <= 360) or not (-90 <= dec <= 90):
            return 1e6  # Large penalty for invalid values
        trial_point = SkyCoord(ra=x[0]*u.deg, dec=x[1]*u.deg, frame='icrs')
        separations = trial_point.separation(coords)
        return np.max(separations.deg)

    # Run minimization
    result = minimize(
        max_separation,
        initial_guess,
#        bounds=[(0, 360), (-90, 90)],
        method='Powell'  # Good for non-smooth functions like max()
    )

    if not result.success:
        raise RuntimeError("Could not find beam center. "
                           "Minimization failed: " + result.message)

    optimal_ra, optimal_dec = result.x
    return optimal_ra, optimal_dec
