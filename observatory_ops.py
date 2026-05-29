import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

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
