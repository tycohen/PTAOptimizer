import numpy as np
from . import gravitational_waves as gw

class Pulsar(object):
    """
    Class to store an individual 15-yr pulsar

    Attributes:
    __________
    name : string
            pulsar name
    period : float
                pulse period in seconds
    dm : float
            dispersion measure in pc cm^-3
    dec : float
            declination in degrees
    ra : float
            right ascension in degrees
    dtd : float
            scintillation timescale in seconds
    dnud : float
            scintillation bandwidth in GHz
    taud : float
            scattering timescale in us
    dist : float
            DM distance in kpc
    w50 : float
            FWHM of L-band pulse profile in us
    weff : float
            Effective width of the L-band profile in us
    uscale : float
                scaling factor to distribute intensity across
                pulse profile
    s_1000 : float
                flux density at 1 GHz in mJy
    spindex : float
                spectral index
    sig_j_single : float
                    single-pulse RMS jitter
    template : np.array
                    template used to measure profile parameters
    parfile : str
              Parfile contents
    sigmas : dict
                dictionary of dictionaries of RMS components
                for each instrument
    telescope_noise : dict
                dictionary of FrequencyOptimizer.TelescopeNoise objects
                for each instrument
    optimum : dict
              dictionary of optimized parameters for each instrument
    t_int : dict
            dictionary of integration times for each instrument
    redamp : float
            dimensionless strain pulsar red noise amplitude
    redgamma : float
            positive pulsar red noise PSD spectral index
    """
    def __init__(self,
                 name=None,
                 period=None,
                 dm=None,
                 dec=None,
                 ra=None,
                 dtd=None,
                 dnud=None,
                 taud=None,
                 dist=None,
                 w50=None,
                 weff=None,
                 uscale=None,
                 s_1000=None,
                 spindex=None,
                 sig_j_single=None,
                 parfile=None,
                 template=None,
                 sigmas=None,
                 telescope_noise=None,
                 optimum=None,
                 t_int=None,
                 redamp=None,
                 redgamma=None,
                 *args,
                 **kwargs):
        """
        ___init___ function for the Pulsar class
        """

        self.name = name
        self.period = period
        self.dm = dm
        self.dec = dec
        self.ra = ra
        self.dtd = dtd
        self.dnud = dnud
        self.taud = taud
        self.dist = dist
        self.w50 = w50
        self.weff = weff
        self.uscale = uscale
        self.s_1000 = s_1000
        self.spindex = spindex
        self.sig_j_single = sig_j_single
        self.parfile = parfile
        self.template = template
        self.sigmas = {} if sigmas is None else dict(sigmas)
        if telescope_noise is None:
            self.telescope_noise = {}
        else:
            self.telescope_noise = dict(telescope_noise)
        self.optimum = {} if optimum is None else dict(optimum)
        self.t_int = {} if t_int is None else dict(t_int)
        self.redamp = redamp
        self.redgamma = redgamma

    @property
    def redalpha(self):
        """
        Negative strain spectral index
        """
        if self.redamp is None or self.redgamma is None:
            return None
        if self.redamp > 0. and self.redgamma > 0.:
            _, alpha = gw.rednoise_psd2charstrain(self.redamp, self.redgamma)
        else:
            alpha = 0.
        return alpha
        
    def sigma_jitter(self, t_int):
        """Return intrinsic jitter noise (in us)
        for given integration time in seconds"""
        n_pulses = t_int / self.period
        return self.sig_j_single / np.sqrt(n_pulses)            

    def add_sigmas(self, instr_name, sigma_dict):
        """instr_name : str
                        name of timing instrument
        sigma_dict : dict
                        dictionary of RMS components"""
        keys = ['sigma_tot',
                'sigma_white',
                'sigma_dm',
                'sigma_tel',
                'sigma_rn']
        if not all([k in list(sigma_dict.keys()) for k in keys]):
            keystr = ", ".join(keys)
            raise KeyError("'sigma_dict' must contain the keys: {}".format(keystr))
        self.sigmas.update({instr_name: sigma_dict})

    def get_instr_keys(self):
        return [k for k in self.sigmas]

    def detected(self, only=None):
        """
        Is the pulsar is detected by every telescope?
        
        Parameters
        ----------
        only : None or list (optional)
                list of receiver keys to only check if not None

        Returns
        -------
        bool
        """
        if only is None:
            rcvrs = self.get_instr_keys()
        else:
            rcvrs = only
        return all([self.sigmas[r]["sigma_tot"] > 0 for r in rcvrs])
        
