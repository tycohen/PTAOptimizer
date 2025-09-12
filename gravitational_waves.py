import numpy as np
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import pickle

SECS_PER_YEAR = 365.25 * 24 * 3600.0

def get_hasasia_psrs(pta, instr, timespan_yr=None,
                     cadence=12, n_freqs=400, use_best_instr=False,
                     gwb_strainamp=2.4e-15, gwb_spindex=-2/3.,
                     return_sencurve=False):
    """
Build a dict of hasasia.sensitivity.Pulsar and 
hasasia.sensitivity.Spectrum objects from PTA object

Parameters:
__________

pta: pta.PTA object
instr: str or list
       Instrument or list of instruments. If use_best_instr=True,
       overrides and chooses instrument with lowest RMS
timespan_yr: float, int, or None
       data timespan in years
       if not None, overrides individual pulsar.timespan_yr
cadence: int or float
       cadence of observations (number/year), default is monthly
n_freqs: int
       number of frequency bins to use in power spectrum
gwb_strainamp: float
       dimensionless GWB strain amplitude, default is NG15 Bayesian 
       posterior GWB amplitude
gwb_spindex: float
       spectral index of dimensionless GWB strain spectrum
Returns:
_______
psrdict: dict containing list of hasasia.sensitivity.Pulsar objects,
a dict of hasasia.sensitivity.Spectrum objects for each pulsar,
GW frequencies, PTA cadence, GWB strain amplitude, and GWB spectral index
    """
    names = [p.name for p in pta.psrlist]
    # get sky positions for all of the pulsars
    ras = np.array([p.ra for p in pta.psrlist])
    decs = np.array([p.dec for p in pta.psrlist])
    phi = ras * np.pi / 180.
    theta = np.pi / 2 - decs * np.pi / 180.

    if timespan_yr is not None:
        timespans = np.full(len(pta.psrlist), timespan_yr)
    else:
        try:
            timespans = np.array([p.timespan_yr for p in pta.psrlist])
        except AttributeError as e:
            raise AttributeError("'Pulsar' attribute 'timespan_yr' must be set "
                                 "when 'timespan_yr' is None") from e
    if use_best_instr:
        instr = [t[1] for t in pta.sigma_best()]
    else:
        if isinstance(instr, str):
            instr = [instr] * len(pta.psrlist)

    sigma_tots = np.array([p.sigmas[i]["sigma_tot"]
                           for i, p in zip(instr, pta.psrlist)])
    try:
        rn_strainamps = np.array([p.rn_strainamp for p in pta.psrlist])
        rn_strainidxs = np.array([p.rn_strainindex for p in pta.psrlist])
    except AttributeError:
        rn_strainamps = None
        rn_strainidxs = None
    max_time_s = max(timespans) * SECS_PER_YEAR
    freqs = np.linspace(1 / max_time_s,
                        0.5 * cadence / SECS_PER_YEAR,
                        n_freqs)
    # build list of pulsars
    psrs = hsim.sim_pta(psr_names=names,
                        timespan=timespans,
                        cad=cadence,
                        sigma=sigma_tots * 1e-6,
                        phi=phi,
                        theta=theta,
                        A_rn=rn_strainamps,
                        alpha=rn_strainidxs,
                        A_gwb=gwb_strainamp,
                        alpha_gwb=-2/3,
                        freqs=freqs)
    spectra = {}
    for p in psrs:
        sp = hsen.Spectrum(p, freqs=freqs)
        sp.NcalInv
        spectra[p.name] = sp
    psrdict= {"psrs": {p.name: p for p in psrs},
              "freqs": freqs,
              "spectra": spectra,
              "cadence": cadence,
              "instruments": instr,
              "gwb_strainamp": gwb_strainamp,
              "gwb_spindex": gwb_spindex}
    return psrdict
    
def update_noise_spectra_approx(psrdict, pta_new):
    """
Use the approximation NcalInv = Tf/Pn(f) to update
in-place, each hasasia.sensitivity.Spectrum object in 'psrdict'
with the noise properties of the respective pulsars in 
'pta_new.psrlist'
    """
    for p, instr in zip(pta_new.psrlist, psrdict["instruments"]):
        old_sp = psrdict["spectra"][p.name]
        new_psd = build_pulsar_psd(old_sp, p, instr,
                                   psrdict["cadence"],
                                   psrdict["gwb_strainamp"],
                                   psrdict["gwb_spindex"])
        new_sp = hsen.Spectrum(psrdict["psrs"][p.name],
                               freqs=psrdict["freqs"])
        new_sp.update_NcalInv_with_approx(new_psd)
        psrdict["spectra"][p.name] = new_sp
    return

def gwb_snr(psrdict,
            return_sencurve=False):
    """
Compute the S/N of a GWB from list of 
hasasia.sensitivity.Spectrum objects

psrdict: dict
       dictionary containing the following keys/values:
    spectra: dict of hasasia.sensitivity.Spectrum objects
    freqs: list or array of GW frequencies to compute spectrum
    gwb_strainamp: dimensionless GWB strain amplitude
    gwb_spindex: spectral index of dimensionless GWB strain spectrum
return_sencurve: bool
       instead of returning just the S/N, return the 
       hasasia.sensitivity.GWBSensitivityCurve
Returns:
_______

snr: float, S/N of the GWB
OR
scurve: hasasia.sensitivity.GWBSensitivityCurve
    """
    scurve = hsen.GWBSensitivityCurve(list(psrdict["spectra"].values()))
    Sh = hsen.S_h(psrdict["gwb_strainamp"],
                  psrdict["gwb_spindex"],
                  psrdict["freqs"])
    snr = scurve.SNR(Sh)
    if return_sencurve:
        return scurve
    else:
        return snr

def build_pulsar_psd(sp, pulsar, instr, cad, gwb_amp, gwb_idx):
    """
    Compute new pulsar PSD from existing Spectrum object

    Parameters:
    __________

    sp: hasasia.sensitivity.Spectrum
    pulsar: pulsar.Pulsar object
    instr: (str) instrument which measured TOA uncertainty
    cad: (int, float) observing cadence in number/year
    gwb_amp: dimensionless GWB strain amplitude
    gwb_idx: spectral index of dimensionless GWB strain spectrum
    
    Returns:
    _______

    new_psd: (numpy.ndarray) new PSD with same length as 'sp'
    """
    try:
        rnamp = pulsar.rn_strainamp
        rnidx = pulsar.rn_strainindex
    except AttributeError:
        rnamp = 0
        rnidx = 0
    new_psd = sp.add_white_noise_power(pulsar.sigmas[instr]["sigma_tot"] * 1e-6,
                                       SECS_PER_YEAR / cad, vals=True) \
                + sp.add_red_noise_power(rnamp, rnidx, vals=True) \
                + sp.add_red_noise_power(gwb_amp, gwb_idx, vals=True)
    return new_psd
    
def rednoise_psd2charstrain(amp_red, gamma_red):
    """
    Convert red noise amplitude in us yr^1/2 and spectral index of
    timing residual PSD to dimensionless strain GWB amplitude
    and spectral index
    (See Arzoumanian et al., 2021: NG12 Search for GWB, Eq. 2) 
    """
    alpha = (gamma_red + 3.) / 2.
    amp_strain = amp_red * np.sqrt(12) * np.pi * 3.171e-14
    return amp_strain, alpha
