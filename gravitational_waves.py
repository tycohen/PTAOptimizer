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
    gwb_psdamp, gwb_gamma = rednoise_charstrain2psd(gwb_strainamp,
                                                    gwb_spindex)
    
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
                        alpha_gwb=gwb_spindex,
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
              "gwb_spindex": gwb_spindex,
              "gwb_gamma": gwb_gamma}
    return psrdict
    
def update_noise_spectra_approx(psrdict, pta_new):
    """
Use the approximation NcalInv = Tf/Pn(f) to update
in-place, each hasasia.sensitivity.Spectrum object in 'psrdict'
with the noise properties of the respective pulsars in 
'pta_new.psrlist'
    """
    for p, instr in zip(pta_new.psrlist, psrdict["instruments"]):
        psd_new = build_pulsar_psd(psrdict["spectra"][p.name],
                                   p, instr,
                                   psrdict["cadence"],
                                   psrdict["gwb_strainamp"],
                                   psrdict["gwb_gamma"])
        psrdict["spectra"][p.name].update_NcalInv_with_approx(psd_new)
    return

def gwb_snr(psrdict,
            return_sencurve=False,
            gwb_strainamp=None,
            gwb_spindex=None):
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
gwb_strainamp: float
    if set, overrides value from psrdict when computing the GWB strain PSD
gwb_spindex: float
    if set, overrides value from psrdict when computing the GWB strain PSD
Returns:
_______

snr: float, S/N of the GWB
OR
scurve: hasasia.sensitivity.GWBSensitivityCurve
    """
    scurve = hsen.GWBSensitivityCurve(list(psrdict["spectra"].values()))
    if gwb_strainamp is None:
        gwb_strainamp = psrdict["gwb_strainamp"]
    if gwb_spindex is None:
        gwb_spindex = psrdict["gwb_spindex"]
    Sh = hsen.S_h(gwb_strainamp,
                  gwb_spindex,
                  psrdict["freqs"])
    snr = scurve.SNR(Sh)
    if return_sencurve:
        return scurve
    else:
        return snr

def build_pulsar_psd(sp, pulsar, instr, cad, gwb_amp, gwb_gamma):
    """
    Compute new pulsar PSD from existing Spectrum object

    Parameters:
    __________

    sp: hasasia.sensitivity.Spectrum
    pulsar: pulsar.Pulsar object
    instr: (str) instrument which measured TOA uncertainty
    cad: (int, float) observing cadence in number/year
    gwb_amp: dimensionless GWB strain amplitude
    gwb_gamma: (positive) PSD spectral index the GWB
    
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
    # hasasia convention is positive gamma
    new_psd = sp.add_white_noise_power(pulsar.sigmas[instr]["sigma_tot"] * 1e-6,
                                       SECS_PER_YEAR / cad, vals=True) \
                + sp.add_red_noise_power(rnamp, rnidx, vals=True) \
                + sp.add_red_noise_power(gwb_amp, gwb_gamma, vals=True)
    return new_psd

def build_W_matrix(psrdict):
    """
    Compute the noise-independent W matrix for white noise-only GWB S/N

    .. math::
    {\bf W}_{IJ} =     
        \begin{cases}
            \frac{1}{2}\frac{T_{IJ}}{T_{\rm obs}}\chi_{IJ}^2, & I \neq J\\
            0, & I=J
        \end{cases}
    """
    phis = np.array([p.phi for p in psrdict["psrs"].values()])
    thetas = np.array([p.theta for p in psrdict["psrs"].values()])
    ThetaIJ, chi_IJ, pairs, chiRSS = hsen.HellingsDownsCoeff(phis, thetas)
    T_obs = hsen.get_Tspan(list(psrdict["psrs"].values()))
    spectra = list((psrdict["spectra"].values()))
    T_IJ = np.array([hsen.get_TspanIJ(spectra[ii], spectra[jj])
                     for ii, jj in zip(pairs[0], pairs[1])])
    w_unraveled = .5 * T_IJ * (chi_IJ ** 2) / T_obs
    N = len(psrdict["psrs"])
    W = np.zeros((N, N))
    # symmetric, zero-diagonal
    W[pairs[0], pairs[1]] = w_unraveled
    W[pairs[1], pairs[0]] = w_unraveled
    return W

def build_Q_matrix(psrdict):
    """
    Compute the noise-independent Q matrix for white noise-only GWB S/N
    that can satisfy the quadratic form :math: `\rho^2 = p^T {\bf Q} p`

    .. math::
        
    {\bf Q} = 2T_\mathrm{obs}
    \left[\int^{f_\mathrm{Nyq}} df S_h(f)^2 
    \left(\frac{\mathcal{T(f)}\mathcal{R}(f)}{2\Delta t}\right)^2 \right]{\bf W}

    where p[i] = sigma[i] ** -2
    """
    T_obs = hsen.get_Tspan(list(psrdict["psrs"].values()))
    Tfs = [s.Tf for s in list(psrdict["spectra"].values())]
    Tf0 = Tfs[0]
    if not all([np.array_equal(Tf0, t) for t in Tfs[1:]]):
        raise ValueError("hasasia.sensitivity.Spectrum.Tf must be "
                         "the same for all spectra in psrdict")
    delta_t = SECS_PER_YEAR / psrdict["cadence"]
    Rf = hsen.resid_response(psrdict["freqs"])
    Sh = hsen.S_h(psrdict["gwb_strainamp"],
                  psrdict["gwb_spindex"],
                  psrdict["freqs"])
    integrand = (Sh * Tf0 * Rf / (2 * delta_t)) ** 2
    integral = np.trapz(y=integrand,
                        x=psrdict["freqs"],
                        axis=0)
    W = build_W_matrix(psrdict)
    return 2 * T_obs * integral * W

def build_tildeQ_blocks(psrdict):
    """
    Compute the sub-matrices of the block-diagonal, noise-independent 
    \tilde{Q} matrix for computing the white noise + red noise
    GWB S/N that can satisfy the quadratic form 
    :math: `\rho^2 = \tilde{p}^T {\bf \tilde{Q}} \tilde{p}`

    .. math::
    {\bf Q}_{IJ}(f_k) =    
    \begin{cases}
    2T_{\mathrm{obs}} \, w_k S_h^2(f_k)\, {\bf W}, & I \neq J, \\
    0, & I = J
    \end{cases}

    Returns a length N_GWfreqs array of N_pulsar x N_pulsar sub-matrices
    """
    N = len(psrdict["psrs"])
    n_freqs = len(psrdict["freqs"])
    freqdiff = np.diff(psrdict["freqs"])
    df = freqdiff[0]
    if not np.allclose(df, freqdiff):
        raise ValueError("psrdict['freqs'] is not evenly spaced. "
                         "build_tildeQ_matrix only valid for evenly spaced "
                         "frequencies.")
    trapz_freq_wts = df * np.concatenate([[0.5], np.ones(n_freqs - 2), [0.5]])
    T_obs = hsen.get_Tspan(list(psrdict["psrs"].values()))
    Sh = hsen.S_h(psrdict["gwb_strainamp"],
                  psrdict["gwb_spindex"],
                  psrdict["freqs"])
    W = build_W_matrix(psrdict)
    Q_fk = [2 * T_obs * w_k * (Sh_k ** 2) * W
            for w_k, Sh_k in zip(trapz_freq_wts, Sh)]
    return np.array(Q_fk)

def gwb_snr_quad(Q_fk, a_fk):
    """
    Compute the GWB S/N from the quadratic form
    :math: `\rho^2 = \sum_k a(f_k)^T {\bf Q}(f_k) a(f_k)`
    """
    return np.sum([ak @ (Qk @ ak) for ak, Qk in zip(a_fk, Q_fk)])
    
def rednoise_psd2charstrain(amp_red, gamma_red):
    """
    Convert red noise amplitude in us yr^1/2 and spectral index of
    timing residual PSD to dimensionless strain GWB amplitude
    and spectral index
    (See Arzoumanian et al., 2021: NG12 Search for GWB, Eq. 2) 
    """
    alpha = (3. - gamma_red) / 2.
    amp_strain = amp_red * np.sqrt(12) * np.pi * 3.171e-14
    return amp_strain, alpha

def rednoise_charstrain2psd(amp_strain, alpha):
    """
    Convert dimensionless strain GWB amplitude and spectral index 
    to red noise amplitude in us yr^1/2 and spectral index of
    timing residual PSD
    (See Arzoumanian et al., 2021: NG12 Search for GWB, Eq. 2) 
    """
    gamma_red = 3 - 2. * alpha
    amp_red = 3.154e13 * amp_strain / (np.sqrt(12) * np.pi)
    return amp_red, gamma_red
