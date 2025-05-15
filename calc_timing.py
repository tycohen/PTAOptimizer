import cPickle
import functools
import numpy as np
from os import path
from mock import patch, mock_open
from io import StringIO
from astropy.coordinates import SkyCoord
import astropy.units as u
import frequencyoptimizer as fop
from PTAOptimizer.telescope import Telescope
import PTAOptimizer.observatory_ops as oops


def calc_timing(pta,
                nus,
                rxspecfile=None,
                t_int=1800.,
                dec_lim=None,
                lat=None,
                gainmodel=None,
                gainexp=None,
                timefac=0.):
    if rxspecfile is None:
        raise ValueError('rxspecfile must be defined')
    for p in pta.psrlist:
        scope = Telescope(name=path.splitext(path.basename(rxspecfile))[0],
                          dec_lim=dec_lim,
                          lat=lat,
                          gainmodel=gainmodel,
                          gainexp=gainexp)
        scope.timefac = timefac
        ra_str = p.name[1:3] + 'h' + p.name[3:5] + 'm' # get RA from Jname
        j2k_coords = SkyCoord(ra=ra_str, dec=p.dec*u.deg, frame='icrs')
        # initial scope noise to get the rx specs
        scope_noise_init = fop.TelescopeNoise(1.,
                                              1.,
                                              T=t_int,
                                              rxspecfile=rxspecfile)
        scope_noise_init.gain = oops.get_gains(scope,
                                               p.dec,
                                               scope_noise_init.get_gain(nus))
        if 0. in scope_noise_init.gain:
            # if any gains are zero, psr below at least 1 scopes horizon
            p.add_sigmas(scope.name, (-2, -2, -2, -2, -2))
            continue
        else:
            if isinstance(timefac, np.ndarray):
                scope_noise_init.T = get_tobs(scope_noise_init.get_T(nus),
                                         scope,
                                         p.dec)
            else:
                scope_noise_init.T = scope_noise_init.get_T(nus)
            # if 0. in scope_noise_init.T:
            #     print('Zero in scope_noise_init.T : {}'.format(scope_noise_init.T))
            #     print('Gain = {}'.format(scope_noise_init.gain))

            # re-initialize telescope noise with dec-dependent gains, int time
            scope_noise = fop.TelescopeNoise(rx_nu=nus,
                                             gain=scope_noise_init.gain,
                                             T_rx=scope_noise_init.get_T_rx(nus),
                                             epsilon=scope_noise_init.get_epsilon(nus),
                                             T=scope_noise_init.T)
            p.telescope_noise.update({scope.name : scope_noise})
            pulsar_noise = fop.PulsarNoise('', 
                                           alpha=-1 * p.spindex,
                                           dtd=p.dtd,
                                           dnud=p.dnud,
                                           taud=p.taud,
                                           C1=1.16,
                                           I_0=p.s_1000,
                                           DM=p.dm,
                                           D=p.dist,
                                           tauvar=0.5 * p.taud,
                                           Weffs=p.weff,
                                           W50s=p.w50,
                                           Uscale=p.uscale,
                                           sigma_Js=p.sigma_jitter(scope_noise.T),
                                           glon=j2k_coords.galactic.b.degree,
                                           glat=j2k_coords.galactic.l.degree)
            gal_noise = fop.GalacticNoise()
            fop_inst = fop.FrequencyOptimizer(pulsar_noise,
                                              gal_noise,
                                              scope_noise,
                                              nchan=len(nus),
                                              numax=get_ctrfreq(nus),
                                              numin=get_ctrfreq(nus),
                                              vverbose=False)
            sigma_tup = fop_inst.calc_single(nus)
            p.add_sigmas(scope.name, sigma_tup)
    return

def get_tobs(t0, scope, psr_dec, horiz=0., cutoff=1.08e5):
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

def get_ctrfreq(nus):
    mid = float(len(nus)) / 2
    if len(nus) % 2 != 0:
        return nus[int(mid - .5)]
    else:
        return (nus[int(mid)] + nus[int(mid - 1)]) / 2.

def chime_only(write=False):
    """Returns a CHIME-only pta.PTA object"""
    with open('NG15yr.pta', 'rb') as ptaf:
        pta = cPickle.load(ptaf)
    for p in pta.psrlist:   # reset the sigma dicts to being empty
        p.sigmas = {}
    print('Timing CHIME')
    chime_nus = np.arange(0.6 - 0.4 / 2., 0.6 + 0.4 / 2., 0.005)[:-1]
    chime_gainexp = np.full(len(chime_nus), 1.)
    chime_timefac = np.full(len(chime_nus), 1.)
    calc_timing(pta,
                chime_nus,
                rxspecfile="./rxspecs/CHIME.txt",
                dec_lim=(90., -35.),
                lat=49.32,
                gainmodel='cos',
                gainexp=chime_gainexp,
                timefac=chime_timefac)
    if write:
        with open('NG15yr_CHIMEonly.pta', 'wb') as ptaf:
            cPickle.dump(pta, ptaf)
    return pta

# Decorator for patching FrequencyOptimizer.TelescopeNoise.get_rxspecs (mocks open)
def patch_open_in_get_rxspecs(target_file, fake_data):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            if getattr(self, 'rxspecfile', None) == target_file:
                with patch('frequencyoptimizer.open',
                           mock_open(read_data=fake_data),
                           create=True):
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        return wrapped
    return decorator                                                                 

# Decorator for patching FrequencyOptimizer.TelescopeNoise.__init__
# (mocks isfile and abspath)
def patch_telnoise_init(target_file):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            if kwargs.get('rxspecfile') == target_file:
                with patch('frequencyoptimizer.os.path.isfile',
                           return_value=True), \
                     patch('frequencyoptimizer.os.path.abspath',
                           lambda x: x):
                     return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        return wrapped
    return decorator   

if __name__ == '__main__':
    """
    'Main' function; calculate sigmas for multiple telescope configs
    and writes out to .pta file. File is overwritten each time.
    """
    with open('NG20yr-DSA.pta', 'rb') as ptaf:
        pta = cPickle.load(ptaf)

    dsa2k_nus = np.linspace(1.35 - 1.3 / 2, 1.35 + 1.3 / 2, 100 + 1)[:-1]
    t_int = 3600.
    print('Timing DSA1650 with Full Array for {} min/psr'.format(t_int / 60.))
    calc_timing(pta,
                dsa2k_nus,
                rxspecfile="rxspecs/DSA1650.txt",
                t_int=t_int,
                dec_lim=(90., -30.),
                lat=37.23,
                gainmodel=None,
                gainexp=None)
    
    rx_freq = pta.psrlist[0].telescope_noise["DSA1650"].rx_nu
    rx_eps = pta.psrlist[0].telescope_noise["DSA1650"].epsilon
    rx_Trx = pta.psrlist[0].telescope_noise["DSA1650"].T_rx
    rx_gain = pta.psrlist[0].telescope_noise["DSA1650"].gain
    sigma_tots = np.array([p.sigmas["DSA1650"]["sigma_tot"]
                           for p in pta.psrlist])
    tint_fac = 1
    n_improve = 1
    while n_improve > 0:
        rx_gain /= 2. # A / 2
        t_int *= 2. # 2 * T
        tint_fac *= 2
        instr_name = "DSA1650 A-div-{} {}*T".format(tint_fac,
                                                tint_fac)
        colstack = np.column_stack([rx_freq, rx_Trx, rx_gain, rx_eps])
        rxfile_buf = StringIO()
        np.savetxt(rxfile_buf,
                   colstack,
                   header="Freq	Trx	G	eps",
                   delimiter="\t",
                   fmt="%.6f")
        mock_rxspec_content = rxfile_buf.getvalue()
        # mock incrementally reduced gain rxspecfile input
        fop.TelescopeNoise.get_rxspecs = patch_open_in_get_rxspecs(instr_name,
                                                                   mock_rxspec_content)(
            fop.TelescopeNoise.get_rxspecs
        )
        fop.TelescopeNoise.__init__ = patch_telnoise_init(instr_name)(
            fop.TelescopeNoise.__init__
        )
        print('Timing {}'.format(instr_name))
        calc_timing(pta,
                    dsa2k_nus,
                    rxspecfile=instr_name,
                    t_int=t_int,
                    dec_lim=(90., -30.),
                    lat=37.23,
                    gainmodel=None,
                    gainexp=None)
        sigma_tots_new = np.array([p.sigmas[instr_name]["sigma_tot"]
                                   for p in pta.psrlist])
        n_improve = np.sum((sigma_tots - sigma_tots_new) > 0)
        print("{} psrs improved".format(n_improve))
        sigma_tots = sigma_tots_new
        
    # with open('NG20yr-DSA.pta', 'wb') as ptaf:
    #     cPickle.dump(pta, ptaf)
