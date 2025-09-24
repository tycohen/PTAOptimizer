import os
# avoid BLAS oversubscription in each worker
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import pickle
import numpy as np
from os import path
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.coordinates import SkyCoord
import astropy.units as u
import frequencyoptimizer as fop
from PTAOptimizer.telescope import Telescope
import PTAOptimizer.observatory_ops as oops
from optimize import OptimizeFrequency


def calc_timing(pta,
                nus,
                rxspecfile=None,
                t_int=None,
                dec_lim=None,
                lat=None,
                gainmodel=None,
                gainexp=None,
                timefac=0.,
                optimize_freq=None,
                verbose=False,
                max_workers=None):
    if rxspecfile is None:
        raise ValueError('rxspecfile must be defined')
    if not isinstance(optimize_freq, (OptimizeFrequency, type(None))):
        raise TypeError("If set, 'optimize_freq' must be "
                        "None or optimize.OptimizeFrequency")

    if optimize_freq is not None and optimize_freq.ncpu > 1:
        raise ValueError("ncpu for FrequencyOptimizer must be 1"
                         " when parallelizing over pulsars")
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 2)

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for p in pta.psrlist:
            futures.append(ex.submit(
                time_single_pulsar, p, nus, rxspecfile, t_int, dec_lim, lat,
                gainmodel, gainexp, timefac, optimize_freq
            ))
        
        for fut in as_completed(futures):
            psrname, instr_name, sigma_tup, telnoise, optimum_dict = fut.result()
            p = pta.get_single_pulsar(psrname)
            p.add_sigmas(instr_name, sigma_tup)
            try:
                p.optimum.update(optimum_dict)
            except AttributeError:
                pass
            p.telescope_noise.update({instr_name : telnoise})
                
def time_single_pulsar(p, nus, rxspecfile, t_int, dec_lim, lat,
                       gainmodel=None, gainexp=None, timefac=0.,
                       optimize_freq=None):
    """
    Single pulsar TOA uncertainty calculator

    Returns
    -------
    pulsar name: str
    instrument name: str
    tuple of TOA noise components
    scope noise: frequencyoptimizer.TelescopeNoise
    optimum: dict of optimized observing parameters
    """
    scope = Telescope(name=path.splitext(path.basename(rxspecfile))[0],
                      dec_lim=dec_lim,
                      lat=lat,
                      gainmodel=gainmodel,
                      gainexp=gainexp)
    # per-pulsar integration time
    if t_int is not None:
        t_int_psr = float(t_int)
    else:
        # fall back to p.t_int[scope.name]
        if not hasattr(p, "t_int"):
            raise ValueError(
                "'t_int' is None and Pulsar {} has no attribute 't_int'. "
                "Either set p.t_int[{!r}] or provide a scalar t_int."
                .format(p.name, scope.name)
            )
        if not isinstance(p.t_int, dict):
            raise TypeError(
                "Pulsar {}.t_int must be a dict mapping instrument->seconds,"
                " got {}"
                .format(p.name, type(p.t_int))
            )
        try:
            t_int_psr = float(p.t_int[scope.name])
        except KeyError as e:
            raise KeyError(
                "Pulsar {} is missing key {!r} in 't_int' dict."
                .format(p.name, e.args[0])
            )

    if not np.isfinite(t_int_psr) or t_int_psr <= 0:
        raise ValueError(
            "Invalid t_int for {} with {}: {} (must be finite and > 0)"
            .format(p.name, scope.name, t_int_psr)
        )

    scope.timefac = timefac
    if not hasattr(p, "ra"):
        ra_str = p.name[1:3] + 'h' + p.name[3:5] + 'm' # get RA from Jname
        j2k_coords = SkyCoord(ra=ra_str, dec=p.dec*u.deg, frame='icrs')
    else:
        j2k_coords = SkyCoord(ra=p.ra*u.deg, dec=p.dec*u.deg, frame='icrs')
    # initial scope noise to get the rx specs
    scope_noise_init = fop.TelescopeNoise(1.,
                                          1.,
                                          T=t_int_psr,
                                          rxspecfile=rxspecfile)
    scope_noise_init.gain = oops.get_gains(scope,
                                           p.dec,
                                           scope_noise_init.get_gain(nus))
    if 0. in scope_noise_init.gain:
        # if any gains are zero, psr below at least 1 scopes horizon
        return p.name, scope.name, (-2, -2, -2, -2, -2), scope_noise_init, {}
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
        if optimize_freq is None:
            fop_inst = fop.FrequencyOptimizer(pulsar_noise,
                                              gal_noise,
                                              scope_noise,
                                              nchan=len(nus),
                                              numax=get_ctrfreq(nus),
                                              numin=get_ctrfreq(nus),
                                              vverbose=False)
            sigma_tup = fop_inst.calc_single(nus)
            return p.name, scope.name, sigma_tup, scope_noise, {}
        else: # optimize observing frequency within band
            fop_inst = fop.FrequencyOptimizer(pulsar_noise,
                                              gal_noise,
                                              scope_noise,
                                              nchan=len(nus),
                                              numax=max(nus + np.diff(nus)[0]),
                                              numin=min(nus),
                                              enforce_numax=True,
                                              verbose=False,
                                              nsteps=optimize_freq.nsteps,
                                              dnu=optimize_freq.dnu,
                                              log=optimize_freq.log_grid,
                                              levels=optimize_freq.levels,
                                              colors=optimize_freq.colors,
                                              lws=optimize_freq.lws,
                                              ncpu=optimize_freq.ncpu)
            # ensure full band is included in grid
            B_full = fop_inst.numax - fop_inst.numin
            C_full = fop_inst.numin + B_full / 2.
            fop_inst.Cs = np.unique(np.sort(np.append(fop_inst.Cs, C_full)))
            fop_inst.Bs = np.unique(np.sort(np.append(fop_inst.Bs, B_full)))
            # get optimum ctr freq, BW
            fop_inst.calc()
            ctr_opt, bw_opt = fop_inst.get_optimum()
            numin_opt = ctr_opt - bw_opt / 2.
            numax_opt = ctr_opt + bw_opt / 2.
            optimum = {scope.name + "_freqopt" : {"nu_min" : numin_opt,
                                                  "nu_max" : numax_opt}}
            # re-calculate sigmas in optimized band
            nus_opt = np.linspace(numin_opt,
                                  numax_opt,
                                  len(nus) + 1)[:-1]
            scope_noise_init_opt = fop.TelescopeNoise(1.,
                                                      1.,
                                                      T=t_int_psr,
                                                      rxspecfile=rxspecfile)
            scope_noise_init_opt.gain = oops.get_gains(scope,
                                        p.dec,
                                        scope_noise_init_opt.get_gain(nus_opt))
            if isinstance(timefac, np.ndarray):
                scope_noise_init_opt.T = get_tobs(
                    scope_noise_init_opt.get_T(nus_opt),
                    scope,
                    p.dec)
            else:
                scope_noise_init_opt.T = scope_noise_init_opt.get_T(nus_opt)
            pulsar_noise.sigma_Js = p.sigma_jitter(scope_noise_init_opt.T)
            scope_noise_opt = fop.TelescopeNoise(rx_nu=nus_opt,
                                gain=scope_noise_init_opt.gain,
                                T_rx=scope_noise_init_opt.get_T_rx(nus_opt),
                                epsilon=scope_noise_init_opt.get_epsilon(nus_opt),
                                T=scope_noise_init_opt.T)
            fop_inst_opt = fop.FrequencyOptimizer(pulsar_noise,
                                                  gal_noise,
                                                  scope_noise_opt,
                                                  nchan=len(nus_opt),
                                                  numax=max(nus_opt),
                                                  numin=min(nus_opt),
                                                  verbose=False)
            sigma_tup_opt = fop_inst_opt.calc_single(nus_opt)

            if optimize_freq.plot:
                plot_fname = "{}_{}.png".format(p.name,
                                                scope.name)
                fop_inst.plot(path.join(optimize_freq.plotdir,
                                        plot_fname),
                              doshow=False,
                              minimum="k*")

            return (p.name, scope.name + "_freqopt",
                    sigma_tup_opt, scope_noise_opt, optimum)

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
        pta = pickle.load(ptaf, encoding="latin1")
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
            pickle.dump(pta, ptaf)
    return pta
    
if __name__ == '__main__':
    """
    'Main' function; calculate sigmas for multiple telescope configs
    and writes out to .pta file. File is overwritten each time.
    """
    with open('NG15yr.pta', 'rb') as ptaf:
        pta = pickle.load(ptaf, encoding="latin1")
    print('Timing AO L-S')
    LbandSlo_nus = np.arange(1.44 - .618 / 2, 1.868, 0.011)
    Shi_nus = np.arange(2.227 - .354 / 2, 2.227 + .354 / 2, 0.01)[:-1]
    aoLS_nus = np.sort(np.concatenate([LbandSlo_nus, Shi_nus]))
    calc_timing(pta,
                aoLS_nus,
                rxspecfile="./rxspecs/AO_Lwide_Swide_logain.txt",
                dec_lim=(39., 0.),
                t_int=1800.,
                lat=18.44,
                gainmodel=None,
                gainexp=None)

    print('Timing AO 430-L')
    nus_ao430 = np.arange(.432 - .02 / 2, .432 + .02 / 2, 0.00125)[:-1]
    nus_aoL = np.arange(1.44 - .58 / 2, 1.44 + .58 / 2, 0.00125)[:-1]
    ao430L_nus = np.concatenate([nus_ao430, nus_aoL])
    calc_timing(pta,
                ao430L_nus,
                rxspecfile="./rxspecs/AO_430_Lwide_logain.txt",
                t_int=1800.,
                dec_lim=(39., 0.),
                lat=18.44,
                gainmodel=None,
                gainexp=None)
    
    print('Timing GB 800-1200')
    nus_gb800 = np.arange(.820 - .200 / 2, .820 + .200 / 2, 0.009)
    nus_gb1_2 = np.arange(1.510 - .800 / 2, 1.510 + .800 / 2, 0.009)[:-1]
    gbt80012_nus = np.concatenate([nus_gb800, nus_gb1_2])
    calc_timing(pta,
                gbt80012_nus,
                rxspecfile="./rxspecs/GBT_Rcvr_800-Rcvr_1_2_logain.txt",
                t_int=1800.,
                dec_lim=(90., -46.),
                lat=38.42,
                gainmodel=None,
                gainexp=None)

    print('Timing GBT-L + VLA-S')
    vlaS_nus = np.arange(3. - 2. / 2., 3. + 2. / 2., 0.009)
    gbL_vlaS_nus = np.concatenate([nus_gb1_2, vlaS_nus])
    calc_timing(pta,
                gbL_vlaS_nus,
                rxspecfile="./rxspecs/GBT_Rcvr_1_2_VLAS_logain.txt",
                t_int=1800.,
                dec_lim=(90., -46.),
                lat=38.42,
                gainmodel=None,
                gainexp=None)

    print('Timing CHIME + GBT-L')
    chime_nus = np.arange(0.6 - 0.4 / 2., 0.6 + 0.4 / 2., 0.009)[:-1]
    chime_gbtL_nus = np.concatenate([chime_nus, nus_gb1_2])
    chime_gbtL_gainexp = np.concatenate([np.full(len(chime_nus), 1.),
                                         np.full(len(nus_gb1_2), 0.)])
    chime_gbtL_timefac = np.concatenate([np.full(len(chime_nus), 1.),
                                         np.full(len(nus_gb1_2), 0.)])
    calc_timing(pta,
                chime_gbtL_nus,
                rxspecfile="./rxspecs/CHIME-GBTL_logain.txt",
                dec_lim=(90., -20.),
                lat=49.32,
                gainmodel='cos',
                gainexp=chime_gbtL_gainexp,
                timefac=chime_gbtL_timefac)

    print('Timing CHIME + UWBR')
    gbuwb_ctrfreq = 2.35 #GHz
    gbuwb_bw = 3.3 # GHz
    gbuwb_nus = np.arange(gbuwb_ctrfreq - gbuwb_bw / 2,
                        gbuwb_ctrfreq + gbuwb_bw / 2,
                        0.009)[:-1]
    chime_lt_uwbr_nus = chime_nus[chime_nus < gbuwb_nus[0]]
    chime_uwbr_nus = np.concatenate([chime_lt_uwbr_nus,
                                     gbuwb_nus])
    chime_uwbr_gainexp = np.concatenate([np.full(len(chime_lt_uwbr_nus), 1.),
                                         np.full(len(gbuwb_nus), 0.)])
    chime_uwbr_timefac = np.concatenate([np.full(len(chime_lt_uwbr_nus), 1.),
                                         np.full(len(gbuwb_nus), 0.)])
    calc_timing(pta,
                chime_uwbr_nus,
                rxspecfile="./rxspecs/CHIME-GBTUWBR.txt",
                dec_lim=(90., -20.),
                lat=49.32,
                gainmodel='cos',
                gainexp=chime_uwbr_gainexp,
                timefac=chime_uwbr_timefac)

    with open('parallel.pta', 'wb') as ptaf:
        pickle.dump(pta, ptaf)
