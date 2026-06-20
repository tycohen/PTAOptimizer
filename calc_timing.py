import os
import pickle
import numpy as np
from os import path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from threadpoolctl import threadpool_limits
from astropy.coordinates import SkyCoord
import astropy.units as u
import frequencyoptimizer as fop
from telescope import Telescope
import observatory_ops as oops

def calc_timing(pta,
                nus,
                rxspecfile=None,
                scope_name=None,
                t_int=None,
                dec_lim=None,
                lat=None,
                gainmodel=None,
                gainexp=None,
                timefac=0.,
                optimize_freq=None,
                verbose=False,
                max_workers=1):
    if rxspecfile is None:
        raise ValueError('rxspecfile must be defined')
    if scope_name is None:
        scope_name = path.splitext(path.basename(rxspecfile))[0]
    if optimize_freq is not None:
        if isinstance(optimize_freq, bool):
            raise TypeError("If set, 'optimize_freq' must be "
                            "an optimize.OptimizeFrequency object, "
                            "not a boolean.")

    if optimize_freq is not None and optimize_freq.ncpu > 1 and max_workers > 1:
        raise ValueError("optimize_freq.ncpu and max_workers cannot both be "
                         "> 1. Do not parallelize over pulsars and frequency "
                         "optimization simulataneously.")
    safe_ncpu = max(1, (os.cpu_count() or 1) - 2)
    if max_workers > safe_ncpu:
        max_workers = safe_ncpu

    with threadpool_limits(1): # Cap BLAS/MKL threads
        if max_workers == 1: # dont spawn child processes, run in serial
            for p in pta.psrlist:
                psrname, instr_name, sigma_tup, telnoise, optimum_dict = time_single_pulsar(
                    p, nus, rxspecfile, scope_name, t_int, dec_lim, lat,
                    gainmodel, gainexp, timefac, optimize_freq, verbose
                )
                p.add_sigmas(instr_name, sigma_tup)
                try:
                    p.optimum.update(optimum_dict)
                except AttributeError:
                    pass
                p.telescope_noise.update({instr_name : telnoise})
            return

        ctx = mp.get_context("fork")
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers,
                                 mp_context=ctx) as ex:
            for p in pta.psrlist:
                futures.append(ex.submit(
                    time_single_pulsar, p, nus, rxspecfile, scope_name,
                    t_int, dec_lim,
                    lat, gainmodel, gainexp, timefac, optimize_freq
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
                
def time_single_pulsar(p, nus, rxspecfile, scope_name, t_int, dec_lim, lat,
                       gainmodel=None, gainexp=None, timefac=0.,
                       optimize_freq=None, vverbose=False):
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
    scope = Telescope(name=scope_name,
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
            scope_noise_init.T = oops.get_tobs(scope_noise_init.get_T(nus),
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
                                       glon=j2k_coords.galactic.l.degree,
                                       glat=j2k_coords.galactic.b.degree)
        gal_noise = fop.GalacticNoise()
        if optimize_freq is None:
            fop_inst = fop.FrequencyOptimizer(pulsar_noise,
                                              gal_noise,
                                              scope_noise,
                                              nchan=len(nus),
                                              numax=max(nus) + np.diff(nus)[0],
                                              numin=min(nus),
                                              vverbose=vverbose)
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
                                              min_bw=optimize_freq.min_bw,
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
            # interpolate gainexp and timefac flags to optimal nus array
            if isinstance(gainexp, np.ndarray):
                gainexp_opt = np.interp(nus_opt, nus, gainexp)
            else:
                gainexp_opt = gainexp
            scope_opt = Telescope(name=scope.name,
                                  dec_lim=scope.dec_lim,
                                  lat=scope.lat,
                                  gainmodel=scope.gainmodel,
                                  gainexp=gainexp_opt)
            if isinstance(timefac, np.ndarray):
                timefac_opt = np.interp(nus_opt, nus, timefac)
                scope_opt.timefac = timefac_opt                
            else:
                scope_opt.timefac = timefac
            scope_noise_init_opt.gain = oops.get_gains(scope_opt,
                                        p.dec,
                                        scope_noise_init_opt.get_gain(nus_opt))
            if isinstance(timefac, np.ndarray):
                scope_noise_init_opt.T = oops.get_tobs(
                    scope_noise_init_opt.get_T(nus_opt),
                    scope_opt,
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

