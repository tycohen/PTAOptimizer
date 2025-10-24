from os import path
from scipy.interpolate import PchipInterpolator
import numpy as np
import cma
import PTAOptimizer.observatory_ops as oops
from calc_timing import calc_timing
import gravitational_waves as gw

class OptimizeFrequency(object):
    """
    Class to store arguments to frequencyoptimizer.FrequencyOptimizer

    Attributes:
    ----------

    nsteps: int
            Number of steps in the grid to run when log=True
    dnu: float
         Delta nu grid spacing when log=False
    log_grid: bool
         Use a log-space grid of center frequencies and bandwidths
    frac_bw: bool
         Run in fractional bandwidth mode
    full_bandwidth: bool
         Enforce full bandwidth in calculations
    plot: bool
          Write optimizer grid plots
    plotdir: string
             Directory in which to write plots
    levels: numpy.ndarray
            Array of contour levels for plotting
    colors: list
            List of contour colors for plotting
    lws: list
         List of contour linewidths for plotting
    ncpu: int
          Number of cpus to use for parallel computing
    """
    def __init__(self,
                 nsteps=20,
                 dnu=None,
                 log_grid=True,
                 frac_bw=False,
                 full_bandwidth=False,
                 plot=False,
                 plotdir=".",
                 levels=None,
                 colors=None,
                 lws=None,
                 ncpu=1):
        """
        ___init___ function for the OptimizeFrequency class
        """

        self.nsteps = nsteps
        if not log_grid and dnu is None:
            raise ValueError("'dnu' must be set if log_grid = False")
        self.dnu = dnu
        self.log_grid = log_grid
        self.frac_bw = frac_bw
        self.full_bandwidth = full_bandwidth
        self.ncpu = ncpu
        self.plot = plot
        if isinstance(plotdir, str):
            if path.isdir(plotdir):
                self.plotdir = plotdir
            else:
                raise OSError("Directory '{}' does not exist.".format(plotdir))
        else:
            raise TypeError("'plotdir' must be a string")
        if isinstance(levels, (np.ndarray, type(None))):
            self.levels = levels
        else:
            raise TypeError("'levels' must be None or a numpy.ndarray")            
        if isinstance(colors, (type(None), list)):
            self.colors = colors
        else:
            raise TypeError("'colors' must be None or a list")
        if isinstance(lws, (type(None), list)):
            self.lws = lws
        else:
            raise TypeError("'lws' must be None or a list")

class OptimizeTime(object):
    """
    Class to optimize observing time parameter(s)

    Attributes:
    ----------

    """
    def __init__(self,
                 pta,
                 nus,
                 rxspecfile,
                 dec_lim=None,
                 lat=None,
                 t_int0=None,
                 t_int_min=60.,
                 t_int_maxtot=gw.SECS_PER_YEAR / 12.,
                 epoch_days=365.25 / 12,
                 timefac=0.,
                 gainmodel=None,
                 gainexp=None,
                 optimize_freq=None,
                 timespan_yr=None,
                 cadence=None,
                 n_gw_freq=400,
                 gwb_strainamp=2.4e-15,
                 gwb_spindex=-2/3.,
                 use_best_instr=False,
                 max_evals=2000,
                 max_workers=1):
        """
        ___init___ function for the OptimizeFrequency class
        """
        self.pta = pta
        self.nus = nus
        self.rxspecfile = rxspecfile
        self.dec_lim = dec_lim
        self.lat = lat
        self.scope_horizon = self.dec_lim[1] - self.lat + 90.
        self.epoch_days = epoch_days
        self.t_int_min = t_int_min
        self.t_int_max = np.array([oops.uptime(p.dec,
                                                 self.lat,
                                                 horiz=self.scope_horizon,
                                                 epoch_days=self.epoch_days)
                                   for p in self.pta.psrlist])
        if t_int0 is not None and not len(t_int0) == len(self.pta.psrlist):
            raise ValueError("'t_int0' must have same shape as pta.psrlist: "
                             "({},) not {}".format(len(self.pta.psrlist),
                                                   len(t_int0)))
        self.t_int0 = t_int0
        self.t_int_maxtot = t_int_maxtot
        self.timefac = timefac
        self.gainmodel = gainmodel
        self.gainexp = gainexp
        self.optimize_freq = optimize_freq
        self.instr_name = path.splitext(path.basename(rxspecfile))[0]
        if optimize_freq:
            self.optstr = "_freqopt"
        else:
            self.optstr = ""
        self.instr_name_opt = self.instr_name + self.optstr
        self.max_workers = max_workers
        self.max_evals = max_evals
        self.timespan_yr = timespan_yr
        self.cadence = cadence
        self.n_gw_freq = n_gw_freq
        if use_best_instr:
            raise NotImplementedError("Best instrument selection not currently "
                                      "supported.")
        else:
            self.use_best_instr = use_best_instr
        self.gwb_strainamp = gwb_strainamp
        self.gwb_spindex = gwb_spindex
        self.tint_grid_names = None
        
    def _make_tint_grid(self, n_levels, log=False):
        """
        Make an 'n_levels' x N pulsars grid of integration times
        """
        tgrid = []
        for tmax in self.t_int_max:
            if log:
                tgrid.append(np.logspace(np.log10(self.t_int_min),
                                         np.log10(tmax),
                                         num=n_levels,
                                         dtype=float))
            else:
                tgrid.append(np.linspace(self.t_int_min, tmax,
                                         num=n_levels, dtype=float))
        return np.array(tgrid)
            
    def set_tint_from_grid(self, n_levels, log=False):
        """
        Set Pulsar t_int dicts with keys "self.instr_name + _tinti"
        where i is from 0 to 'n_levels' based on a grid of integration times
        Resets Pulsar sigmas, telescope_noise, optimum and t_int dicts
        """
        self.tint_grid_names = []
        for p in self.pta.psrlist:
            if hasattr(p, "t_int"):
                p.t_int.clear()
        self._reset_pta_inplace()
        tgrid = self._make_tint_grid(n_levels, log=log)
        for i, t_vec in enumerate(tgrid.T):
            instr_name = self.instr_name + "_tint{}".format(i)
            self.tint_grid_names.append(instr_name)
            self._set_t_int_vector(t_vec, instr_name)

    def fill_tint_lookup_table(self):
        """
        Compute 'sigmas' dict for each integration time set using
        'set_tint_from_grid'
        """
        for instr in self.tint_grid_names:
            calc_timing(self.pta,
                        self.nus,
                        scope_name=instr,
                        rxspecfile=self.rxspecfile,
                        t_int=None,
                        dec_lim=self.dec_lim,
                        lat=self.lat,
                        gainmodel=self.gainmodel,
                        gainexp=self.gainexp,
                        timefac=self.timefac,
                        optimize_freq=self.optimize_freq,
                        verbose=False,
                        max_workers=self.max_workers)
        
    def _reset_pta_inplace(self):
        """
        Clear timing fields for next iteration
        """
        for p in self.pta.psrlist:
            try:
                p.sigmas.clear()
            except AttributeError:
                p.sigmas = {}
            try:
                p.telescope_noise.clear()
            except AttributeError:
                p.telescope_noise = {}
            try:
                p.optimum.clear()
            except AttributeError:
                p.optimum = {}

    def _set_t_int_vector(self, t_vec, instr_name=None):
        """
        Set per-pulsar integration times for instrument
        Uses instr_name attr if None supplied
        """
        if instr_name is None:
            instr_name = self.instr_name
        for ti, p in zip(t_vec, self.pta.psrlist):
            if not hasattr(p, "t_int") or not isinstance(p.t_int, dict):
                p.t_int = {}
            p.t_int[instr_name] = float(ti)

    def interp_sigma(self, pulsar, tint_find):
        """
        interpolate sigma_tot(tint) for a single pulsar at tint=tint_find
        """
        tint = [pulsar.t_int[k] for k in self.tint_grid_names]
        sigma = [pulsar.sigmas[k + self.optstr]["sigma_tot"]
                 for k in self.tint_grid_names]
        f = PchipInterpolator(tint, sigma)
        return f(tint_find)
        
    def _set_sigma_interp_lut(self, t_vec):
        """
        Set sigmas and t_int in instrument key named 
        self.instr_name + '_sigma_interp'
        for interpolated values from lookup table for each pulsar
        """
        for ti, p in zip(t_vec, self.pta.psrlist):
            interp_key = self.instr_name + "_sigma_interp"
            p.t_int[interp_key] = ti
            p.sigmas[interp_key] = {}
            p.sigmas[interp_key]["sigma_tot"] = self.interp_sigma(p, ti)

    def _project_to_feasible(self, x):
        """
        Enforce t_int_min ≤ x_i ≤ t_int_max[i] and sum(x) ≤ t_int_maxtot.
        Clip to box, then scale down proportionally if over budget.
        """
        y = np.clip(np.asarray(x, dtype=float),
                    self.t_int_min, self.t_int_max)
        s = float(np.sum(y))
        if s <= float(self.t_int_maxtot) + 1e-12:
            return y
        if s > 0.0:
            y *= (float(self.t_int_maxtot) / s)
        return np.minimum(y, self.t_int_max)

    def _feasible(self, x, tol=1e-10):
        x = np.asarray(x, dtype=float)
        return (x >= -tol).all() and (x <= self.t_int_max + tol).all() \
            and (x.sum() <= self.t_int_maxtot + tol) \
            and (x >= self.t_int_min - tol).all()

    def evaluate_snr(self, psrdict, t_vec):
        """
        Compute the GWB S/N for updated vector of integration times
        """
        self._reset_pta_inplace()
        self._set_t_int_vector(t_vec)

        calc_timing(self.pta,
                    self.nus,
                    rxspecfile=self.rxspecfile,
                    t_int=None,
                    dec_lim=self.dec_lim,
                    lat=self.lat,
                    gainmodel=self.gainmodel,
                    gainexp=self.gainexp,
                    timefac=self.timefac,
                    optimize_freq=self.optimize_freq,
                    verbose=False,
                    max_workers=self.max_workers)
        gw.update_noise_spectra_approx(psrdict, self.pta)
        return float(gw.gwb_snr(psrdict))

    def maximize_snr_with_cma(self,
                              sigma0=0.3,
                              seed=42,
                              popsize=None,
                              start_diag=0,
                              cma_stds=None,
                              updatecovwait=None,
                              verbose=False):
        """
        Wrapper for CMA to maximize GW SNR over per-pulsar, per-epoch integration
        times using CMA-ES.

        Parameters:
        __________
        sigma0: (float) initial standard deviation of free parameters
        seed: (int) seed for optimizer
        popsize: (float) the number of new proposed solutions per iteration,
when None, popsize = 4 + 3 * np.log(N)
        start_diag: (int) number of iterations with diagonal covariance matrix
        cma_stds: (list or numpy.ndarray) multipliers for sigma0 in each coordinate
        updatecovwait: number of iterations without distribution update
maxfevals        -> inf  #v maximum number of function evaluations

        """
        N = len(self.pta.psrlist)
        # Initial guess
        if self.t_int0 is None:
            # if no initial guess, start with even distribution of time
            x0 = self._project_to_feasible(np.full(N,
                                            (self.t_int_maxtot / N) - 1e-10))
        else:
            x0 = self._project_to_feasible(self.t_int0)


        self._reset_pta_inplace()
        self._set_t_int_vector(x0)

        calc_timing(self.pta,
                    self.nus,
                    rxspecfile=self.rxspecfile,
                    t_int=None,
                    dec_lim=self.dec_lim,
                    lat=self.lat,
                    gainmodel=self.gainmodel,
                    gainexp=self.gainexp,
                    timefac=self.timefac,
                    optimize_freq=self.optimize_freq,
                    verbose=False,
                    max_workers=self.max_workers)
        psrdict = gw.get_hasasia_psrs(self.pta,
                                      instr=self.instr_name_opt,
                                      timespan_yr=self.timespan_yr,
                                      cadence=self.cadence,
                                      n_freqs=self.n_gw_freq,
                                      use_best_instr=self.use_best_instr,
                                      gwb_strainamp=self.gwb_strainamp,
                                      gwb_spindex=self.gwb_spindex)

        # CMA setup (minimize -SNR)
        opts = {
            "seed": int(seed),
            "bounds": [np.full(N, float(self.t_int_min), dtype=float),
                       self.t_int_max.astype(float)],
            "verb_disp": int(verbose),
            "maxfevals": int(self.max_evals),
            # Optional tuning parameters
            "popsize": popsize,
            "CMA_diagonal": start_diag, 
            "CMA_stds": cma_stds,
            "updatecovwait": updatecovwait
        }
        if popsize is None:
            opts["popsize"] = 4 + 3 * np.log(N)
        else:
            opts["popsize"] = popsize
            
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        best_snr = -np.inf
        best_x = x0.copy()

        while not es.stop():
            X = es.ask()
            Z = []
            f_vals = []
            for x in X:
                z = self._project_to_feasible(x)
                Z.append(z)
                try:
                    snr = self.evaluate_snr(psrdict, z)
                    f = -snr
                except Exception:
                    # steer CMA away from failures
                    snr = -1e9
                    f = 1e9
                f_vals.append(float(f))
                if snr > best_snr:
                    best_snr = float(snr)
                    best_x = z.copy()
            es.tell(Z, f_vals)
            if verbose:
                es.disp()
        # store the result
        self.res = es.result  # (xbest, fbest, evals_best, evals, iterations, ...)

        # check that optimal within bounds
        x_star = self.res.xbest
        if not self._feasible(x_star):
            raise RuntimeError("CMA returned infeasible xbest; "
                               "check ask/tell wiring.")
        snr_star = float(-self.res.fbest)
        # Ensure we return the best seen feasible point
        if best_snr > snr_star:
            x_star, snr_star = best_x, best_snr
        for x, p in zip(x_star, self.pta.psrlist):
            p.optimum.update({self.instr_name_opt: {}})
            p.optimum[self.instr_name_opt]["t_int"] = x
        return x_star, snr_star

