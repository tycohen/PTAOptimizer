from os import path
from warnings import warn
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, Bounds, LinearConstraint
import numpy as np
import cma
from . import observatory_ops as oops
from .calc_timing import calc_timing
from . import gravitational_waves as gw
from . import samplers as samp
import copy
import datetime

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
    min_bw: float
         Minimum bandwidth (GHz) to consider in frequency optimization
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
                 min_bw=None,
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
        self.min_bw = min_bw
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
    pta : pta.PTA instance
    nus : evenly-spaced numpy.ndarray of observing frequencies (GHz)
    rxspecfile : default name or path to receiver specifications file
    dec_lim : tuple of (max, min) telescope declination limits (deg)
    lat : (float) telescope latitude (deg)
    t_int0 : (numpy.ndarray) initial, per-pulsar integration times (sec)
    t_int_min : (float) minimum integration time (sec)
    t_int_maxtot : (float) total time budget per epoch (sec)
    epoch_days : (float) length of an epoch (days)
    timefac : (numpy.ndarray) len(nus) flags to turn on freq-dependent t_int
    gainmodel : (str) telescope elevation-dependent gain model ('cos', 'exp' )
    gainexp : (float or numpy.ndarray) exponent for 'cos' gain model
    optimize_freq : (optimize.OptimizeFrequency) freq optimization parameters
    timespan_yr : (float) PTA data duration (assumed common across pulsars)
    cadence : (int) number of observations per year
    n_gw_freq : (int) number of GW frequencies at which to estimate spectra
    gwb_strainamp : (float) GWB dimensionless strain amplitude (default NG15)
    gwb_spindex : (float) GWB dimensionless strain spectral index (default circular)
    use_best_instr : not implemented
    max_evals : (int) maximum evaluations for self.maximize_snr_with_cma
    max_workers : (int) maximum parallel processes when computing PTA sigmas
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
        ___init___ function for the OptimizeTime class
        """
        self.pta = pta
        self.nus = nus
        self.rxspecfile = rxspecfile
        self.dec_lim = dec_lim
        self.lat = lat
        self.scope_horizon = self.dec_lim[1] - self.lat + 90.
        self.epoch_days = epoch_days
        self.t_int_maxtot = t_int_maxtot        
        self.t_int_min = t_int_min
        uptimes = np.array([oops.uptime(p.dec,
                                        self.lat,
                                        horiz=self.scope_horizon,
                                        epoch_days=self.epoch_days)
                            for p in self.pta.psrlist])
        # clip per-pulsar max to total budget
        self.t_int_max = np.clip(uptimes, None, self.t_int_maxtot)
        if t_int0 is not None and not len(t_int0) == len(self.pta.psrlist):
            raise ValueError("'t_int0' must have same shape as pta.psrlist: "
                             "({},) not {}".format(len(self.pta.psrlist),
                                                   len(t_int0)))
        self.t_int0 = t_int0
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
        if isinstance(timespan_yr, (list, np.ndarray)):
            raise NotImplementedError("Per-pulsar dataspans not currently "
                                      "supported.")
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
        self.snr_grid_from_lut = None
        
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

    def sigma_interpolator(self, pulsar):
        """
        return PchipInterpolator of sigma_tot(tint) for a single pulsar
        """
        tint = [pulsar.t_int[k] for k in self.tint_grid_names]
        sigma = [pulsar.sigmas[k + self.optstr]["sigma_tot"]
                 for k in self.tint_grid_names]
        f = PchipInterpolator(tint, sigma)
        return f
            
    def interp_sigma(self, pulsar, tint_find):
        """
        evaluate interpolated sigma at tint_find
        """
        return self.sigma_interpolator(pulsar)(tint_find)
        
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
    
    def evaluate_snr_from_lut(self, psrdict, t_vec):
        """
        Compute the GWB S/N for updated vector of integration times
        using sigma lookup table instead of calc_timing
        """
        # can't reset pta or it will clear LUT
        self._set_sigma_interp_lut(t_vec)
        gw.update_noise_spectra_approx(psrdict, self.pta)
        return float(gw.gwb_snr(psrdict))
        
    def _lut_check(self):
        """
        Check if lookup table exists and is filled
        """
        if self.tint_grid_names is None:
            raise ValueError("use_lut is True but no tint_grid_names set\n"
                             "call set_tint_from_grid first")
        has_lut = all((k + self.optstr) in p.sigmas
                      and "sigma_tot" in p.sigmas[k + self.optstr]
                      for p in self.pta.psrlist
                      for k in self.tint_grid_names)
        if not has_lut:
            raise ValueError("use_lut is True but lookup table is empty\n"
                             "call fill_tint_lookup_table first")

    def snr_grid_search_on_lut(self,
                               snr_gwbamp=None,
                               snr_gwbidx=None,
                               verbose=False):
        """
        Perform a brute-force grid search for S/N from lookup-table sigmas
        Fills self.snr_grid_from_lut
        Computationally infeasible for PTAs of more than a few pulsars
        Doesn't handle non-timed pulsars

        Parameters:
        ----------
        snr_gwbamp: float
             GWB strain amplitude, if set overrides self.gwb_strainamp only
             when computing the GWB strain PSD, S_h
        snr_gwbidx: float
             GWB Spectral index, if set overrides self.gwb_spindex only
             when computing the GWB strain PSD, S_h
        """
        self._lut_check()
        if any([p.sigmas[k + self.optstr]["sigma_tot"] < 0.
                for k in self.tint_grid_names for p in self.pta.psrlist]):
            raise NotImplementedError("PTA contains an untimed pulsar, "
                                      "filtering of untimed pulsars not supported")
        n_psrs = len(self.pta.psrlist)
        instr_axes = [self.tint_grid_names] * n_psrs
        sigma_instr_list = [[k + self.optstr for k in self.tint_grid_names]]
        sigma_instr_axes = sigma_instr_list * n_psrs
        shape = tuple(len(ax) for ax in instr_axes)
        self.snr_grid_from_lut = np.empty(shape, dtype=float)

        # initiate psrdict at _tint0
        psrdict = gw.get_hasasia_psrs(self.pta,
                                      instr=sigma_instr_axes[0][0],
                                      timespan_yr=self.timespan_yr,
                                      cadence=self.cadence,
                                      n_freqs=self.n_gw_freq,
                                      use_best_instr=False,
                                      gwb_strainamp=self.gwb_strainamp,
                                      gwb_spindex=self.gwb_spindex)
        if snr_gwbamp is None:
            snr_gwbamp = psrdict["gwb_strainamp"]
            snr_gwbidx = psrdict["gwb_spindex"]

        # loop over _tinti instrument tuples
        for idx in np.ndindex(shape):
            tint_keys = [instr_axes[k][idx[k]] for k in range(len(instr_axes))]
            sigma_keys = [sigma_instr_axes[k][idx[k]]
                             for k in range(len(instr_axes))]
            t_vec = [p.t_int[i] for p, i in zip(self.pta.psrlist,
                                                tint_keys)]
            if not self._feasible(t_vec, tol=1e-8):
                self.snr_grid_from_lut[idx] = np.nan
                continue
            psrdict["instruments"] = sigma_keys
            gw.update_noise_spectra_approx(psrdict, self.pta)
            self.snr_grid_from_lut[idx] = float(gw.gwb_snr(
                psrdict,
                gwb_strainamp=snr_gwbamp,
                gwb_spindex=snr_gwbidx))
        return

    def _grid_search_best_indices(self):
        if self.snr_grid_from_lut is None:
            raise ValueError("snr_grid_from_lut is None\n"
                             "Run snr_grid_search_on_lut first")
        grid = self.snr_grid_from_lut
        nan_mask = np.isnan(grid)
        # NaNs to -inf, get max, then mutate back. Memory efficient, no copies
        grid[nan_mask] = -np.inf
        try:
            flat_best = np.argmax(grid)
        finally:
            grid[nan_mask] = np.nan
        return np.unravel_index(flat_best, grid.shape)

    def optimum_tint_grid(self):
        """
        Get optimum integration time vector from lookup table and grid of S/N
        """
        warn("It is unlikely for the grid resolution to be sufficient to "
             "determine the true optimum if the optimum lies on the budget "
             "boundary.")
        best_idx = self._grid_search_best_indices()
        best_tint = np.array([p.t_int[self.tint_grid_names[i]]
                              for p, i in zip(self.pta.psrlist, best_idx)])
        return best_tint

    def maximize_snr_with_cma(self,
                              sigma0=1.,
                              seed=42,
                              penalty_weight=1e4,
                              use_lut=False,
                              popsize=None,
                              start_diag=0,
                              cma_stds=None,
                              updatecovwait=None,
                              log10_t=False,
                              verbose=False):
        """
        Wrapper for CMA to maximize GW SNR over per-pulsar, per-epoch integration
        times using CMA-ES.

        Parameters:
        __________
        sigma0: (float) initial standard deviation of free parameters (in seconds)
        seed: (int) seed for optimizer
        use_lut: (bool) use lookup table/interpolation for calculating sigmas
        popsize: (float) number of new proposed solutions per iteration,
                 when None, popsize = 4 + 3 * np.log(N)
        start_diag: (int) number of iterations with diagonal covariance matrix
        cma_stds: (list or numpy.ndarray) multipliers for sigma0 in each
                  coordinate (same length as number of pulsars)
        updatecovwait: (int or None) number of iterations without distribution
                       update before covariance is adapted again
        log10_t : bool
            If True, CMA-ES optimizes in y = log10(t) space, optimal time vector
            still returned in seconds.

        Returns
        _______
        x_star : numpy.ndarray
            Best integration-time vector found (seconds per pulsar)
        snr_star : float
            Corresponding GWB S/N at x_star (recomputed without penalty)
        """
        if use_lut:
            self._lut_check()

        N = len(self.pta.psrlist)

        # Basic feasibility checks for the time budget
        min_total = N * float(self.t_int_min)
        max_total = float(np.sum(self.t_int_max))
        if self.t_int_maxtot < min_total:
            raise ValueError(
                "Total time budget t_int_maxtot={} is smaller than "
                "N * t_int_min={} (no feasible solution)."
                .format(self.t_int_maxtot, min_total)
            )
        if self.t_int_maxtot > max_total:
            warn("Total time budget t_int_maxtot={} exceeds sum(t_int_max)={}. "
                 "Constraint sum(t_i) = t_int_maxtot cannot be satisfied exactly."
                 .format(self.t_int_maxtot, max_total))

        # Initial guess x0 (in seconds)
        if self.t_int0 is not None:
            t0_lin = np.array(self.t_int0, dtype=float)
            if t0_lin.shape[0] != N:
                raise ValueError("'t_int0' has wrong length: {}, expected {}"
                                 .format(t0_lin.shape[0], N))
        else:

            t0_lin = np.full(N, self.t_int_maxtot / N, dtype=float)
        # Start from roughly equal allocation of the total budget
        # Clip initial times to per-pulsar box constraints
        t_lower = np.full(N, float(self.t_int_min), dtype=float)
        t_upper = self.t_int_max.astype(float)
        t0_lin = np.clip(t0_lin, t_lower, t_upper)

        if log10_t:
            # CMA variable is y = log10(t)
            x0 = np.log10(t0_lin)
            lower_bounds = np.log10(t_lower)
            upper_bounds = np.log10(t_upper)
        else:
            # CMA variable is t itself (seconds)
            x0 = t0_lin.copy()
            lower_bounds = t_lower
            upper_bounds = t_upper

        # Initialize PTA state and psrdict for hasasia
        # use interpolation if lookup table, otherwise call calc_timing
        if use_lut:
            # can't reset pta or it will clear LUT
            self._set_sigma_interp_lut(t0_lin)
            instr_name_lut_depdt = self.instr_name + "_sigma_interp"
        else:
            # Full calculation: reset PTA and compute sigmas for t0_lin
            self._reset_pta_inplace()
            self._set_t_int_vector(t0_lin)
            instr_name_lut_depdt = self.instr_name_opt
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
                                      instr=instr_name_lut_depdt,
                                      timespan_yr=self.timespan_yr,
                                      cadence=self.cadence,
                                      n_freqs=self.n_gw_freq,
                                      use_best_instr=self.use_best_instr,
                                      gwb_strainamp=self.gwb_strainamp,
                                      gwb_spindex=self.gwb_spindex)

        # CMA-ES setup
        opts = {
            "seed": int(seed),
            "verb_disp": int(bool(verbose)),
            "maxfevals": int(self.max_evals),
            "bounds": [lower_bounds, upper_bounds],  # in CMA variable space
        }

        # Optional tuning parameters
        if popsize is not None:
            opts["popsize"] = int(popsize)
        else:
            # Default heuristic: 4 + 3 * log(N)
            opts["popsize"] = int(4 + 3 * np.log(N))

        if start_diag:
            opts["CMA_diagonal"] = int(start_diag)
        if cma_stds is not None:
            opts["CMA_stds"] = np.asarray(cma_stds, dtype=float)
        if updatecovwait is not None:
            opts["updatecovwait"] = int(updatecovwait)

        def objective(x_cma):
            """
            Objective for CMA-ES: minimize -SNR + penalty
            where penalty enforces sum(t_i) ≈ t_int_maxtot.

            x_cma is either:
              - t_vec (seconds)           if log10_t == False
              - y_vec = log10(t_vec)      if log10_t == True
            """
            x_cma = np.asarray(x_cma, dtype=float)
            if x_cma.shape[0] != N:
                return 1e9
            if not np.all(np.isfinite(x_cma)):
                return 1e9

            # Map CMA variable -> linear times
            if log10_t:
                t_vec = 10.0 ** x_cma
            else:
                t_vec = x_cma

            # Total time budget penalty in linear space
            total_time = float(np.sum(t_vec))
            rel_err = (total_time - self.t_int_maxtot) / self.t_int_maxtot
            penalty = penalty_weight * (rel_err ** 2)

            try:
                if use_lut:
                    snr = self.evaluate_snr_from_lut(psrdict, t_vec)
                else:
                    snr = self.evaluate_snr(psrdict, t_vec)
            except NotImplementedError:
                return 1e9

            if not np.isfinite(snr):
                return 1e9

            return -float(snr) + penalty

        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0, float(sigma0), opts)

        while not es.stop():
            X = es.ask()                    # list of candidate t_vecs
            fX = [objective(x) for x in X]  # their objective values
            es.tell(X, fX)                  # update CMA distribution
            if verbose:
                es.disp()

        # Best candidate in CMA variable space
        x_best = np.array(es.result.xbest, dtype=float)

        # Map back to linear times
        if log10_t:
            t_star = 10.0 ** x_best
        else:
            t_star = x_best

        # Enforce box constraints in linear space for safety
        t_star = np.clip(t_star, t_lower, t_upper)

        # True SNR at t_star (no penalty)
        if use_lut:
            snr_star = self.evaluate_snr_from_lut(psrdict, t_star)
        else:
            snr_star = self.evaluate_snr(psrdict, t_star)

        # Check how well the total-time constraint is satisfied
        total_time_star = float(np.sum(t_star))
        rel_err_star = (total_time_star - self.t_int_maxtot) / self.t_int_maxtot
        if abs(rel_err_star) > 1e-2:
            warn("CMA solution deviates from time budget by {:.2%} "
                 "(sum(t) = {}, budget = {})."
                 .format(rel_err_star, total_time_star, self.t_int_maxtot))

        # save optimal times in optimum dict
        for ti, p in zip(t_star, self.pta.psrlist):
            p.optimum.update({self.instr_name_opt: {}})
            p.optimum[self.instr_name_opt]["t_int"] = float(ti)

        return t_star, float(snr_star)

    def project_to_budget_equality(self, x, tol=1e-8, max_iter=80):
        """
        Euclidean projection onto the feasible set

            C = { t : sum(t) = B,  tmin <= t <= tmax }.

        using the "capped simplex with lower bounds" projection:
            t_i = clip(x_i - λ, tmin, tmax_i)
        with λ chosen so that sum(t_i) = B.
        """
        x = np.asarray(x, dtype=float)
        N = x.size
        B = float(self.t_int_maxtot)
        lo = np.full(N, float(self.t_int_min))
        hi = np.asarray(self.t_int_max, dtype=float).reshape(N)

        if np.any(hi < lo):
            raise ValueError("Infeasible bounds: some t_int_max < t_int_min.")

        s_lo = float(lo.sum())
        s_hi = float(hi.sum())
        if not (s_lo - tol <= B <= s_hi + tol):
            raise ValueError(
                "Equality constraint infeasible: budget not in "
                f"[sum(tmin), sum(tmax)] = [{s_lo}, {s_hi}], got {B}."
            )

        # If clipping already satisfies the budget, return immediately
        t0 = np.clip(x, lo, hi)
        s0 = float(t0.sum())
        if abs(s0 - B) <= tol:
            return t0

        # g(λ) = sum(clip(x - λ, lo, hi)) - B, monotone decreasing in λ
        def g(lam):
            return float(np.clip(x - lam, lo, hi).sum() - B)

        # Bracket the root
        lam_lo = -1.0
        lam_hi = 1.0

        while g(lam_lo) < 0.0:
            lam_lo *= 2.0
        while g(lam_hi) > 0.0:
            lam_hi *= 2.0

        lam_mid = 0.0
        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            val = g(lam_mid)

            if abs(val) <= tol:
                break

            # g is decreasing in λ
            if val > 0.0:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

        t = np.clip(x - lam_mid, lo, hi)

        # Final numerical guard
        s = float(t.sum())
        if abs(s - B) > 10 * tol:
            raise RuntimeError("Projection did not converge: "
                               "|sum(t)-B|={} > {}. "
                               "Try increasing max_iter or "
                               "relaxing tol.".format(abs(s-B),
                                                      10*tol))
        return t

    def interp_sigmadot(self, pulsar, tint):
        """
        evaluate :math: `\dot{\sigma}_i(t)` for pulsar i at t=tint
        """
        sigdot_t_i = self.sigma_interpolator(pulsar).derivative()(tint)
        return sigdot_t_i

    def _wn_sigma_and_sigmadot_seconds(self, t_vec):
        """
        Return per-pulsar sigma(t) and sigmadot(t) in seconds
        for WN-only quadratic form using LUT interpolation

        t_vec in seconds
        """
        t_vec = np.asarray(t_vec, dtype=float)
        N = len(self.pta.psrlist)
        if t_vec.shape != (N,):
            raise ValueError(f"t_vec must have shape ({N},), got {t_vec.shape}")

        # sigma(t) from LUT+PCHIP (microseconds), then convert to seconds
        sigma_us = np.array([self.interp_sigma(p, ti)
                             for p, ti in zip(self.pta.psrlist, t_vec)],
                            dtype=float)
        # sigmadot(t) from us/s, then convert to s/s
        sigmadot_us_per_s = np.array([self.interp_sigmadot(p, ti)
                                      for p, ti in zip(self.pta.psrlist, t_vec)],
                                     dtype=float)
        sigma_s = sigma_us * 1e-6
        sigmadot_s_per_s = sigmadot_us_per_s * 1e-6

        if np.any(~np.isfinite(sigma_s)) or np.any(~np.isfinite(sigmadot_s_per_s)):
            raise ValueError("Non-finite sigma or sigmadot encountered "
                             "in interpolation.")
        if np.any(sigma_s <= 0.0):
            raise ValueError("Non-positive sigma encountered.\n"
                             "t_vec = {}\n"
                             "sigma = {}\nsigma_dot = {}".format(t_vec,
                                                                 sigma_s,
                                                                 sigmadot_s_per_s))
        return sigma_s, sigmadot_s_per_s

    def wn_objective_and_grad(self, t_vec, Qmat):
        """
        White-noise-only quadratic objective and analytic gradient:

            F(t) = rho^2(t) = p(t)^T Q p(t),   p_i(t_i) = sigma_i(t_i)^{-2}.

        Gradient:
            dF/dt_i = -4 * sigma_i(t_i)^(-3) * sigmadot_i(t_i) * (Q p(t))_i

        Parameters
        ----------
        t_vec : array-like, shape (N,)
            Integration times in seconds 
        Qmat : ndarray, shape (N, N)
            Noise-independent Q matrix (WN-only quadratic form).

        Returns
        -------
        F : float
            rho^2
        grad : ndarray, shape (N,)
            dF/dt (units: rho^2 per second)
        """
        t_vec = np.asarray(t_vec, dtype=float)
        sigma_s, sigmadot_s = self._wn_sigma_and_sigmadot_seconds(t_vec)

        p = sigma_s ** -2
        Qp = Qmat @ p
        F = float(p @ Qp)

        grad = -4.0 * (sigma_s ** -3) * sigmadot_s * Qp
        return F, grad

    def wn_objective_only(self, t_vec, Qmat):
        """Convenience: return F(t)=p^T Q p only."""
        F, _ = self.wn_objective_and_grad(t_vec, Qmat)
        return F

    def wn_grad_only(self, t_vec, Qmat):
        """Convenience: return grad F(t) only."""
        _, grad = self.wn_objective_and_grad(t_vec, Qmat)
        return grad
    
    def wn_rn_objective_and_grad(self, t_vec,  Q_fk, psrdict):
        """
        White-noise + red-noise quadratic objective

            :math: `F(t) \equiv \rho^2 = \sum_k a(f_k)^T {\bf Q}(f_k) a(f_k)`

        where :math: `a(f_k) = 1 / S_I(f_k)`
        and analytic gradient

            :math: `dF/dt_i = -8 \Delta t sigma_i(t_i) sigmadot_i(t_i) \sum_{k=1}^{N_f} \left[\left(Q_k a_k \right)_i \right]\frac{a_i(f_k, t)}{P_i(f_k, t)}`

        Parameters
        ----------
        t_vec : array-like, shape (N,)
            Integration times in seconds 
        Q_fk : ndarray, shape (N, N)
            Noise-independent, frequency-dependent Q(f_k) block matrices
        psrdict: dict
            dictionary containing the following keys/values:
                spectra: dict of hasasia.sensitivity.Spectrum objects
                freqs: list or array of GW frequencies to compute spectrum
                gwb_strainamp: dimensionless GWB strain amplitude
                gwb_spindex: spectral index of dimensionless GWB strain spectrum
        Returns
        -------
        F : float
            rho^2
        grad : ndarray, shape (N,)
            dF/dt (units: rho^2 per second)
        """
        t_vec = np.asarray(t_vec, dtype=float)
        # update psrdict["spectra"] with interpd sigmas using NcalInv approx
        self._set_sigma_interp_lut(t_vec)
        psds = gw.update_noise_spectra_approx(psrdict,
                                              self.pta,
                                              return_psds=True)

        sigma_s, sigmadot_s = self._wn_sigma_and_sigmadot_seconds(t_vec)
        a_fk = np.array([1 / s.S_I for s in psrdict["spectra"].values()]).T
        F = gw.gwb_snr2_quad(Q_fk, a_fk)

        tconst = -8. * (gw.SECS_PER_YEAR / self.cadence) * sigma_s * sigmadot_s
        P_fk = np.array(psds).T        
        Qk_ak = np.einsum("kij,kj->ki", Q_fk, a_fk)
        grad = tconst * np.sum(Qk_ak * (a_fk / P_fk), axis=0)
        return F, grad

    def wn_rn_objective_only(self, t_vec, Q_fk, psrdict):
        """
        Convenience: return F(t) = sum_k[a(f_k)^T Q(f_k) a(f_k)] only
        from wn_rn_objective_and_grad
        """
        F, _ = self.wn_rn_objective_and_grad(t_vec, Q_fk, psrdict)
        return F

    def wn_rn_grad_only(self, t_vec, Q_fk, psrdict):
        """Convenience: return grad F(t) only."""
        _, grad = self.wn_rn_objective_and_grad(t_vec, Q_fk, psrdict)
        return grad

    def wn_kkt_report(self, Qmat, t, eps_act=1e-6):
        """
        Return a dict of metric for white noise only
        solution stationarity
        """
        t = np.asarray(t, float)
        F, gradF = self.wn_objective_and_grad(t, Qmat)
        tmin = float(self.t_int_min)
        tmax = np.asarray(self.t_int_max, float)
        at_lo = t <= (tmin + eps_act)
        at_hi = t >= (tmax - eps_act)
        free  = ~(at_lo | at_hi)

        # lambda estimate: mean grad over free set
        lam = float(np.mean(gradF[free])) if np.any(free) else float(np.mean(gradF))

        # KKT gaps (should be ~0 or correct-signed)
        gap_free = np.zeros_like(t)
        gap_lo   = np.zeros_like(t)
        gap_hi   = np.zeros_like(t)
        gap_free[free] = gradF[free] - lam
        gap_lo[at_lo]  = gradF[at_lo] - lam          # should be <= 0 at lo-active
        gap_hi[at_hi]  = lam - gradF[at_hi]          # should be <= 0 at hi-active

        return {
            "F": F,
            "sum": float(np.sum(t)),
            "lam": lam,
            "t": t,
            "gradF": gradF,
            "free": free,
            "at_lo": at_lo,
            "at_hi": at_hi,
            "gap_free_inf": float(np.max(np.abs(gap_free[free])) if np.any(free) else 0.0),
            "gap_lo_maxpos": float(np.max(np.maximum(0.0, gap_lo[at_lo])) if np.any(at_lo) else 0.0),
            "gap_hi_maxpos": float(np.max(np.maximum(0.0, gap_hi[at_hi])) if np.any(at_hi) else 0.0),
            "idx_lo": np.where(at_lo)[0],
            "idx_hi": np.where(at_hi)[0],
        }

    def _budget_linear_constraint(self):
        """
        Linear equality constraint: sum(t) = B
        """
        N = len(self.pta.psrlist)
        B = float(self.t_int_maxtot)
        A = np.ones((1, N), dtype=float)
        return LinearConstraint(A, lb=np.array([B]), ub=np.array([B]))


    def _time_bounds(self):
        """
        Box bounds: tmin <= t_i <= tmax_i
        """
        N = len(self.pta.psrlist)
        lo = np.full(N, float(self.t_int_min), dtype=float)
        hi = np.asarray(self.t_int_max, dtype=float).reshape(N)
        return Bounds(lo, hi, keep_feasible=True)

    def _make_cached_fun_jac(self, objective_and_grad,
                             objective_args=(),
                             bound_penalty=1e12, obj_scale=1.0):
        """
        Return cached (fun, jac) for trust-constr on f(t) = -F(t)/obj_scale.
        """
        cache = {"x": None, "f": None, "g": None}

        lo = float(self.t_int_min)
        hi = np.asarray(self.t_int_max, dtype=float).reshape(len(self.pta.psrlist))

        obj_scale = float(obj_scale)
        if not np.isfinite(obj_scale) or obj_scale <= 0.0:
            raise ValueError(f"obj_scale must be positive finite, got {obj_scale}")

        def eval_fg(x):
            x = np.asarray(x, dtype=float)

            if cache["x"] is not None and np.array_equal(x, cache["x"]):
                return cache["f"], cache["g"]

            # DOMAIN GUARD: if out of bounds, return smooth quadratic penalty
            # v_i = amount below lo or above hi (>=0)
            v_lo = np.maximum(0.0, lo - x)
            v_hi = np.maximum(0.0, x - hi)
            v = v_lo + v_hi
            if np.any(v > 0.0):
                # minimize f, so positive penalty
                f = float(bound_penalty * np.dot(v, v))

                # gradient of penalty: 2*w*v * dv/dx
                # dv/dx = -1 where x<lo, +1 where x>hi, 0 otherwise
                dv_dx = np.zeros_like(x)
                dv_dx[x < lo] = -1.0
                dv_dx[x > hi] = +1.0
                g = (2.0 * float(bound_penalty) * v * dv_dx)

                cache["x"] = x.copy()
                cache["f"] = f
                cache["g"] = g
                return f, g

            # ---- SAFE: in-bounds, evaluate real objective/grad ----
            F, gradF = objective_and_grad(x, *objective_args)
            f = -float(F) / obj_scale
            g = -np.asarray(gradF, dtype=float) / obj_scale

            cache["x"] = x.copy()
            cache["f"] = f
            cache["g"] = g
            return f, g

        def fun(x):
            f, _ = eval_fg(x)
            return f

        def jac(x):
            _, g = eval_fg(x)
            return g

        return fun, jac


    def maximize_snr_trust_constr(
        self,
        noisemodel="wn",
        t0=None,
        maxiter=500,
        init_trustrad=1.0,
        init_constrpenalty=1.0,
        gtol=1e-6,
        xtol=1e-10,
        barrier_tol=1e-10,
        obj_scale="equal",
        verbose=0,
        return_history=False,
    ):
        """
        Maximize objective F(t) subject to:
            sum(t) = B
            tmin <= t <= tmax
        Supports:
            noisemodel='wn'   : WN-only quadratic form
            noisemodel='wnrn' : WN + RN quadratic form
        Uses scipy.optimize.minimize(method="trust-constr") on f(t) = -F(t).

        Parameters
        ----------
        t0 : optional initial guess (N,)
            If provided, will be projected to the feasible set {bounds + equality}.
        maxiter, gtol, xtol, barrier_tol : trust-constr tolerances
        obj_scale : str or float
            Constant factor by which to scale the objective visible to the
            optimizer. If set to 'equal', obj_scale is F(ti) where ti is a 
            vector of equal times that sum to the budget. If set to 'init' and
            t0 is also set, obj_scale = F(t0), otherwise defaults to 'equal'.
        verbose : int
            0 quiet, 1 basic, 2+ more output (scipy style)
        return_history : bool
            If True, log t, F, ||proj grad||_inf (approx) each callback.

        Returns
        -------
        t_star : ndarray (N,)
        F_star : float
        info : dict
        """
        self._lut_check()
        N = len(self.pta.psrlist)
        B = float(self.t_int_maxtot)

        # --- initial point ---
        if t0 is None:
            t0 = np.full(N, B / N, dtype=float)

        # Ensure feasibility (bounds + equality) using your existing projector
        t0 = self.project_to_budget_equality(t0)

        noisemodel = str(noisemodel).lower()
        if noisemodel == "wn":
            Qobj = self.calc_Qmat()
            psrdict = None
            objective_and_grad = self.wn_objective_and_grad
            objective_args = (Qobj,)
        elif noisemodel == "wnrn":
            Qobj, psrdict = self.calc_Qfk()
            # Ensure the spectra update uses the interpolated sigma key.
            interp_key = self.instr_name + "_sigma_interp"
            psrdict["instruments"] = [interp_key] * len(self.pta.psrlist)

            # Seed that interpolation key in the PTA before any objective call.
            self._set_sigma_interp_lut(t0)

            objective_and_grad = self.wn_rn_objective_and_grad
            objective_args = (Qobj, psrdict)
        else:
            raise ValueError("Unknown noisemodel '{}'. "
                             "Expected 'wn' or 'wnrn'.".format(noisemodel))
        
        if obj_scale == "equal":
            teq = self.project_to_budget_equality(np.full(N, B / N, dtype=float))
            F0, _ = objective_and_grad(teq, *objective_args)
            obj_scale = max(1.0, abs(float(F0)))
        elif obj_scale == "init":
            F0, _ = objective_and_grad(t0, *objective_args)
            obj_scale = max(1.0, abs(float(F0)))
        else:
            obj_scale = float(obj_scale)
            if not np.isfinite(obj_scale) or obj_scale <= 0.0:
                raise ValueError("obj_scale must be 'equal', 'init', or a "
                                 "positive finite float; got {}".format(obj_scale))
        init_constrpenalty_eff = float(init_constrpenalty) / float(obj_scale)
        # --- constraints + bounds ---
        lc = self._budget_linear_constraint()
        bnds = self._time_bounds()

        # --- objective + jac (cached) ---
        fun, jac = self._make_cached_fun_jac(objective_and_grad,
                                             objective_args=objective_args,
                                             bound_penalty=1e12,
                                             obj_scale=obj_scale)
        hist = [] if return_history else None

        def callback(x, state=None):
            if not return_history:
                return
            x = np.asarray(x, dtype=float)
            # Evaluate true (max) objective for logging
            F, gradF = objective_and_grad(x, *objective_args)

            # Approx “projected gradient” infinity norm for equality manifold
            g = np.asarray(gradF, float).copy()
            lam = float(np.mean(g))
            g = g - lam

            # bound push feasibility diagnostic
            lo = float(self.t_int_min)
            hi = np.asarray(self.t_int_max, float).reshape(N)
            at_lo = x <= (lo + 1e-12)
            at_hi = x >= (hi - 1e-12)
            g[at_lo & (g < 0.0)] = 0.0
            g[at_hi & (g > 0.0)] = 0.0

            hist.append({
                "t": x.copy(),
                "F": float(F),
                "sum": float(np.sum(x)),
                "sum_err": float(np.sum(x) - B),
                "gnorm_inf_proj_diag": float(np.max(np.abs(g))),
                "at_lo": at_lo,
                "at_hi": at_hi
            })

        res = minimize(
            fun=fun,
            x0=t0,
            jac=jac,
            method="trust-constr",
            bounds=bnds,
            constraints=[lc],
            options={
                "initial_tr_radius": init_trustrad,
                "initial_constr_penalty": init_constrpenalty_eff,
                # "factorization_method": "svd",    # robustness (slower)
                "maxiter": int(maxiter),
                "gtol": float(gtol),
                "xtol": float(xtol),
                "barrier_tol": float(barrier_tol),
                "verbose": int(verbose),
            },
            callback=callback if return_history else None,
        )

        t_star = np.asarray(res.x, dtype=float)
        F_star, gradF_star = objective_and_grad(t_star, *objective_args)

        info = {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "n_iter": int(getattr(res, "nit", -1)),
            "n_eval": int(getattr(res, "nfev", -1)),
            "t": t_star,
            "noisemodel": noisemodel,
            "gradF": np.asarray(gradF_star, dtype=float),
            "F": float(F_star),
            "sum": float(np.sum(t_star)),
            "sum_err": float(np.sum(t_star) - B),
            "history": hist,
            "raw_result": res,
            "obj_scale": float(obj_scale),
            "scaled_fun": float(res.fun),
            "optimality": float(getattr(res, "optimality", np.nan)),
            "constr_violation": float(getattr(res, "constr_violation", np.nan)),
            "tr_radius": float(getattr(res, "tr_radius", np.nan)),
            "constr_penalty": float(getattr(res, "constr_penalty", np.nan))
        }
        return t_star, float(F_star), info

    def calc_Qmat(self):
        """
        Returns:
        -------
        Qmat : numpy.ndarray
            The noise-independent (Npsr,Npsr) matrix for computing the white
        noise-only objective
        """
        psrdict = gw.get_hasasia_psrs(self.pta,
                                      self.tint_grid_names[0] + self.optstr,
                                      timespan_yr=self.timespan_yr,
                                      cadence=self.cadence,
                                      n_freqs=self.n_gw_freq,
                                      use_best_instr=False,
                                      gwb_strainamp=self.gwb_strainamp,
                                      gwb_spindex=self.gwb_spindex)
        return gw.build_Q_matrix(psrdict)

    def calc_Qfk(self):
        """
        Returns:
        -------
        Q_fk : numpy.ndarray
            The noise-independent (N_GWfreq, Npsr, Npsr) sub-matrices
        Q_fk of the block-diagonal \tilde{Q} for computing the white noise +
        red noise objective
        psrdict : dict
            Dictionary of pulsars, their spectra and GWB properties
        """
        psrdict = gw.get_hasasia_psrs(self.pta,
                                      self.tint_grid_names[0] + self.optstr,
                                      timespan_yr=self.timespan_yr,
                                      cadence=self.cadence,
                                      n_freqs=self.n_gw_freq,
                                      use_best_instr=False,
                                      gwb_strainamp=self.gwb_strainamp,
                                      gwb_spindex=self.gwb_spindex)
        return gw.build_tildeQ_blocks(psrdict), psrdict

    def random_multistart_optimizer(self,
                                    nsamp=100,
                                    maxiter=500,
                                    spiky_delta_t=3600.,
                                    init_trustrad=1e4,
                                    init_trustconstrpen=1e3,
                                    trustconstr_gtol=1e-8,
                                    trustcontrs_xtol=1e-8,
                                    verbose=False,
                                    vverbose=False,
                                    optimizer="trust-constr",
                                    sample_type="uniform",
                                    noisemodel="wn"):
        """
        An optimal solution stability diagnostic to test sensitivity
        to optimizer initial conditions.
        Run nsamp gradient optimizers (trust-constr)
        with randomly sampled initial time vectors either 'spiky'
        (allocated to one pulsar near it's upper bound) or 'uniform'
        on the feasible polytope that obey the budget. Supports
        white noise-only noise model 'wn' or white noise + red noise
        model 'wnrn'.

        Returns:
        -------
        t_opts : (nsamp, Npulsar) numpy.ndarray
                 optimal time vectors for each random start
        f_opts : (nsamp,) numpy.ndarray
                 objective optimum for each random start
        t0s_proj : (nsamp, Npulsar) numpy.ndarray
                 initial starting vector projected onto feasible polytope for
                 each random start
        """
        valid_samplers = ("spiky", "uniform")
        valid_optimizers = ("trust-constr")
        if optimizer.lower() not in valid_optimizers:
            raise ValueError("{} is not a valid optimizer. "
                             "Valid optimizers are {}".format(optimizer,
                                                              valid_optimizers))
        valid_noisemodels = ("wn", "wnrn")
        if noisemodel.lower() not in valid_noisemodels:
            raise ValueError("{} is not a valid noise model. "
                             "Valid noise models are {}".format(noisemodel,
                                                                valid_noisemodels))
        N = len(self.pta.psrlist)
        if sample_type == "spiky":
            t0s = samp.spiky_t0_vec(nsamp, N, self.t_int_max, self.t_int_min,
                                    self.t_int_maxtot, spiky_delta_t)            
        elif sample_type == "uniform":
            t0s = samp.hit_and_run_budget_sampler(
                    nsamp,
                    np.full(N, self.t_int_min),
                    self.t_int_max,
                    self.t_int_maxtot)
        else:
            raise ValueError("{} is not a valid sample type. "
                             "Valid sample types are {}".format(sample_type,
                                                                valid_samplers))
        t0s_proj = np.array([self.project_to_budget_equality(t) for t in t0s])
        f_opts = []
        t_opts = []
        for i, t0 in enumerate(t0s_proj):
            print("multistart iter {}/{}".format(i + 1, nsamp))
            if optimizer.lower() == "trust-constr":
                if vverbose:
                    vverbose = 2
                if noisemodel.lower() == "wn":
                    t_opt, f_opt, debug = self.maximize_snr_trust_constr(
                        noisemodel="wn",
                        t0=t0,
                        maxiter=maxiter,
                        init_trustrad=init_trustrad,
                        init_constrpenalty=init_trustconstrpen,
                        gtol=trustconstr_gtol,
                        xtol=trustcontrs_xtol,
                        verbose=vverbose,
                        return_history=True)
                if noisemodel.lower() == "wnrn":
                    t_opt, f_opt, debug = self.maximize_snr_trust_constr(
                        noisemodel="wnrn",
                        t0=t0,
                        maxiter=maxiter,
                        init_trustrad=init_trustrad,
                        init_constrpenalty=init_trustconstrpen,
                        gtol=trustconstr_gtol,
                        xtol=trustcontrs_xtol,
                        verbose=vverbose,
                        return_history=True)
            f_opts.append(f_opt)
            t_opts.append(t_opt)
            eps_kkt, kkt_info = kkt_residual_box_eq(debug["gradF"], t_opt,
                                                    self.t_int_min,
                                                    self.t_int_max,
                                                    self.t_int_maxtot,
                                                    scale=True, tol=1e-4)
            if (eps_kkt > 1e-3) and verbose:
                print("eps_kkt > 1e-3 for iter {}. "
                      "Inspect convergence.".format(i))
        return np.array(t_opts), np.array(f_opts), np.array(t0s_proj)

    def random_timeswap_perturbation(self, t_opt, nswaps=100,
                                     noisemodel="wn",
                                     F_opt=None, delta_t=3600.0,
                                     adaptive_delta=False,
                                     verbose=False, tol=1e-10):
        """
        Heuristic local-optimality check:
        move +delta_t from j -> i (i gains time, j loses time), keeping sum fixed.
        Returns: swap indices array ([incr, decr]), swap time vectors, delta F
        """
        N = len(self.pta.psrlist)

        if noisemodel == "wn":
            objective_and_grad = self.wn_objective_and_grad
            Qobj = self.calc_Qmat()        
            objective_args = (Qobj,)
        elif noisemodel == "wnrn":
            Qobj, psrdict = self.calc_Qfk()
            # Ensure the spectra update uses the interpolated sigma key.
            interp_key = self.instr_name + "_sigma_interp"
            psrdict["instruments"] = [interp_key] * len(self.pta.psrlist)

            # Seed that interpolation key in the PTA before any objective call.
            self._set_sigma_interp_lut(t_opt)

            objective_and_grad = self.wn_rn_objective_and_grad
            objective_args = (Qobj, psrdict)
        else:
            raise ValueError("Unknown noisemodel '{}'. "
                             "Expected 'wn' or 'wnrn'.".format(noisemodel))


        if F_opt is None:
            F_opt, _ = objective_and_grad(t_opt, *objective_args)

        swap_idxs = [np.random.choice(N, size=2, replace=False) for _ in range(nswaps)]
        F_swaps = np.full(nswaps, np.nan, dtype=float)
        t_swaps = np.empty((nswaps, N), dtype=float)

        tmin = float(self.t_int_min)
        tmax = np.asarray(self.t_int_max, dtype=float).reshape(N)

        for k, (i, j) in enumerate(swap_idxs):
            t = t_opt.copy()

            if adaptive_delta:
                max_dt = min(tmax[i] - t[i], t[j] - tmin)
                if max_dt <= 0:
                    t_swaps[k] = t
                    continue
                dt = min(delta_t, 0.99 * max_dt)
            else:
                dt = delta_t

            t[i] += dt
            t[j] -= dt
            t_swaps[k] = t

            # sum is preserved by construction; no projection
            if not self._feasible(t, tol=1e-8):
                F_swaps[k] = np.nan
                continue

            F, _ = objective_and_grad(t, *objective_args)
            F_swaps[k] = F

            if verbose and (F > F_opt):
                print("Swap {:.1f}s: {} -> {} gives higher F ({:.4f})"
                      " than F_opt ({:.4f})".format(dt, j, i, F, F_opt))

        return swap_idxs, t_swaps, (F_swaps - F_opt)

    
def kkt_residual_box_eq(gradF, tvec, t_mins, t_maxes, t_budget,
                        tol=1e-8, scale=True):
    """
    Bound-aware KKT residual for maximize F(t)=rho^2 subject to:
    t_mins[i] <= tvec[i] <= t_maxes[i],  sum(tvec)=t_budget.

    Parameters
    ----------
    gradF    : (n,) gradient of F wrt t at the point t
    tvec     : (n,) vector of integration times (seconds)
    t_mins   : (n,) lower bounds (seconds)
    t_maxes  : (n,) upper bounds (seconds)
    t_budget : scalar time budget (seconds)
    tol      : t_mins, t_maxes absolute boundary tolerance (seconds)
    scale    : (bool) return a normalized residual

    Returns
    -------
    eps_kkt : scalar residual (scaled if scale=True)
    info : dict with lambda, residual vector, feasibility, active sets
    """
    t = np.asarray(tvec, float)
    gradF = np.asarray(gradF, float)
    l = np.asarray(t_mins, float)
    u = np.asarray(t_maxes, float)

    at_lo = t <= (l + tol)
    at_hi = t >= (u - tol)
    interior = ~(at_lo | at_hi)

    # Estimate lambda
    if np.any(interior):
        lam = np.median(gradF[interior])
    else:
        # lambda should satisfy: gradF[lo] >= lam >= gradF[hi]
        lo_min = np.min(gradF[at_lo]) if np.any(at_lo) else np.inf
        hi_max = np.max(gradF[at_hi]) if np.any(at_hi) else -np.inf
        if hi_max <= lo_min:
            lam = 0.5 * (hi_max + lo_min)
        else:
            # No lambda can satisfy the bound KKT inequalities
            lam = np.mean(gradF)  # fallback; residual will be large

    # Build bound-aware violation residual r
    r = np.zeros_like(gradF)
    r[interior] = gradF[interior] - lam
    # at lower: want gradF - lam <= 0  -> violation is positive part
    r[at_lo] = np.maximum(0.0, gradF[at_lo] - lam)
    # at upper: want gradF - lam >= 0  -> violation is pos part of (lam - gradF)
    r[at_hi] = np.maximum(0.0, lam - gradF[at_hi])

    r_inf = np.max(np.abs(r)) if r.size else 0.0
    feas_eq = abs(np.sum(t) - t_budget)

    if scale:
        denom = max(1.0, np.max(np.abs(gradF)), abs(lam))
        eps_kkt = max(r_inf / denom, feas_eq / max(1.0, abs(t_budget)))
    else:
        eps_kkt = max(r_inf, feas_eq)

    info = dict(
        lambda_=lam,
        r=r,
        r_inf=r_inf,
        feas_eq=feas_eq,
        n_lo=int(np.sum(at_lo)),
        n_hi=int(np.sum(at_hi)),
        n_int=int(np.sum(interior)),
    )
    return eps_kkt, info
