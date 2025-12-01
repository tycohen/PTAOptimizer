from os import path
from warnings import warn
from scipy.interpolate import PchipInterpolator
import numpy as np
import cma
import PTAOptimizer.observatory_ops as oops
from calc_timing import calc_timing
import gravitational_waves as gw
import copy

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

    def snr_grid_search_on_lut(self, verbose=False):
        """
        Perform a brute-force grid search for S/N from lookup-table sigmas
        Fills self.snr_grid_from_lut
        Computationally infeasible for PTAs of more than a few pulsars
        Doesn't handle non-timed pulsars
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
            self.snr_grid_from_lut[idx] = float(gw.gwb_snr(psrdict))
        return

    def _grid_search_best_indices(self):
        if self.snr_grid_from_lut is None:
            raise ValueError("snr_grid_from_lut is None\n"
                             "Run snr_grid_search_on_lut first")
        flat = np.nan_to_num(self.snr_grid_from_lut,
                             nan=-np.inf)
        return np.unravel_index(np.argmax(flat), flat.shape)

    def optimum_tint_grid(self):
        """
        Get optimum integration time vector from lookup table and grid of S/N
        """
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
            x0 = np.array(self.t_int0, dtype=float)
            if x0.shape[0] != N:
                raise ValueError("'t_int0' has wrong length: {}, expected {}"
                                 .format(x0.shape[0], N))
        else:
            # Start from roughly equal allocation of the total budget
            x0 = np.full(N, self.t_int_maxtot / N, dtype=float)

        # Clip x0 to per-pulsar box constraints
        lower_bounds = np.full(N, float(self.t_int_min), dtype=float)
        upper_bounds = self.t_int_max.astype(float)
        x0 = np.clip(x0, lower_bounds, upper_bounds)

        # Initialize PTA state and psrdict for hasasia
        # use interpolation if lookup table, otherwise call calc_timing
        if use_lut:
            # can't reset pta or it will clear LUT
            self._set_sigma_interp_lut(x0)
            instr_name_lut_depdt = self.instr_name + "_sigma_interp"
        else:
            # Full calculation: reset PTA and compute sigmas for x0
            self._reset_pta_inplace()
            self._set_t_int_vector(x0)
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
            "bounds": [lower_bounds, upper_bounds],
        }

        # Optional tuning parameters
        if popsize is not None:
            opts["popsize"] = popsize
        else:
            # Default heuristic: 4 + 3 * log(N)
            opts["popsize"] = int(4 + 3 * np.log(N))

        if start_diag:
            opts["CMA_diagonal"] = int(start_diag)
        if cma_stds is not None:
            opts["CMA_stds"] = np.asarray(cma_stds, dtype=float)
        if updatecovwait is not None:
            opts["updatecovwait"] = int(updatecovwait)

        def objective(t_vec):
            """
            Objective for CMA-ES: minimize -SNR + penalty
            where penalty enforces sum(t_i) ≈ t_int_maxtot.
            """
            t_vec = np.asarray(t_vec, dtype=float)
            if t_vec.shape[0] != N:
                return 1e9

            if not np.all(np.isfinite(t_vec)):
                return 1e9

            # Total time budget penalty
            # penalty = penalty_weight * ((sum(t) - t_int_maxtot)/t_int_maxtot)^2
            total_time = float(np.sum(t_vec))
            rel_err = (total_time - self.t_int_maxtot) / self.t_int_maxtot
            penalty = penalty_weight * (rel_err ** 2)

            try:
                if use_lut:
                    snr = self.evaluate_snr_from_lut(psrdict, t_vec)
                else:
                    snr = self.evaluate_snr(psrdict, t_vec)
            except Exception:
                # Any failure => very bad candidate
                return 1e9

            if not np.isfinite(snr):
                return 1e9

            # CMA-ES minimizes the objective
            return -float(snr) + penalty

        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0, float(sigma0), opts)

        # Optimization loop
        while not es.stop():
            X = es.ask()                    # list of candidate t_vecs
            fX = [objective(x) for x in X]  # their objective values
            es.tell(X, fX)                  # update CMA distribution
            if verbose:
                es.disp()

        # Best candidate found by CMA (in terms of the penalized objective)
        x_star = np.array(es.result.xbest, dtype=float)
        # Clip to safety (should already satisfy bounds due to CMA)
        x_star = np.clip(x_star, lower_bounds, upper_bounds)

        # Recompute SNR at x_star without penalty to get the true merit
        if use_lut:
            snr_star = self.evaluate_snr_from_lut(psrdict, x_star)
        else:
            snr_star = self.evaluate_snr(psrdict, x_star)

        # Check how well the total-time constraint is satisfied
        total_time_star = float(np.sum(x_star))
        rel_err_star = (total_time_star - self.t_int_maxtot) / self.t_int_maxtot
        if abs(rel_err_star) > 1e-2:
            warn("CMA solution deviates from time budget by {:.2%} "
                 "(sum(t) = {}, budget = {})."
                 .format(rel_err_star, total_time_star, self.t_int_maxtot))
        # save optimal times in optimum dict
        for x, p in zip(x_star, self.pta.psrlist):
            p.optimum.update({self.instr_name_opt: {}})
            p.optimum[self.instr_name_opt]["t_int"] = x
        return x_star, float(snr_star)
