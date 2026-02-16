from os import path
from warnings import warn
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
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

    def _feasible_gradient_equal_bounds(self, t, gradF, eps_act=1e-10):
        """
        Build a 'feasible' gradient for constraints:
            sum(t)=B,  tmin <= t <= tmax

        We:
          1) define free set as indices not near bounds,
          2) subtract mean over free set to enforce equality-tangent direction,
          3) zero components that would push active bounds outward (infeasible).

        Returns
        -------
        g : ndarray, shape (N,)
            gradient projected to feasible directions (ascent sense).
        free_mask : ndarray[bool]
            mask of free variables used for mean subtraction.
        lam_hat : float or None
            estimated lambda (mean grad over free set), None if no free vars.
        """
        t = np.asarray(t, float)
        gradF = np.asarray(gradF, float)

        N = t.size
        tmin = float(self.t_int_min)
        tmax = np.asarray(self.t_int_max, float).reshape(N)

        at_lo = t <= (tmin + eps_act)
        at_hi = t >= (tmax - eps_act)
        free = ~(at_lo | at_hi)

        g = gradF.copy()

        lam_hat = None
        if np.any(free):
            lam_hat = float(np.mean(g[free]))
            g[free] = g[free] - lam_hat
        else:
            # fully clamped: no movement possible (or infeasible geometry)
            g[:] = 0.0
            return g, free, lam_hat

        # Bound feasibility: if at lower bound, cannot decrease t -> if g<0, zero it
        g[at_lo & (g < 0.0)] = 0.0
        # if at upper bound, cannot increase t -> if g>0, zero it
        g[at_hi & (g > 0.0)] = 0.0

        # Ensure equality-tangent again on remaining free components (optional but stabilizing)
        # This matters if we zeroed some free components due to near-bound logic.
        free2 = (g != 0.0) & free
        if np.any(free2):
            g[free2] -= float(np.mean(g[free2]))

        return g, free, lam_hat

    def maximize_snr_projected_gradient(
        self,
        Qmat,
        t0=None,
        max_iter=200,
        gtol=1e-6,
        ftol=1e-12,
        eps_act=1e-10,
        alpha0=1.0,
        alpha_min=1e-6,
        alpha_max=100,
        use_bb=True,
        bb_variant=1,
        c1=1e-4,
        tau=0.5,
        max_ls=40,
        verbose=True,
        return_history=True,
    ):
        """
        Maximize F(t)=rho^2(t)=p(t)^T Q p(t) subject to:
            sum(t)=t_int_maxtot and t_int_min <= t <= t_int_max

        Uses projected gradient ascent with Armijo backtracking and hard projection
        via project_to_budget_equality().

        Returns
        -------
        t : ndarray
            feasible optimizer iterate
        F : float
            objective value at t (rho^2)
        info : dict
            diagnostics + (optional) iteration history
        """
        N = len(self.pta.psrlist)
        B = float(self.t_int_maxtot)
        tmin = float(self.t_int_min)
        tmax = np.asarray(self.t_int_max, float).reshape(N)

        # --- init t ---
        if t0 is None:
            # equal-time initial guess then project (handles bounds)
            t0 = np.full(N, B / N, dtype=float)
        t = self.project_to_budget_equality(t0)

        # Evaluate objective and grad
        F, gradF = self.wn_objective_and_grad(t, Qmat)
        g, free_mask, lam_hat = self._feasible_gradient_equal_bounds(t, gradF, eps_act=eps_act)

        #KKT: classify active sets
        at_lo = t <= (tmin + eps_act)
        at_hi = t >= (tmax - eps_act)
        free  = ~(at_lo | at_hi)

        #KKT: lambda estimate (fallback if no free vars)
        if lam_hat is None:
            # if fully clamped, nothing to do; treat as converged or break
            lam = float(np.mean(gradF))
        else:
            lam = float(lam_hat)

        #KKT: residuals for a maximization problem
        if np.any(free):
            r_free = float(np.max(np.abs(gradF[free] - lam)))
        else:
            r_free = 0.0
        if np.any(at_lo):            
            r_lo = float(np.max(np.maximum(0.0, gradF[at_lo] - lam)))
        else:
            r_lo = 0.0
        if np.any(at_hi):            
            r_hi = float(np.max(np.maximum(0.0, lam - gradF[at_hi])))
        else:
            r_hi = 0.0
        kkt_resid = max(r_free, r_lo, r_hi)
        
        #BB: initialize BB memory (previous iterate and feasible-gradient)
        n_free = int(np.sum(free_mask))
        t_prev = None
        gradF_prev = None          
        free_prev = None           
        g_prev = None # no longer used
        
        hist = []
        if return_history:
            sum_err = float(t.sum() - B)
            min_margin = float(np.min(t - tmin))
            max_margin = float(np.min(tmax - t))
            hist.append({
                "iter": 0,
                "F": F,
                "gradF": gradF.copy(),
                "gnorm_inf": float(np.max(np.abs(g))),
                "kkt_resid": kkt_resid,
                "n_free": int(np.sum(free_mask)),
                "n_lo": int(np.sum(t <= (tmin + eps_act))),
                "n_hi": int(np.sum(t >= (tmax - eps_act))),
                "alpha": np.nan,
                "ls_iters": 0,
                "lam_hat": lam_hat,
                "sum_err": sum_err,
                "min_margin": min_margin,
                "max_margin": max_margin
            })

        if verbose:
            print(f"[0] F={F:.6e}  ||g||_inf={np.max(np.abs(g)):.3e}  free={np.sum(free_mask)}")

        # --- main loop ---
        for k in range(1, max_iter + 1):
            #KKT: convergence test (instead of ||g||)
            if kkt_resid <= gtol:
                if verbose:
                    print(f"Converged: KKT_resid={kkt_resid:.3e} <= {gtol}")
                break

            # Ascent direction: use feasible gradient directly
            d = g

            # directional derivative proxy (should be positive for ascent)
            gTd = float(np.dot(g, d))
            if gTd <= 0.0:
                # fallback: something odd with projection/active-set; stop or reset
                if verbose:
                    print("Warning: non-ascent direction encountered (g·d<=0). Stopping.")
                break

            # line search with projection
            #BB: choose step size alpha using raw gradF curvature
            # on a stable-free subspace
            if use_bb and (t_prev is not None) and (gradF_prev is not None) \
               and (free_prev is not None):
                s_full = t - t_prev
                y_full = gradF - gradF_prev

                #BB: restrict to indices free at BOTH iterates
                # (avoid active-set / eps_act chatter)
                bb_mask = free_mask & free_prev

                s = s_full[bb_mask]
                y = y_full[bb_mask]

                sty = float(np.dot(s, y))
                if bb_variant == 2:
                    yty = float(np.dot(y, y))
                    denom = yty
                    num = -sty
                else:
                    sts = float(np.dot(s, s))
                    denom = -sty
                    num = sts

                if (not np.isfinite(denom)) or (abs(denom) < 1e-30) or \
                   (denom <= 0.0) or (not np.isfinite(num)):
                    alpha = float(alpha0)
                else:
                    alpha = float(num / denom)
            else:
                alpha = float(alpha0)
            # clip alpha
            alpha = min(alpha, alpha_max)
            alpha = max(alpha, alpha_min)
            F_old = F
            ls_iters = 0
            accepted = False
            for ls in range(max_ls):
                ls_iters = ls + 1
                t_trial = self.project_to_budget_equality(t + alpha * d)
                F_trial = self.wn_objective_only(t_trial, Qmat)

                #DIAG: unprojected trial and projection correction
                u = t + alpha * d                 # pre-projection trial
                s_raw = u - t                     # = alpha*d
                s_eff = t_trial - t               # realized step after projection
                proj_corr = t_trial - u           # what projection changed

                #DIAG: predicted directional derivatives (raw vs realized)
                pred_raw = float(np.dot(gradF, s_raw))
                pred_eff = float(np.dot(gradF, s_eff))
                
                #DIAG: relative size of projection effect
                raw_norm = float(np.linalg.norm(s_raw))
                corr_norm = float(np.linalg.norm(proj_corr))
                eff_norm = float(np.linalg.norm(s_eff))
                proj_ratio = (corr_norm / raw_norm) if raw_norm > 0 else np.nan

                # Armijo sufficient increase condition using the
                # *realized projected step*
                s_trial = t_trial - t
                if not np.any(s_trial):
                    # projection didn't move: no backtracking will help
                    accepted = False
                    break
                # Armijo using the realized projected step and the true gradient
                pred = float(np.dot(gradF, s_trial))
                if pred <= 0.0:
                    alpha *= tau
                    continue

                armijo_rhs = F_old + c1 * pred
                if F_trial >= armijo_rhs:
                    #BB: store previous accepted iterate before updating
                    t_prev = t.copy()
                    gradF_prev = gradF.copy()     
                    free_prev = free_mask.copy()  
                    g_prev = g.copy()
                    #DIAG: cache accepted-step diagnostics before overwriting t
                    t_before = t.copy()
                    u_acc = u.copy()                   # from DIAG section above
                    s_raw_acc = s_raw.copy()
                    s_eff_acc = s_eff.copy()
                    proj_corr_acc = proj_corr.copy()
                    pred_raw_acc = pred_raw
                    pred_eff_acc = pred_eff
                    proj_ratio_acc = proj_ratio
                    
                    # accept
                    t = t_trial
                    F, gradF = self.wn_objective_and_grad(t, Qmat)
                    g, free_mask, lam_hat = self._feasible_gradient_equal_bounds(t, gradF, eps_act=eps_act)

                    #KKT: classify active sets
                    at_lo = t <= (tmin + eps_act)
                    at_hi = t >= (tmax - eps_act)
                    free  = ~(at_lo | at_hi)

                    #KKT: lambda estimate (fallback if no free vars)
                    if lam_hat is None:
                        # if fully clamped, nothing to do; treat as converged or break
                        lam = float(np.mean(gradF))
                    else:
                        lam = float(lam_hat)

                    #KKT: residuals for a maximization problem
                    if np.any(free):
                        r_free = float(np.max(np.abs(gradF[free] - lam)))
                    else:
                        r_free = 0.0
                    if np.any(at_lo):            
                        r_lo = float(np.max(np.maximum(0.0, gradF[at_lo] - lam)))
                    else:
                        r_lo = 0.0
                    if np.any(at_hi):            
                        r_hi = float(np.max(np.maximum(0.0, lam - gradF[at_hi])))
                    else:
                        r_hi = 0.0
                    kkt_resid = max(r_free, r_lo, r_hi)
                    
                    # fix t_prev and g_prev for 1 iter if bounds hit
                    if int(np.sum(free_mask)) != n_free:
                        t_prev = None
                        gradF_prev = None      #BB:
                        free_prev = None       #BB:
                        g_prev = None          # optional / legacy
                        n_free = int(np.sum(free_mask))
                    accepted = True
                    break
                else:
                    alpha *= tau
            else:
                # line search failed
                if verbose:
                    print("Line search failed to find improvement. Stopping.")
                break
            if not accepted:
                if verbose:
                    print("Line search failed (no acceptable projected step)."
                          " Stopping.")
                break
            if return_history:
                sum_err = float(t.sum() - B)
                min_margin = float(np.min(t - tmin))
                max_margin = float(np.min(tmax - t))
                hist.append({
                    "iter": k,
                    "F": F,
                    "t": t.copy(),
                    "gradF": gradF.copy(),
                    "gnorm_inf": float(np.max(np.abs(g))),
                    "kkt_resid": kkt_resid,
                    "n_free": int(np.sum(free_mask)),
                    "n_lo": int(np.sum(t <= (tmin + eps_act))),
                    "n_hi": int(np.sum(t >= (tmax - eps_act))),
                    "alpha": alpha,
                    "ls_iters": ls_iters,
                    "lam_hat": lam_hat,
                    "sum_err": sum_err,
                    "min_margin": min_margin,
                    "max_margin": max_margin,
                    "t_before": t_before,              # optional
                    "u": u_acc,                        # optional
                    "s_raw": s_raw_acc,
                    "s_eff": s_eff_acc,
                    "proj_corr": proj_corr_acc,
                    "pred_raw": pred_raw_acc,
                    "pred_eff": pred_eff_acc,
                    "proj_ratio": proj_ratio_acc,
                })

            if verbose and (k % 5 == 0 or k == 1):
                dF = F - F_old
                print(f"[{k}] F={F:.6e}  dF={dF:.3e}  ||g||_inf={np.max(np.abs(g)):.3e}  alpha={alpha:.2e}  free={np.sum(free_mask)}")

            # objective stall criterion (optional)
            if abs(F - F_old) <= ftol * max(1.0, abs(F_old)):
                if verbose:
                    print(f"Stalled: |dF|={abs(F-F_old):.3e} <= ftol*scale")
                break

        info = {
            "n_iter": (hist[-1]["iter"] if return_history else k),
            "F": F,
            "t": t,
            "history": hist if return_history else None,
        }
        return t, F, info

    def dependent_pulsar_for_equality_reparam(self,
                                              Qmat,
                                              t_ref=None,
                                              eps_act=1e-10,
                                              eps_score=1e-30):
        """
        Choose dependent index k for equality-elimination reparam using
        score = slack / (eps + |gradF_i - median(gradF)|).

        Uses WN-only grad (wn_objective_and_grad). Later you can swap this
        to full-objective grad with the same interface.
        """
        N = len(self.pta.psrlist)
        B = float(self.t_int_maxtot)
        tmin = float(self.t_int_min)
        tmax = np.asarray(self.t_int_max, float).reshape(N)

        # Reference point
        if t_ref is None:
            t_ref = np.full(N, B / N, dtype=float)
        t_ref = self.project_to_budget_equality(t_ref)

        # Raw gradient at reference
        _, gradF = self.wn_objective_and_grad(t_ref, Qmat)
        gradF = np.asarray(gradF, dtype=float)

        # Slack at reference (distance to nearest bound)
        slack_lo = t_ref - tmin
        slack_hi = tmax - t_ref
        slack = np.minimum(slack_lo, slack_hi)

        # Avoid choosing a pulsar already (near) active by eps_act
        at_lo = t_ref <= (tmin + eps_act)
        at_hi = t_ref >= (tmax - eps_act)
        active = at_lo | at_hi

        # Robust center of gradients: median over non-active entries if possible
        if np.any(~active):
            g_med = float(np.median(gradF[~active]))
        else:
            g_med = float(np.median(gradF))

        # Score formula
        denom = eps_score + np.abs(gradF - g_med)
        score = slack / denom

        # Invalidate active / zero-slack / non-finite candidates
        bad = active | (slack <= 0.0) | (~np.isfinite(score))
        score = score.copy()
        score[bad] = -np.inf

        k = int(np.argmax(score))

        # If everything is invalid (should be rare), fall back to max slack
        if not np.isfinite(score[k]):
            slack2 = slack.copy()
            slack2[~np.isfinite(slack2)] = -np.inf
            k = int(np.argmax(slack2))

        return k

    def _reparam_split_indices(self, k):
        """Return list of free indices (all except k), in ascending order."""
        N = len(self.pta.psrlist)
        if k < 0 or k >= N:
            raise ValueError(f"k out of range: {k} for N={N}")
        idx_free = [i for i in range(N) if i != k]
        return idx_free

    def t_from_free_y(self, y, k):
        """
        Map reduced variable y (= t for all i != k) to full feasible candidate t
        by setting:
            t_i = y_i for i != k
            t_k = B - sum_{i != k} t_i
        Does NOT clip; caller decides how to handle infeasibility.
        """
        y = np.asarray(y, dtype=float)
        N = len(self.pta.psrlist)
        idx_free = self._reparam_split_indices(k)
        if y.shape != (len(idx_free),):
            raise ValueError(f"y must have shape ({len(idx_free)},), got {y.shape}")

        B = float(self.t_int_maxtot)

        t = np.empty(N, dtype=float)
        t[idx_free] = y
        t[k] = B - float(np.sum(y))
        return t

    def free_y_from_t(self, t, k):
        """Inverse map: drop component k."""
        t = np.asarray(t, dtype=float)
        N = len(self.pta.psrlist)
        if t.shape != (N,):
            raise ValueError(f"t must have shape ({N},), got {t.shape}")
        idx_free = self._reparam_split_indices(k)
        return t[idx_free].copy()

    def reparam_free_bounds(self, k):
        """
        Bounds for y variables in reduced problem.
        Since y_i are literally t_i for i != k, bounds are just box bounds:
            tmin <= y_i <= tmax_i
        The dependent feasibility (t_k bounds) is handled separately (penalty).
        """
        N = len(self.pta.psrlist)
        tmin = float(self.t_int_min)
        tmax = np.asarray(self.t_int_max, dtype=float).reshape(N)
        idx_free = self._reparam_split_indices(k)
        bounds = [(tmin, float(tmax[i])) for i in idx_free]
        return bounds

    def _dependent_bound_violation(self, t, k):
        """
        Return (v, side) where v is signed violation magnitude for t_k:
          - if t_k < tmin: v = tmin - t_k > 0   (lower violation)
          - if t_k > tmax_k: v = t_k - tmax_k > 0 (upper violation)
          - else: v = 0
        side in {"lo","hi",None}
        """
        tmin = float(self.t_int_min)
        tmax = float(np.asarray(self.t_int_max, dtype=float).reshape(-1)[k])
        tk = float(t[k])
        if tk < tmin:
            return (tmin - tk), "lo"
        if tk > tmax:
            return (tk - tmax), "hi"
        return 0.0, None

    def wn_objective_and_grad_reparam_y(
        self,
        y,
        Qmat,
        k,
        dep_penalty_weight=0.0,
    ):
        """
        WN-only objective/grad in reduced coordinates y = t_{i != k}.

        Full map:
            t = t(y), with t_k = B - sum(y)

        Gradient mapping:
            dF/dy_i = dF/dt_i - dF/dt_k

        Optional dependent-bound penalty:
            P = w * v(t_k)^2, where v is the (positive) bound violation magnitude.
        """
        y = np.asarray(y, dtype=float)
        idx_free = self._reparam_split_indices(k)

        # Build full t and evaluate base objective/gradient in t-space
        t = self.t_from_free_y(y, k)
        F, gradF_t = self.wn_objective_and_grad(t, Qmat)
        gradF_t = np.asarray(gradF_t, dtype=float)

        # Map gradient to y
        gk = float(gradF_t[k])
        grad_y = gradF_t[idx_free] - gk  # broadcast scalar gk

        # Optional penalty if dependent pulsar violates bounds
        if dep_penalty_weight and dep_penalty_weight > 0.0:
            v, side = self._dependent_bound_violation(t, k)
            if v > 0.0:
                w = float(dep_penalty_weight)
                F = float(F + w * (v ** 2))

                # dv/dt_k is:
                #   lo: v = tmin - t_k  => dv/dt_k = -1
                #   hi: v = t_k - tmax  => dv/dt_k = +1
                dv_dtk = -1.0 if side == "lo" else +1.0

                # dP/dt_k = 2 w v dv/dt_k
                dP_dtk = 2.0 * w * v * dv_dtk

                # Since t_k = B - sum(y), dt_k/dy_i = -1 for all i!=k
                # so dP/dy_i = dP/dt_k * dt_k/dy_i = - dP/dt_k
                grad_y = grad_y - dP_dtk

        return float(F), np.asarray(grad_y, dtype=float)

    def maximize_snr_reparam_lbfgsb(
        self,
        Qmat,
        t0=None,
        k=None,
        dep_penalty_weight=1e6,
        maxiter=500,
        gtol=1e-6,
        verbose=False,
        return_history=False,
    ):
        """
        Maximize WN-only objective with equality eliminated by choosing a
        dependent pulsar index k.

        Uses scipy.optimize.minimize with L-BFGS-B on y = t_{i!=k}.

        The equality sum(t)=B is enforced exactly via the reparam.
        Dependent bound feasibility t_k in [tmin, tmax_k] is enforced via
        a smooth quadratic penalty (weight dep_penalty_weight).
        """
        N = len(self.pta.psrlist)
        B = float(self.t_int_maxtot)

        # pick k if not given
        if k is None:
            # help the heuristic by using a feasible reference point
            if t0 is None:
                t_ref = np.full(N, B / N, dtype=float)
            else:
                t_ref = np.asarray(t0, dtype=float)
            t_ref = self.project_to_budget_equality(t_ref)
            k = self.dependent_pulsar_for_equality_reparam(Qmat, t_ref=t_ref)
        if verbose:
            print("Dependent index k:", k, "pulsar:", self.pta.psrlist[k].name)
        idx_free = self._reparam_split_indices(k)

        # initial feasible-ish t, then y0 = drop(k)
        if t0 is None:
            t0 = np.full(N, B / N, dtype=float)
        t0 = self.project_to_budget_equality(t0)
        y0 = self.free_y_from_t(t0, k)

        bounds = self.reparam_free_bounds(k)

        # optional callback history
        hist = [] if return_history else None

        def fun_and_jac(y):
            F, grad_y = self.wn_objective_and_grad_reparam_y(
                y, Qmat, k, dep_penalty_weight=dep_penalty_weight
            )
            # scipy minimizes; we want maximize => minimize -F
            return -F, -grad_y

        def callback(y):
            if not return_history:
                return
            t = self.t_from_free_y(y, k)
            F, grad_y = self.wn_objective_and_grad_reparam_y(
                y, Qmat, k, dep_penalty_weight=dep_penalty_weight
            )
            v, side = self._dependent_bound_violation(t, k)
            hist.append({
                "y": np.asarray(y, float).copy(),
                "t": np.asarray(t, float).copy(),
                "F": float(F),
                "dep_violation": float(v),
                "dep_violation_side": side,
                "gnorm_inf_y": float(np.max(np.abs(grad_y))),
            })

        res = minimize(
            fun=lambda y: fun_and_jac(y)[0],
            x0=y0,
            jac=lambda y: fun_and_jac(y)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "gtol": float(gtol),
                "disp": bool(verbose),
            },
            callback=callback if return_history else None,
        )

        y_star = np.asarray(res.x, dtype=float)
        t_star = self.t_from_free_y(y_star, k)

        # final safety: if dependent violated slightly, you’ll see it in diagnostics
        # (don’t silently clip here; clipping breaks equality).
        F_star, _ = self.wn_objective_and_grad(t_star, Qmat)

        info = {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "n_iter": int(getattr(res, "nit", -1)),
            "n_eval": int(getattr(res, "nfev", -1)),
            "k_dependent": int(k),
            "t": t_star,
            "F": float(F_star),
            "history": hist,
            "raw_result": res,
        }
        return t_star, float(F_star), info
    
    def wn_kkt_report(self, Qmat, t, eps_act=1e-6):
        """
        Return a dict of metric for white noise only
        solution stationarity
        """
        t = np.asarray(t, float)
        F, gradF = ot.wn_objective_and_grad(t, Qmat)
        tmin = float(ot.t_int_min)
        tmax = np.asarray(ot.t_int_max, float)
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
