
# PTAOptimizer

A python package for optimizing Pulsar Timing Array observations

---
---
## The Pulsar class (`pulsar.Pulsar`)

Class to store individual pulsar attributes, TOA uncertainties, and optimization results

### Attributes: 

* `name` : `str`

pulsar name
* `period` : `float`

pulse period in seconds
* `dm` : `float`

dispersion measure in pc cm^-3	   
* `dec` : `float`

declination in degrees
* `ra` : `float`

right ascension in degrees
* `dtd` : `float`

scintillation timescale in seconds
* `dnud` : `float`

scintillation bandwidth in GHz
* `taud` : `float`

scattering timescale in us
* `dist` : `float`

Earth-pulsar distance in kpc	    
* `w50` : `float`

full-width at half-maximum of pulse profile in us
* `weff` : `float`

Effective width of pulse profile in us
* `uscale` : `float`

scaling factor to distribute intensity across
                pulse profile
* `s_1000` : `float`

flux density at 1 GHz in mJy
* `spindex` : `float`

spectral index
* `sig_j_single` : `float`

single-pulse RMS jitter in us
* `template` : `numpy.ndarray` (optional)

template used to measure profile parameters
* `parfile` : str (optional)

parfile contents
* `sigmas` : `dict`

dictionary of dictionaries of RMS components
                for each instrument
* `telescope_noise` : `dict` (optional)

dictionary of FrequencyOptimizer.TelescopeNoise objects
                for each instrument
* `optimum` : `dict` (optional)

dictionary of optimized parameters for each instrument
* `t_int` : `dict` (optional)

dictionary of integration times for each instrument.
Use to set per-pulsar integration times. Will override integration time
arguments to `calc_timing.calc_timing` or receiver specification files.
* `redamp` : `float` (optional)

dimensionless strain pulsar red noise amplitude.
If not explicitly set, will be assumed to be zero.
* `redgamma` : `float` (optional)

positive pulsar red noise timing residual power spectrum
spectral index. If not explicitly set, will be assumed to be zero.

### Properties

* `redalpha(self)`

return negative strain spectral index converted from `redgamma`

### Methods

---
`sigma_jitter(self, t_int)`

Return intrinsic jitter noise (in us)
for given integration time in seconds

---
`add_sigmas(self, instr_name, sigma_dict)`

Add a dictionary of sigmas to self.sigmas under key `instr_name`

<ins>Parameters</ins>

`instr_name` : `str`

name of timing instrument

`sigma_dict` : `dict`

dictionary of RMS components containing (at least) keys
`'sigma_tot', 'sigma_white', 'sigma_dm', 'sigma_tel', 'sigma_rn'`

`get_instr_keys(self)`

returns instrument key names for sigmas dict

---
`detected(self, only=[])`

Is the pulsar detected with every instrument in `self.sigmas`?

<ins>Parameters</ins>

`only` : `list`

subset of instruments to query

---
---
## The PTA class (`pta.PTA`)


Class to store pulsar timing array attributes and constituent pulsars

### Attributes: 

* `name` : `str` (optional)

PTA name
* `psrlist` : list

list of `pulsar.Pulsar` objects


### Methods

---
`get_single_pulsar(self, psr_name)`

return `pulsar.Pulsar` object whose name matches `psr_name`

---
`sigma_best(self, exclude=[])`

Get the best instrument for each pulsar
and return list of tuples of (pulsar name, instrument, sigma_tot)

<ins>Parameters</ins>

`exclude` : list (optional, default = [])

list of telescope name substrings to exclude when sorting RMS's

---
`write_to_text(self, filename)`

Write total RMS for each pulsar at each instrument to file

---
`write_2best_to_markdown(self, filename, exclude=[])`

Write each pulsar's best two instrument options and total RMS for each to a markdown table

---
`make_deluxetable(self, exclude_names=[], save=True, savedir=".", split_row_idx=None,
longtable=False,fontsize=r"scriptsize",comments_macro="table caption",footnote_dict=None)`

Generate a publication-quality AASTex `deluxetable`

<ins>Parameters</ins>

`exclude_names` : `list`

exclude these pulsars from the table

`save` : `bool`

save to file with name `self.name + 'psr_params.tex'`

`savedir` : `string`

path to save directory

`split_row_idx` : `list` or `None`

optional list of indices to the left of which to split the table
vertically, creating `len(split_row_idx)` separate tables in the same file

`longtable` : `bool`

adds a `\startlongtable` before table environment

`fontsize` : `raw str`

latex named font size (no backslash)

`comments_macro` : `str`

optional custom latex macro for inserting content into `\tablecomments`

----
----
## `calc_timing` module

Time one or all pulsars with a particular instrument

---
`calc_timing(pta, nus, rxspecfile=None, scope_name=None, t_int=None, dec_lim=None, lat=None, gainmodel=None, gainexp=None, timefac=0., optimize_freq=None, verbose=False, max_workers=1):`

Compute TOA uncertainties for each pulsar in a whole PTA with a single instrument or instrument combination. Modifies `PTA` instance in-place, by updating `sigmas` dict for each pulsar.

<ins>Parameters</ins>

`pta` : `pta.PTA` instance

`nus` : `numpy.ndarray`

evenly-spaced, monotinically increasing array of observing frequency subbands (GHz) at which to calculate frequency-dependent timing effects

`rxspecfile` : `string`

path to a receiver specifications file containing a header
```
#freq	Trx	G	eps
```
and four tab-separated columns of observing frequencies (GHz), receiver temperature (K), multiplicative gain (K / Jy), and fractional polarization-calibration gain error

`scope_name` : `string` or `None`

optional telescope name used as instrument key. If `None`, uses basename of `rxspecfile`.

`t_int` : `float` or `None`

PTA-wide integration time (s) per subband. If `None`, each `pulsar` in `pta.psrlist`
must have the `t_int` attr set to a dictionary containing a key that matches `rxspecfile` or `scope_name`.

`dec_lim` : `tuple`

tuple of telescope declination limits (degrees) in order `(max, min)`

`lat` : `float`

telescope latitude (degrees)

`gainmodel` : `string` or `None`

optional elevation-dependent telescope gain, options are `'exp'` or `'cos'`

`gainexp` : `numpy.ndarray` or `None`

exponent for `'cos'` gain model. If array, must have length `len(nus)`. Can acts as a flag to turn on elevation-dependent gain for specific subbands

`timefac` : `numpy.ndarray` or `None`

optional binary integer array of length `len(nus)`, acts as a flag to turn on elevation-dependent integration time for specific subbands

`optimize_freq` : `optimize.OptimizeFrequency` instance or `None`

arguments to pass to `frequencyoptimizer.FrequencyOptimizer`. If set, will compute the TOA uncertainty in the subband that minimizes `'sigma_tot'`, stored under the key
`scope_name + '_freqopt'`.

`verbose` : `bool`

print sigma computation and results to stdout

`max_workers` : `int` or `None`

number of parallel forks over which to distribute pulsars for computation. Must not exceed number of CPUs - 2. Cannot simultaneously parallelize frequency-optimization (`optimize.OptimizeFrequency.ncpu`).

---
---
## The OptimizeFrequency class (`optimize.OptimizeFrequency`)

container for observing frequency-optimization arguments to pass to `frequencyoptimizer.FrequencyOptimizer`

### Attributes

* `nsteps`: `int`

Number of steps in the grid to run when `log_grid = True`
* `dnu` : `float`

Delta nu, search grid frequency spacing when `log_grid = False`
* `log_grid` : `bool`

Use a log-space grid of center frequencies and bandwidths
* `frac_bw` : `bool`

Run in fractional bandwidth mode
* `full_bandwidth` : `bool`

Enforce full bandwidth in calculations
* `min_bw`: `float`

Minimum bandwidth (GHz) to consider in frequency optimization
* `plot`: `bool`

Write optimizer grid plots
* `plotdir`: `string`

Directory in which to write plots
* `levels` : `numpy.ndarray`

Array of contour levels for plotting
* `colors` : `list`

List of contour colors for plotting
* `lws` : `list`

List of contour linewidths for plotting
* `ncpu` : `int`

Number of cpus to use for parallel computing. Cannot mix with pulsar-parallelization (`calc_timing(max_workers>1)`)

---
---
## The OptimizeTime class (`optimize.OptimizeTime`)

Compute the optimal per-pulsar integration time that maximizes sensitivity to the gravitational wave background.

### Attributes

* `pta` : `pta.PTA instance`

`pta.psrlist` must not be mutated, instance attribute will still point to original `PTA` list
* `nus` : `numpy.ndarray`

evenly-spaced array of observing frequency subbands (GHz)
* rxspecfile : `string`

path to receiver specifications file (see `calc_timing` module)

* `dec_lim` : `tuple`

`(max, min)` telescope declination limits (degress)

* `lat` : `float`

telescope latitude (degrees)
* `t_int0` : `numpy.ndarray` or `None`

initial guess, per-pulsar integration times (s). If `None`, assumes budget distributed equally.
* `t_int_min` : `float`

minimum integration time (s). Default is 60 s.
* `t_int_maxtot` : `float`

total time budget per epoch (s). Default is 1 month.
* `epoch_days` : `float`

duration of an observing epoch (days). Default is 30 days.
* `timefac` : `numpy.ndarray` or `None`

binary array of flags to turn on freq-dependent `t_int`
* `gainmodel` : `string`

telescope elevation-dependent gain model ('cos', 'exp')
* `gainexp` : `float` or `numpy.ndarray`

exponent for 'cos' gain model
* `optimize_freq` : `optimize.OptimizeFrequency` or `None`

freq optimization parameters
* `timespan_yr` : `float`

PTA data duration (years, assumed common across pulsars)
* `cadence` : `int`

number of observations per year

* `n_gw_freq` : `int`

number of GW frequencies at which to estimate power spectra. Default = 400.

* `gwb_strainamp` : `float`

GWB dimensionless strain amplitude (default=`2.4e-15`, NANOGrav 15-year Bayesian posterior)
* `gwb_spindex` : `float`

GWB dimensionless strain spectral index (default=-2/3, ensemble of circular SMBHB)
* `max_workers` : `int`

maximum parallel forks when computing PTA sigmas

### Methods

---
`set_tint_from_grid(self, n_levels, log=False)`

Set up resolution of sigmas lookup table. Set Pulsar `t_int` dicts with keys `"self.instr_name + _tinti"` where i is from 0 to `n_levels` based on a grid of integration times. Resets `Pulsar` `sigmas`, `telescope_noise`, `optimum` and `t_int` dicts

---
`fill_tint_lookup_table(self)`

Compute `sigmas` dict for each integration time set using `self.set_tint_from_grid`

---
`sigma_interpolator(self, pulsar)`

return scipy.interpolate.PchipInterpolator of sigma_tot(tint) for a single pulsar

---
`interp_sigma(self, pulsar, tint_find)`

evaluate interpolated sigma at `tint_find`

---
`evaluate_snr(self, psrdict, t_vec)`

Compute the GWB S/N for updated vector of integration times using `calc_timing.calc_timing`

---
`evaluate_snr_from_lut(self, psrdict, t_vec)`

Compute the GWB S/N for updated vector of integration times
using sigma lookup table instead of `calc_timing`

---
`maximize_snr_trust_constr(self, noisemodel="wn", t0=None, maxiter=500, init_trustrad=1.0, init_constrpenalty=1.0, gtol=1e-6, xtol=1e-10, barrier_tol=1e-10, obj_scale="equal", verbose=0, return_history=False)`

Maximize GWB S/N squared using Scipy trust-constr algorithm. 

<ins>Parameters</ins>

`noisemodel` : `string`

Pulsar residual power spectrum model. Options are `'wn'`, which neglects pulsar intrinsic red noise and GWB noise or `'wnrn'` which includes pulsar and GWB red noise. Default is `'wn'`.

`t0` : `numpy.ndarray` or `None`

optional initial guess (N,). If provided, will be projected to the feasible set {bounds + budget equality}. Default is equally-distributed time.

`maxiter` : `int`

scipy trust-constr maximum number of algorithm iterations

`init_trustrad` : `float`

scipy trust-constr initial radius of trust region for 2nd-order approximation of objective

`init_constrpenalty` : `float`

scipy trust-constr initial value of dynamic penalty for violating constraints

`gtol` : `float`

scipy trust-constr tolerance for infinity norm of Lagrangian gradient

`xtol` : `float`

scipy trust-constr tolerance for change in trust radius

`barrier_tol` : `float`

scipy trust-constr tolerance for inequality constraint barrier parameter

`obj_scale` : `'equal'`, `float` or `None`

optimize a scaled objective instead of the GWB S/N squared. Default is `'equal'`, the GWB S/N squared of equally-distributed time

`return_history` : `bool`

If `True`, log t, F, ||proj grad||_inf (approx) at each iteration in `info`

<ins>Returns</ins>

`t_star` : `(len(pta.psrlist),) numpy.ndarray`

optimal integration time (s) vector

`F_star` : `float`

value of the un-scaled objective function at the optimum (GWB S/N squared)

`info` : `dict`

dictionary of final `scipy.minimize` results, violations at termination, and optionally per-iteration evaluations (`return_history=True`)

---
`random_multistart_optimizer(self, nsamp=100, maxiter=500, spiky_delta_t=3600., init_trustrad=1e4, init_trustconstrpen=1e3, trustconstr_gtol=1e-8, trustcontrs_xtol=1e-8, verbose=False, vverbose=False, optimizer="trust-constr", sample_type="uniform", noisemodel="wn")`

An optimal solution stability diagnostic to test sensitivity to optimizer
initial conditions. Run nsamp gradient optimizers with randomly sampled
initial time vectors either 'spiky' (allocated to one pulsar near it's upper bound)
or 'uniform' on the feasible polytope that obey the budget.
Supports white noise-only noise model `'wn'` or white noise + red noise
model `'wnrn'`.

<ins>Returns</ins>

t_opts : `(nsamp, len(pta.psrlist)) numpy.ndarray`

optimal time vectors for each random start optimizer

f_opts : `(nsamp,) numpy.ndarray`

objective maxima for each random start optimizer

t0s_proj : `(nsamp, len(pta.psrlist)) numpy.ndarray`

feasible initial starting vectors for each random start optimizer

---
`random_timeswap_perturbation(self, t_opt, nswaps=100, noisemodel="wn", F_opt=None, delta_t=3600.0, adaptive_delta=False, verbose=False, tol=1e-10)`

Heuristic local-optimality check:

move +delta_t from j -> i (i gains time, j loses time), keeping sum fixed.

<ins>Returns</ins>

swap indices array ([incr, decr]), swap time vectors, delta F

---
`kkt_residual_box_eq(gradF, tvec, t_mins, t_maxes, t_budget, tol=1e-8, scale=True)`

Bound-aware KKT residual for maximize F(t)=rho^2 subject to:

t_mins[i] <= tvec[i] <= t_maxes[i],  sum(tvec)=t_budget.

<ins>Returns</ins>

`eps_kkt` : `float`

scalar residual (scaled if `scale=True`)

`info` : `dict`

dictionary of estimated interior Lagrange multiplier, residual vector, feasibility, active sets