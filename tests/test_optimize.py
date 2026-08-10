"""
Unit tests for optimize module
"""

import unittest
import numpy as np
import parameterized as ptzd
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
from ptaoptimizer.pulsar import Pulsar
from ptaoptimizer.pta import PTA
import ptaoptimizer.gravitational_waves as gw
from ptaoptimizer import optimize

class test_OptimizeTime_state_when_using_LUT(unittest.TestCase):
    """
    Tests for state leakage in OptimizeTime instance or psrdict
    when using lookup table/interpolation to compute GWB S/N
    """
    def setUp(self):
        self.pulsar1 = Pulsar(name="testpulsar1",
                              dec=90.,
                              ra=180.,
                              redamp=0.,
                              redgamma=0.,
                              t_int={"testconfig_tint0": 100.,
                                     "testconfig_tint1": 2.78255940e02},
                              sigmas={"testconfig_tint0": {"sigma_tot": 0.1},
                                      "testconfig_tint1": {"sigma_tot": 0.06}})
        self.pulsar2 = Pulsar(name="testpulsar2",
                              dec=0.,
                              ra=0.,
                              redamp=0.,
                              redgamma=0.,
                              t_int={"testconfig_tint0": 100.,
                                     "testconfig_tint1": 2.78255940e02},
                              sigmas={"testconfig_tint0": {"sigma_tot": 0.1},
                                      "testconfig_tint1": {"sigma_tot": 0.06}})
        self.pta = PTA(psrlist=[self.pulsar1, self.pulsar2])
        self.ot = optimize.OptimizeTime(self.pta,
                                        # nus not needed when using LUT
                                        nus=None,
                                        rxspecfile="testconfig",
                                        dec_lim=(90., -41.),
                                        lat=39.,
                                        t_int0=None,
                                        t_int_min=100.,
                                        # 20% of a month total time allocation
                                        t_int_maxtot=0.2 * gw.SECS_PER_YEAR / 12.,
                                        epoch_days=365.25 / 12,
                                        timefac=0.,
                                        gainmodel=None,
                                        gainexp=None,
                                        optimize_freq=None,
                                        timespan_yr=15.,
                                        cadence=12,
                                        n_gw_freq=400,
                                        gwb_strainamp=2.4e-15,
                                        gwb_spindex=-2/3.,
                                        use_best_instr=False)
        self.ot.tint_grid_names = ["testconfig_tint0", "testconfig_tint1"]
        self.psrdict = gw.get_hasasia_psrs(self.pta,
                                           "testconfig_tint0",
                                           timespan_yr=15.,
                                           cadence=12, n_freqs=400,
                                           use_best_instr=False,
                                           gwb_strainamp=2.4e-15,
                                           gwb_spindex=-2/3.)

    def test_same_tvec_evaluates_to_same_snr(self):
        tvec = np.array([100., 100.])
        snr1 = self.ot.evaluate_snr_from_lut(self.psrdict, tvec)
        snr2 = self.ot.evaluate_snr_from_lut(self.psrdict, tvec)
        self.assertAlmostEqual(snr1, snr2, 10)

    def test_different_tvec_evaluation_state_doesnt_leak(self):
        tvec1 = np.array([100., 100.])
        tvec2 = np.array([2.78255940e02, 2.78255940e02])
        snr_a1 = self.ot.evaluate_snr_from_lut(self.psrdict, tvec1)
        snr_b1 = self.ot.evaluate_snr_from_lut(self.psrdict, tvec2)
        snr_a2 = self.ot.evaluate_snr_from_lut(self.psrdict, tvec1)        
        self.assertAlmostEqual(snr_a1, snr_a2, 10)

class test_OptimizeTime_project_budget_to_equality(unittest.TestCase):
    """
    Tests for capped-simplex budget projection onto feasible set
    """

    @ptzd.parameterized.expand([("_already_feasible",
                                 4, 10., np.full(4, 100.), 120.,
                                 np.array([30., 20., 40., 30.]),
                                 np.array([30., 20., 40., 30.])),
                                ("_pure_shift",
                                 3, 0., np.full(3, 100.), 60.,
                                 np.array([30., 30., 30.]),
                                 np.array([20., 20., 20.])),
                                ("_active_upper_bound",
                                 3, 0., np.array([100., 50., 100.]), 120.,
                                 np.array([100., 100., 0.]),
                                 np.array([70., 50., 0.])),
                                ("_active_lower_bound",
                                 3, 10., np.full(3, 100.), 60.,
                                 np.array([0., 30., 30.]),
                                 np.array([10., 25., 25.])),],
                                name_func=lambda fxn, n, par : "_{}".format(ptzd.parameterized.to_safe_name(str(par.args[0]))).join(fxn.__name__.split("_testname_")))
    def test_input_vector_projects_to_expected_testname_(self,
                                                     testname,
                                                     npsrs, tmin, tmax, budget,
                                                     t_vec_in,
                                                     t_vec_out_expected):
        # the pulsar attributes dont really matter, we will overwrite
        # OptimizeTime attributes computed from pulsar attributes
        self.testpulsar = Pulsar(name="testpulsar1",
                                 dec=90.,
                                 ra=180.,
                                 redamp=0.,
                                 redgamma=0.)
        self.pta = PTA(psrlist=[self.testpulsar] * npsrs)
        self.ot = optimize.OptimizeTime(self.pta,
                                        # nus not needed when using LUT
                                        nus=None,
                                        rxspecfile="",
                                        dec_lim=(90., -41.),
                                        lat=39.,
                                        t_int0=None,
                                        t_int_min=100.,
                                        t_int_maxtot=1e6,
                                        epoch_days=365.25 / 12,
                                        timefac=0.,
                                        gainmodel=None,
                                        gainexp=None,
                                        optimize_freq=None,
                                        timespan_yr=15.,
                                        cadence=12,
                                        n_gw_freq=400,
                                        gwb_strainamp=2.4e-15,
                                        gwb_spindex=-2/3.,
                                        use_best_instr=False)
        # hacky, overwrite with parameterized inputs
        self.ot.t_int_min = tmin
        self.ot.t_int_max = tmax
        self.ot.t_int_maxtot = budget
        t_projected = self.ot.project_to_budget_equality(t_vec_in)
        np.testing.assert_allclose(t_projected, t_vec_out_expected,
                                   err_msg="Mismatch between x==t_projected "
                                   "and y==t_vec_out_expected")
        
    def test_infeasible_vector_raises_ValueError(self):
        # the pulsar attributes dont really matter, we will overwrite
        # OptimizeTime attributes computed from pulsar attributes
        npsrs = 4
        tmin = 50.
        tmax = np.full(4, 60.)
        budget = 100.
        t_vec_in = np.full(4, 0.)
        self.testpulsar = Pulsar(name="testpulsar1",
                                 dec=90.,
                                 ra=180.,
                                 redamp=0.,
                                 redgamma=0.)
        self.pta = PTA(psrlist=[self.testpulsar] * npsrs)
        self.ot = optimize.OptimizeTime(self.pta,
                                        # nus not needed when using LUT
                                        nus=None,
                                        rxspecfile="",
                                        dec_lim=(90., -41.),
                                        lat=39.,
                                        t_int0=None,
                                        t_int_min=100.,
                                        t_int_maxtot=1e6,
                                        epoch_days=365.25 / 12,
                                        timefac=0.,
                                        gainmodel=None,
                                        gainexp=None,
                                        optimize_freq=None,
                                        timespan_yr=15.,
                                        cadence=12,
                                        n_gw_freq=400,
                                        gwb_strainamp=2.4e-15,
                                        gwb_spindex=-2/3.,
                                        use_best_instr=False)
        # hacky, overwrite with parameterized inputs
        self.ot.t_int_min = tmin
        self.ot.t_int_max = tmax
        self.ot.t_int_maxtot = budget
        with self.assertRaises(ValueError):
            t_projected = self.ot.project_to_budget_equality(t_vec_in)

class test_OptimizeTime_wn_objective_and_grad(unittest.TestCase):
    """
    Integration tests for OptimizeTime.wn_objective_and_grad white
    noise-only objective and gradient quadratic form calculation
    """
    
    def setUp(self):
        n_lut_samp = 20
        # realistic pulsar LUTs w/ sigma \propto 1 / sqrt(t)
        tint_psr1 = np.logspace(np.log10(60.), 6.086, n_lut_samp)
        self.pulsar1 = Pulsar(name="testpulsar1", #J1713+0747
                        dec=7.79,
                        ra=258.46,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr1)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.125 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr1)})
        tint_psr2 = np.logspace(np.log10(60.), 5.984, n_lut_samp)
        self.pulsar2 = Pulsar(name="testpulsar2", #J1643-1224
                        dec=-12.42,
                        ra=250.91,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr2)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.651 / np.sqrt(t) + 0.054}
                                for i, t in enumerate(tint_psr2)})
        tint_psr3 = np.logspace(np.log10(60.), 6.012, n_lut_samp)
        self.pulsar3 = Pulsar(name="testpulsar3", #J2145-0750
                        dec=-7.84,
                        ra=326.46,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr3)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 14.368 / np.sqrt(t) + 0.004}
                                for i, t in enumerate(tint_psr3)})
        tint_psr4 = np.logspace(np.log10(60.), 6.079, n_lut_samp)
        self.pulsar4 = Pulsar(name="testpulsar4", #J2017+0603
                        dec=6.05,
                        ra=304.35,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr4)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 5.841 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr4)})
        tint_psr5 = np.logspace(np.log10(60.), 6.065, n_lut_samp)
        self.pulsar5 = Pulsar(name="testpulsar5", #J1102+0249
                        dec=2.824,
                        ra=165.675,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr5)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 21.897 / np.sqrt(t) + 0.013}
                                for i, t in enumerate(tint_psr5)})
        self.pta = PTA(psrlist=[self.pulsar1,
                                self.pulsar2,
                                self.pulsar3,
                                self.pulsar4,
                                self.pulsar5])
        self.npsrs = len(self.pta.psrlist)
        self.ot = optimize.OptimizeTime(self.pta,
                                        # nus not needed when using LUT
                                        nus=None,
                                        rxspecfile="testconfig",
                                        dec_lim=(90., -41.),
                                        lat=39.,
                                        t_int0=None,
                                        t_int_min=100.,
                                        # 20% of a month total time allocation
                                        t_int_maxtot=0.2 * gw.SECS_PER_YEAR / 12.,
                                        epoch_days=365.25 / 12,
                                        timefac=0.,
                                        gainmodel=None,
                                        gainexp=None,
                                        optimize_freq=None,
                                        timespan_yr=15.,
                                        cadence=12,
                                        n_gw_freq=400,
                                        gwb_strainamp=2.4e-15,
                                        gwb_spindex=-2/3.,
                                        use_best_instr=False)
        self.ot.tint_grid_names = self.pulsar1.get_instr_keys()
        self.psrdict = gw.get_hasasia_psrs(self.ot.pta,
                                           self.ot.tint_grid_names[0] + \
                                           self.ot.optstr,
                                           timespan_yr=self.ot.timespan_yr,
                                           cadence=self.ot.cadence,
                                           n_freqs=self.ot.n_gw_freq,
                                           use_best_instr=False,
                                           # GWB RN enters Q only through S_h(f)
                                           # not S_eff(f)
                                           gwb_strainamp=self.ot.gwb_strainamp,
                                           gwb_spindex=self.ot.gwb_spindex)
        self.Qmat = gw.build_Q_matrix(self.psrdict)

    def hasasia_snr_wn_only(self, tvec):
        """
        Compute WN-only S/N with hasasia

        Pulsar noise needs to not contain GWB, but S/N needs to be computed
        on GWB spectrum so can't use gw.gwb_snr or ot.evaluate_snr_from_lut.
        """
        names = [p.name for p in self.psrdict["psrs"].values()]
        phis = np.array([p.phi for p in self.psrdict["psrs"].values()])
        thetas = np.array([p.theta for p in self.psrdict["psrs"].values()])
        # interpolate sigmas at tvec from LUT
        self.ot._set_sigma_interp_lut(tvec)
        interp_key = self.ot.instr_name + "_sigma_interp"
        sigmas_sec = np.array([p.sigmas[interp_key]["sigma_tot"] * 1e-6
                               for p in self.ot.pta.psrlist])
        # initialize pulsars with no GWB contribution to S_eff(f)
        psrs = hsim.sim_pta(psr_names=names,
                            timespan=self.ot.timespan_yr,
                            cad=self.ot.cadence,
                            sigma=sigmas_sec,
                            phi=phis,
                            theta=thetas,
                            A_gwb=0.,
                            alpha_gwb=0.,
                            freqs=self.psrdict["freqs"])
        # construct hasasia spectra
        hs_spectra = []
        for p in psrs:
            sp = hsen.Spectrum(p, freqs=self.psrdict["freqs"])
            # NcalInv approx not needed, NcalInv in quad form exact for WN-only
            sp.NcalInv
            hs_spectra.append(sp)
        scurve = hsen.GWBSensitivityCurve(hs_spectra)
        # GWB strainamp spectrum
        Sh = hsen.S_h(2.4e-15,
                      -2/3,
                      self.psrdict["freqs"])
        return scurve.SNR(Sh)

    @ptzd.parameterized.expand([("_equal_time",
                            np.full(5, 0.2 * gw.SECS_PER_YEAR / (12. * 5))),],
                            name_func=lambda fxn, n, par : "_{}".format(ptzd.parameterized.to_safe_name(str(par.args[0]))).join(fxn.__name__.split("_testname_")))
    def test_wn_objective_only_matches_hasasia_testname_(self,
                                                         testname,
                                                         tvec):
        snr_hasasia = self.hasasia_snr_wn_only(tvec) ** 2
        snr_quadratic = self.ot.wn_objective_only(tvec, self.Qmat)
        np.testing.assert_allclose(snr_hasasia, snr_quadratic,
                                   rtol=1e-8, atol=0)

    def calc_p_t(self, tvec):
        """
        compute the vector p_i = sigma_i(t) ** -2 at t=tint
        """
        p_t = []
        for p, t in zip(self.ot.pta.psrlist, tvec):
            p_t.append((self.ot.interp_sigma(p, t) * 1e-6) ** -2)
        return np.array(p_t)
        
    def finite_diff_deriv(self, psr_idx, t0, h=None):
        """
        Centered finite-difference approximation of dF/dt_i for
            F(t) = p(t)^T Q p(t),  p_i(t_i) = (sigma_i(t_i) [sec])^{-2}.

        Only pulsar i's time is perturbed: t -> t ± h e_i.
        All other pulsar times are held fixed at t_base.

        Parameters
        ----------
        ot : OptimizeTime
            Your OptimizeTime instance (must have LUT filled for interp_sigma).
        pulsar_idx : int
            pulsar index in ot.pta.psrlist
        t_i : float
            The baseline integration time for this pulsar (seconds).
        h : float or None
            Step size in seconds. If None, uses h = 1e-3 * max(1, t_i).

        Returns
        -------
        dF_dt_fd : float
            Finite-difference estimate of dF/dt_i at (t_base with t_i set).
        """
        # Baseline equal time vector
        t_base = np.full(self.npsrs, self.ot.t_int_maxtot / self.npsrs,
                         dtype=float)
        t_base[psr_idx] = t0

        # Choose step
        if h is None:
            h = 1e-3 * max(1.0, abs(t0))
        h = float(h)
        if h <= 0 or not np.isfinite(h):
            raise ValueError(f"h must be finite and > 0, got {h}")

        # Keep perturbations within the LUT domain for THIS pulsar
        tmin = float(self.ot.t_int_min)
        tmax_i = float(self.ot.t_int_max[psr_idx])

        t_plus = min(t0 + h, tmax_i)
        t_minus = max(t0 - h, tmin)

        # If we got clipped, the effective h differs on each side.
        # Use symmetric h if possible; otherwise fall back to one-sided.
        use_central = (t_plus > t0) and (t_minus < t0) and \
            np.isclose(t_plus - t0, t0 - t_minus)

        def F_of_tvec(tvec):
            p_t = self.calc_p_t(tvec)   # uses sigma in seconds internally
            return float(p_t.T @ self.Qmat @ p_t)

        # Evaluate
        tvec_plus = t_base.copy()
        tvec_minus = t_base.copy()
        tvec_plus[psr_idx] = t_plus
        tvec_minus[psr_idx] = t_minus

        F_plus = F_of_tvec(tvec_plus)
        F_minus = F_of_tvec(tvec_minus)

        if use_central:
            heff = t_plus - t0  # == t0 - t_minus
            dF_dt = (F_plus - F_minus) / (2.0 * heff)
            scheme = "central"
        else:
            # One-sided (choose whichever side moved)
            if t_plus > t0:
                heff = t_plus - t0
                F0 = F_of_tvec(t_base)
                dF_dt = (F_plus - F0) / heff
                scheme = "forward"
            elif t_minus < t0:
                heff = t0 - t_minus
                F0 = F_of_tvec(t_base)
                dF_dt = (F0 - F_minus) / heff
                scheme = "backward"
            else:
                raise ValueError("Cannot take a finite-difference step: t_i is pinned at bounds.")
        return dF_dt

    def finite_diff_gradient(self, tvec):
        """
        compute gradient at tvec over all pulsars in self.ot.pta.psrlist
        """
        if len(tvec) != self.npsrs:
            raise ValueError("time vector must be same length as pta.psrlist")
        return np.array([self.finite_diff_deriv(i, t)
                         for i, t in zip(np.arange(self.npsrs), tvec)])

    @ptzd.parameterized.expand([("_equal_time",
                            np.full(5, 0.2 * gw.SECS_PER_YEAR / (12. * 5))),],
                            name_func=lambda fxn, n, par : "_{}".format(ptzd.parameterized.to_safe_name(str(par.args[0]))).join(fxn.__name__.split("_testname_")))
    def test_gradient_matches_finite_difference_testname_(self,
                                                          testname,
                                                          tvec):
        analytic_grad = self.ot.wn_grad_only(tvec, self.Qmat)
        findiff_grad = self.finite_diff_gradient(tvec)
        np.testing.assert_allclose(analytic_grad, findiff_grad,
                                   rtol=1e-6, atol=0,
                                   err_msg="Mismatch between "
                                   "x==OptimizeTime.wn_grad_only "
                                   "and y==finite difference gradient")

class test_OptimizeTime_wn_rn_objective_and_grad(unittest.TestCase):
    """
    Integration tests for OptimizeTime.wn_rn_objective_and_grad white
    noise + red noise objective and gradient quadratic form calculation
    """
    
    def setUp(self):
        n_lut_samp = 20
        # realistic pulsar LUTs w/ sigma \propto 1 / sqrt(t)
        tint_psr1 = np.logspace(np.log10(60.), 6.086, n_lut_samp)
        self.pulsar1 = Pulsar(name="testpulsar1", #J1713+0747
                        dec=7.79,
                        ra=258.46,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr1)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.125 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr1)})
        tint_psr2 = np.logspace(np.log10(60.), 5.984, n_lut_samp)
        self.pulsar2 = Pulsar(name="testpulsar2", #J1643-1224
                        dec=-12.42,
                        ra=250.91,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr2)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.651 / np.sqrt(t) + 0.054}
                                for i, t in enumerate(tint_psr2)})
        tint_psr3 = np.logspace(np.log10(60.), 6.012, n_lut_samp)
        self.pulsar3 = Pulsar(name="testpulsar3", #J2145-0750
                        dec=-7.84,
                        ra=326.46,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr3)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 14.368 / np.sqrt(t) + 0.004}
                                for i, t in enumerate(tint_psr3)})
        tint_psr4 = np.logspace(np.log10(60.), 6.079, n_lut_samp)
        self.pulsar4 = Pulsar(name="testpulsar4", #J2017+0603
                        dec=6.05,
                        ra=304.35,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr4)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 5.841 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr4)})
        tint_psr5 = np.logspace(np.log10(60.), 6.065, n_lut_samp)
        self.pulsar5 = Pulsar(name="testpulsar5", #J1102+0249
                        dec=2.824,
                        ra=165.675,
                        redamp=0.,
                        redgamma=0.,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr5)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 21.897 / np.sqrt(t) + 0.013}
                                for i, t in enumerate(tint_psr5)})
        self.pta = PTA(psrlist=[self.pulsar1,
                                self.pulsar2,
                                self.pulsar3,
                                self.pulsar4,
                                self.pulsar5])
        self.npsrs = len(self.pta.psrlist)
        self.ot = optimize.OptimizeTime(self.pta,
                                        # nus not needed when using LUT
                                        nus=None,
                                        rxspecfile="testconfig",
                                        dec_lim=(90., -41.),
                                        lat=39.,
                                        t_int0=None,
                                        t_int_min=100.,
                                        # 20% of a month total time allocation
                                        t_int_maxtot=0.2 * gw.SECS_PER_YEAR / 12.,
                                        epoch_days=365.25 / 12,
                                        timefac=0.,
                                        gainmodel=None,
                                        gainexp=None,
                                        optimize_freq=None,
                                        timespan_yr=15.,
                                        cadence=12,
                                        n_gw_freq=400,
                                        gwb_strainamp=2.4e-15,
                                        gwb_spindex=-2/3.,
                                        use_best_instr=False)
        self.ot.tint_grid_names = self.pulsar1.get_instr_keys()
        self.psrdict = gw.get_hasasia_psrs(self.ot.pta,
                                           self.ot.tint_grid_names[0] + \
                                           self.ot.optstr,
                                           timespan_yr=self.ot.timespan_yr,
                                           cadence=self.ot.cadence,
                                           n_freqs=self.ot.n_gw_freq,
                                           use_best_instr=False,
                                           gwb_strainamp=self.ot.gwb_strainamp,
                                           gwb_spindex=self.ot.gwb_spindex)
        self.Q_fk = gw.build_tildeQ_blocks(self.psrdict)
        interp_key = self.ot.instr_name + "_sigma_interp"
        self.psrdict["instruments"] = [interp_key] * len(self.ot.pta.psrlist)
        
    @ptzd.parameterized.expand([("_equal_time",
                            np.full(5, 0.2 * gw.SECS_PER_YEAR / (12. * 5))),],
                            name_func=lambda fxn, n, par : "_{}".format(ptzd.parameterized.to_safe_name(str(par.args[0]))).join(fxn.__name__.split("_testname_")))
    def test_wn_rn_objective_only_matches_hasasia_testname_(self,
                                                            testname,
                                                            tvec):
        F_hasasia = self.ot.evaluate_snr_from_lut(self.psrdict, tvec) ** 2
        F_quadratic = self.ot.wn_rn_objective_only(tvec, self.Q_fk, self.psrdict)
        np.testing.assert_allclose(F_hasasia, F_quadratic,
                                   rtol=1e-10, atol=0)

    def finite_diff_deriv(self, psr_idx, t0, h=None):
        """
        Centered finite-difference approximation of dF/dt_i for
            F(t) = sum_k[a_fk(t)^T Q_fk a_fk(t)],
            a_fk(t_i) = 1 / S_i(f_k).

        Only pulsar i's time is perturbed: t -> t ± h e_i.
        All other pulsar times are held fixed at t_base.

        Parameters
        ----------
        ot : OptimizeTime
            Your OptimizeTime instance (must have LUT filled for interp_sigma).
        pulsar_idx : int
            pulsar index in ot.pta.psrlist
        t_i : float
            The baseline integration time for this pulsar (seconds).
        h : float or None
            Step size in seconds. If None, uses h = 1e-3 * max(1, t_i).

        Returns
        -------
        dF_dt_fd : float
            Finite-difference estimate of dF/dt_i at (t_base with t_i set).
        """
        # Baseline equal time vector
        t_base = np.full(self.npsrs, self.ot.t_int_maxtot / self.npsrs,
                         dtype=float)
        t_base[psr_idx] = t0

        # Choose step
        if h is None:
            h = 1e-3 * max(1.0, abs(t0))
        h = float(h)
        if h <= 0 or not np.isfinite(h):
            raise ValueError(f"h must be finite and > 0, got {h}")

        # Keep perturbations within the LUT domain for THIS pulsar
        tmin = float(self.ot.t_int_min)
        tmax_i = float(self.ot.t_int_max[psr_idx])

        t_plus = min(t0 + h, tmax_i)
        t_minus = max(t0 - h, tmin)

        # If we got clipped, the effective h differs on each side.
        # Use symmetric h if possible; otherwise fall back to one-sided.
        use_central = (t_plus > t0) and (t_minus < t0) and \
            np.isclose(t_plus - t0, t0 - t_minus)

        def F_of_tvec(tvec):
            return self.ot.evaluate_snr_from_lut(self.psrdict, tvec) ** 2

        # Evaluate
        tvec_plus = t_base.copy()
        tvec_minus = t_base.copy()
        tvec_plus[psr_idx] = t_plus
        tvec_minus[psr_idx] = t_minus

        F_plus = F_of_tvec(tvec_plus)
        F_minus = F_of_tvec(tvec_minus)

        if use_central:
            heff = t_plus - t0  # == t0 - t_minus
            dF_dt = (F_plus - F_minus) / (2.0 * heff)
            scheme = "central"
        else:
            # One-sided (choose whichever side moved)
            if t_plus > t0:
                heff = t_plus - t0
                F0 = F_of_tvec(t_base)
                dF_dt = (F_plus - F0) / heff
                scheme = "forward"
            elif t_minus < t0:
                heff = t0 - t_minus
                F0 = F_of_tvec(t_base)
                dF_dt = (F0 - F_minus) / heff
                scheme = "backward"
            else:
                raise ValueError("Cannot take a finite-difference step: t_i is pinned at bounds.")
        return dF_dt

    def finite_diff_gradient(self, tvec):
        """
        compute gradient at tvec over all pulsars in self.ot.pta.psrlist
        """
        if len(tvec) != self.npsrs:
            raise ValueError("time vector must be same length as pta.psrlist")
        return np.array([self.finite_diff_deriv(i, t)
                         for i, t in zip(np.arange(self.npsrs), tvec)])

    @ptzd.parameterized.expand([("_equal_time",
                            np.full(5, 0.2 * gw.SECS_PER_YEAR / (12. * 5))),],
                            name_func=lambda fxn, n, par : "_{}".format(ptzd.parameterized.to_safe_name(str(par.args[0]))).join(fxn.__name__.split("_testname_")))
    #@unittest.skip("finite difference (as implemented) not a good comparison")    
    def test_gradient_matches_finite_difference_testname_(self,
                                                          testname,
                                                          tvec):
        analytic_grad = self.ot.wn_rn_grad_only(tvec, self.Q_fk, self.psrdict)
        findiff_grad = self.finite_diff_gradient(tvec)
        np.testing.assert_allclose(analytic_grad, findiff_grad,
                                   rtol=1e-6, atol=0,
                                   err_msg="Mismatch between "
                                   "x==OptimizeTime.wn_rn_grad_only "
                                   "and y==finite difference gradient")
