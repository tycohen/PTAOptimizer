"""
Unit tests for optimize module
"""

import unittest
import numpy as np
import parameterized as ptzd
from pulsar import Pulsar
from pta import PTA
import gravitational_waves as gw
import optimize

class test_OptimizeTime_state_when_using_LUT(unittest.TestCase):
    """
    Tests for state leakage in OptimizeTime instance or psrdict
    when using lookup table/interpolation to compute GWB S/N
    """
    def setUp(self):
        self.pulsar1 = Pulsar(name="testpulsar1",
                              dec=90.,
                              ra=180.,
                              t_int={"testconfig_tint0": 100.,
                                     "testconfig_tint1": 2.78255940e02},
                              sigmas={"testconfig_tint0": {"sigma_tot": 0.1},
                                      "testconfig_tint1": {"sigma_tot": 0.06}})
        self.pulsar2 = Pulsar(name="testpulsar2",
                              dec=0.,
                              ra=0.,
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
                                 ra=180.)
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
                                 ra=180.)
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

        
