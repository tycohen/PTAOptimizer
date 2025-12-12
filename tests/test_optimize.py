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
