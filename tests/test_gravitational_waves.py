"""
Unit tests for gravitational_waves module
"""

import unittest
import numpy as np
import parameterized as ptzd
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
from pulsar import Pulsar
from pta import PTA
import gravitational_waves as gw

class test_update_noise_spectra_approx(unittest.TestCase):

    def setUp(self):
        self.pulsar_old = Pulsar(name="testpulsar",
                                 dec=90.,
                                 ra=180.,
                                 sigmas={"test_config": {"sigma_tot": 10.}})
        self.pulsar_new = Pulsar(name="testpulsar",
                                 dec=90.,
                                 ra=180.,
                                 sigmas={"test_config": {"sigma_tot": 1.}})
        self.pta_old = PTA(psrlist=[self.pulsar_old])
        self.pta_new = PTA(psrlist=[self.pulsar_new])        
        self.psrdict = gw.get_hasasia_psrs(self.pta_old, "test_config",
                                           timespan_yr=15.,
                                           cadence=12, n_freqs=400,
                                           use_best_instr=False,
                                           gwb_strainamp=2.4e-15,
                                           gwb_spindex=-2/3.,
                                           return_sencurve=False)

    def test_Spectrum_updates_with_new_NcalInv(self):
        """
        Checks if Spectrum object in 'psrdict' had the same ID but
        different _NcalInv
        """
        NcalInv_old = self.psrdict["spectra"]["testpulsar"]._NcalInv
        spec_id_old = id(self.psrdict["spectra"]["testpulsar"])
        gw.update_noise_spectra_approx(self.psrdict, self.pta_new)
        NcalInv_new = self.psrdict["spectra"]["testpulsar"]._NcalInv
        spec_id_new = id(self.psrdict["spectra"]["testpulsar"])
        self.assertEqual(spec_id_old, spec_id_new)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(NcalInv_old, NcalInv_new)
