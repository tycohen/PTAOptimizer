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
        self.pulsar1_old = Pulsar(name="testpulsar1",
                                 dec=90.,
                                 ra=180.,
                                 sigmas={"test_config": {"sigma_tot": 10.}})
        self.pulsar2_old = Pulsar(name="testpulsar2",
                                  dec=0.,
                                  ra=0.,
                                  sigmas={"test_config": {"sigma_tot": 10.}})
        self.pulsar1_new = Pulsar(name="testpulsar1",
                                 dec=90.,
                                 ra=180.,
                                 sigmas={"test_config": {"sigma_tot": 1.}})
        self.pulsar2_new = Pulsar(name="testpulsar2",
                                 dec=0.,
                                 ra=0.,
                                 sigmas={"test_config": {"sigma_tot": 1.}})
        self.pta_old = PTA(psrlist=[self.pulsar1_old, self.pulsar2_old])
        self.pta_new = PTA(psrlist=[self.pulsar1_new, self.pulsar2_new])        
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
        NcalInv_old = self.psrdict["spectra"]["testpulsar1"]._NcalInv
        spec_id_old = id(self.psrdict["spectra"]["testpulsar1"])
        gw.update_noise_spectra_approx(self.psrdict, self.pta_new)
        NcalInv_new = self.psrdict["spectra"]["testpulsar1"]._NcalInv
        spec_id_new = id(self.psrdict["spectra"]["testpulsar1"])
        self.assertEqual(spec_id_old, spec_id_new)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(NcalInv_old, NcalInv_new)

    def test_gwb_snr_increases_after_NcalInv_approx_update(self):
        """
        Checks if the value returned by 'gwb_snr' is greater after
        updating each Spectrum.NcalInv with pulsars with a lower 'sigma_tot'
        """
        snr_old = gw.gwb_snr(self.psrdict)
        gw.update_noise_spectra_approx(self.psrdict, self.pta_new)
        snr_new = gw.gwb_snr(self.psrdict)
        self.assertGreater(snr_new, snr_old)

    def test_snr_after_update_independent_of_initial_sigma(self):
        """
        Test if _NcalInv after spectrum update fully controls the noise
        as it pertains to the GWB S/N. Update a 1000 us PTA and a 10 us 
        PTA with 1 us noise using gw.update_noise_spectra_approx and see 
        if both produce the same S/N
        """
        # two psrdicts with very different initial sigmas
        p1_noisy = Pulsar(name="testpulsar1", dec=90., ra=180.,
                          sigmas={"test_config": {"sigma_tot": 1000.}})
        p2_noisy = Pulsar(name="testpulsar2", dec=0., ra=0.,
                          sigmas={"test_config": {"sigma_tot": 1000.}})
        pta_noisy = PTA(psrlist=[p1_noisy, p2_noisy])

        # psrdict_A uses "old" (10 us); psrdict_B uses "very noisy" (1000 us)
        psrdict_A = gw.get_hasasia_psrs(self.pta_old, "test_config",
                                        timespan_yr=15., cadence=12, n_freqs=400,
                                        use_best_instr=False,
                                        gwb_strainamp=2.4e-15, gwb_spindex=-2/3.)
        psrdict_B = gw.get_hasasia_psrs(pta_noisy, "test_config",
                                        timespan_yr=15., cadence=12, n_freqs=400,
                                        use_best_instr=False,
                                        gwb_strainamp=2.4e-15, gwb_spindex=-2/3.)
        # update both to the same "new" sigmas (1 us) 
        gw.update_noise_spectra_approx(psrdict_A, self.pta_new)
        gw.update_noise_spectra_approx(psrdict_B, self.pta_new)

        snr_A = gw.gwb_snr(psrdict_A)
        snr_B = gw.gwb_snr(psrdict_B)

        # If the Spectrum update fully controls the SNR noise,
        # these should match to tight numerical tolerance.
        np.testing.assert_allclose(snr_A, snr_B, rtol=1e-10, atol=0.0)

class test_get_hasasia_psrs(unittest.TestCase):

    def setUp(self):
        self.pulsar1 = Pulsar(name="testpulsar1",
                              dec=90.,
                              ra=180.,
                              sigmas={"test_config": {"sigma_tot": 10.}})
        self.pulsar2 = Pulsar(name="testpulsar2",
                              dec=0.,
                              ra=0.,
                              sigmas={"test_config": {"sigma_tot": 1.}})
        self.pta = PTA(psrlist=[self.pulsar1, self.pulsar2])
        self.gwb_spindex = -2 / 3.
        self.psrdict = gw.get_hasasia_psrs(self.pta, "test_config",
                                           timespan_yr=15.,
                                           cadence=12, n_freqs=400,
                                           use_best_instr=False,
                                           gwb_strainamp=2.4e-15,
                                           gwb_spindex=self.gwb_spindex,
                                           return_sencurve=False)

    def test_psrdict_contains_correct_positive_gwb_gamma_conversion(self):
        gwb_gamma_expected = 13 / 3.
        np.testing.assert_allclose(self.psrdict['gwb_gamma'],
                                   gwb_gamma_expected,
                                   rtol=1e-10)

        
class test_rednoise_charstrain2psd(unittest.TestCase):

    def test_red_noise_strain_alpha_to_psd_gamma_conversion(self):
        red_alpha = -2 / 3.
        red_gamma_answer = 13 / 3.
        __, red_gamma = gw.rednoise_charstrain2psd(1., red_alpha)
        np.testing.assert_allclose(red_gamma,
                                   red_gamma_answer,
                                   rtol=1e-10)
 
class test_rednoise_psd2charstrain(unittest.TestCase):

    def test_red_noise_psd_gamma_to_strain_alpha_conversion(self):
        red_gamma = 13 / 3.
        red_alpha_answer = -2 / 3.
        __, red_alpha = gw.rednoise_psd2charstrain(1., red_gamma)
        np.testing.assert_allclose(red_alpha,
                                   red_alpha_answer,
                                   rtol=1e-10)
       
