"""
Unit tests for gravitational_waves module
"""

import unittest
import numpy as np
from copy import deepcopy
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

@ptzd.parameterized_class(("pulsar_sigma_tots",),
                          [((.1, 10., 0.1, 0.01),),
                           ((100, 20, 1e-3, 10),)])
class test_quadratic_form_snr(unittest.TestCase):
    """
    Test gravitational_waves.build_W_matrix, 
    gravitational_waves.build_Q_matrix,
    gravitational_waves.build_tildeQ_blocks,
    gravitational_waves.gwb_snr_quad
    """
    def setUp(self):
        # initialize pulsars with empty sigma dicts to be filled by ptzd
        self.pulsar1 = Pulsar(name="testpulsar1",
                              dec=90.,
                              ra=180.,
                              sigmas={"test_config": {"sigma_tot": None}})
        self.pulsar2 = Pulsar(name="testpulsar2",
                              dec=0.,
                              ra=0.,
                              sigmas={"test_config": {"sigma_tot": None}})
        self.pulsar3 = Pulsar(name="testpulsar3",
                              dec=45.,
                              ra=270.,
                              sigmas={"test_config": {"sigma_tot": None}})
        self.pulsar4 = Pulsar(name="testpulsar4",
                              dec=60.,
                              ra=270.,
                              sigmas={"test_config": {"sigma_tot": None}})
        self.pta = PTA(psrlist=[self.pulsar1,
                                self.pulsar2,
                                self.pulsar3,
                                self.pulsar4])
        for p, s in zip(self.pta.psrlist, self.pulsar_sigma_tots):
            p.sigmas["test_config"]["sigma_tot"] = s
        self.timespan_yr = 15.
        self.cadence = 12
        self.n_freqs = 400
        self.psrdict = gw.get_hasasia_psrs(self.pta, "test_config",
                                           timespan_yr=self.timespan_yr,
                                           cadence=self.cadence,
                                           n_freqs=self.n_freqs,
                                           use_best_instr=False,
                                           # GWB RN enters Q only through S_h(f)
                                           # not S_eff(f)
                                           gwb_strainamp=2.4e-15,
                                           gwb_spindex=-2/3,
                                           return_sencurve=False)

    def test_white_noise_only_snr_from_build_Q_matrix_consistent_with_hasasia(self):
        """
        Test that p^T * Q * p gives same S/N as
        hasasia.GWBSensitivityCurve.SNR, where p[i] = sigma[i] ** -2
        """
        sigmas_sec = np.array([p.sigmas["test_config"]["sigma_tot"] / 1e6
                               for p in self.pta.psrlist]) 
        invsig2 =  sigmas_sec ** -2 
        Q = gw.build_Q_matrix(self.psrdict)
        snr_quadratic = np.sqrt(np.dot(np.dot(invsig2.T, Q), invsig2))

        # pulsar noise needs to not contain GWB, but S/N needs to be computed
        # on GWB spectrum so can't use gw.gwb_snr
        names = [p.name for p in self.psrdict["psrs"].values()]
        phis = np.array([p.phi for p in self.psrdict["psrs"].values()])
        thetas = np.array([p.theta for p in self.psrdict["psrs"].values()])
        # initialize pulsars with no GWB contribution to S_eff(f)
        psrs = hsim.sim_pta(psr_names=names,
                            timespan=self.timespan_yr,
                            cad=self.cadence,
                            sigma=sigmas_sec,
                            phi=phis,
                            theta=thetas,
                            A_gwb=0.,
                            alpha_gwb=0.,
                            freqs=self.psrdict["freqs"])
        hs_spectra = []
        for p in psrs:
            sp = hsen.Spectrum(p, freqs=self.psrdict["freqs"])
            # No need for NcalInv approx, NcalInv is exact for WN-only
            sp.NcalInv
            hs_spectra.append(sp)
        scurve = hsen.GWBSensitivityCurve(hs_spectra)
        Sh = hsen.S_h(2.4e-15,
                      -2/3,
                      self.psrdict["freqs"])
        snr_hasasia = scurve.SNR(Sh)
        np.testing.assert_allclose(snr_quadratic,
                                   snr_hasasia,
                                   rtol=1e-8)

    def test_white_plus_red_noise_snr_from_build_tildeQ_blocks_consistent_with_hasasia(self):
        """
        Test that :math: `\rho^2 = \sum_k a(f_k)^T {\bf Q}(f_k) a(f_k)`
        gives same S/N as hasasia.GWBSensitivityCurve.SNR, 
        where a_fk[i] = 1 / psrdict["spectra"][i]
        """
        Q_fk = gw.build_tildeQ_blocks(self.psrdict)
        a_fk = np.array([1 / s.S_I for s in self.psrdict["spectra"].values()]).T
        snr_quadratic = np.sqrt(gw.gwb_snr_quad(Q_fk, a_fk))

        sigmas_sec = np.array([p.sigmas["test_config"]["sigma_tot"] / 1e6
                               for p in self.pta.psrlist]) 
        names = [p.name for p in self.psrdict["psrs"].values()]
        phis = np.array([p.phi for p in self.psrdict["psrs"].values()])
        thetas = np.array([p.theta for p in self.psrdict["psrs"].values()])
        psrs = hsim.sim_pta(psr_names=names,
                            timespan=self.timespan_yr,
                            cad=self.cadence,
                            sigma=sigmas_sec,
                            phi=phis,
                            theta=thetas,
                            A_gwb=2.4e-15,
                            alpha_gwb=-2/3,
                            freqs=self.psrdict["freqs"])
        hs_spectra = []
        for p in psrs:
            sp = hsen.Spectrum(p, freqs=self.psrdict["freqs"])
            # a_fk above doesn't contain NcalInv approx, so not needed here
            sp.NcalInv
            hs_spectra.append(sp)
        scurve = hsen.GWBSensitivityCurve(hs_spectra)
        Sh = hsen.S_h(2.4e-15,
                      -2/3,
                      self.psrdict["freqs"])
        snr_hasasia = scurve.SNR(Sh)
        np.testing.assert_allclose(snr_quadratic,
                                   snr_hasasia,
                                   rtol=1e-8)

class test_gwb_snr_white_noise_only(unittest.TestCase):
    """
    Test gravitational_wave.gwb_snr when the pulsar PSD contains
    only white noise
    """
    def setUp(self):
        self.n_lut_samp = 20
        # realistic pulsar LUTs w/ sigma \propto 1 / sqrt(t)
        tint_psr1 = np.logspace(np.log10(60.), 6.086, self.n_lut_samp)
        self.pulsar1 = Pulsar(name="testpulsar1", #J1713+0747
                        dec=7.79,
                        ra=258.46,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr1)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.125 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr1)})
        tint_psr2 = np.logspace(np.log10(60.), 5.984, self.n_lut_samp)
        self.pulsar2 = Pulsar(name="testpulsar2", #J1643-1224
                        dec=-12.42,
                        ra=250.91,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr2)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.651 / np.sqrt(t) + 0.054}
                                for i, t in enumerate(tint_psr2)})
        tint_psr3 = np.logspace(np.log10(60.), 6.012, self.n_lut_samp)
        self.pulsar3 = Pulsar(name="testpulsar3", #J2145-0750
                        dec=-7.84,
                        ra=326.46,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr3)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 14.368 / np.sqrt(t) + 0.004}
                                for i, t in enumerate(tint_psr3)})
        tint_psr4 = np.logspace(np.log10(60.), 6.079, self.n_lut_samp)
        self.pulsar4 = Pulsar(name="testpulsar4", #J2017+0603
                        dec=6.05,
                        ra=304.35,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr4)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 5.841 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr4)})
        tint_psr5 = np.logspace(np.log10(60.), 6.065, self.n_lut_samp)
        self.pulsar5 = Pulsar(name="testpulsar5", #J1102+0249
                        dec=2.824,
                        ra=165.675,
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
        # values to use only when computing the GWB strain PSD S_h
        self.snr_gwbamp = 2.4e-15
        self.snr_gwbidx = -2/3

    def test_gwb_snr_whitenoise_only_matches_quadratic_form(self):
        """
        Test gravitational_waves.gwb_snr matches white-noise-only
        quadratic form SNR when amp and spindex are zeroed in 
        gravitational_waves.get_hasasia_psrs and overridden with 
        kwarg values in gwb_snr
        """
        # just test the middle sigma value for now
        lut_idx = self.n_lut_samp // 2
        instrument = "testconfig_tint{}".format(lut_idx)
        self.psrdict = gw.get_hasasia_psrs(self.pta,
                                           instrument,
                                           timespan_yr=15.,
                                           cadence=12, n_freqs=400,
                                           use_best_instr=False,
                                           # GWB doesnt contribute to pulsar PSD
                                           gwb_strainamp=0.,
                                           gwb_spindex=0.)
        # compute the value from gravitational_waves module
        gwb_snr = gw.gwb_snr(self.psrdict,
                             gwb_strainamp=self.snr_gwbamp,
                             gwb_spindex=self.snr_gwbidx)
        # compute the value from quadratic form
        sigmas_sec = np.array([p.sigmas[instrument]["sigma_tot"] / 1e6
                               for p in self.pta.psrlist])
        invsig2 =  sigmas_sec ** -2
        psrdict_quad = deepcopy(self.psrdict)
        # hack GWB values back into psrdict to compute Q
        psrdict_quad["gwb_strainamp"] = self.snr_gwbamp
        psrdict_quad["gwb_spindex"] = self.snr_gwbidx
        Q = gw.build_Q_matrix(psrdict_quad)
        snr_quadratic = np.sqrt(np.dot(np.dot(invsig2.T, Q), invsig2))
        np.testing.assert_allclose(gwb_snr,
                                   snr_quadratic,
                                   rtol=1e-8)

    def test_gwb_snr_WN_only_matches_quadratic_form_after_update_approx(self):
        """
        test gwb_snr when white noise only matches quadratic white noise form
        after updating spectra with 
        gravitational_waves.update_noise_spectra_approx
        """
        instr_init = "testconfig_tint0"
        self.psrdict = gw.get_hasasia_psrs(self.pta,
                                           instr_init,
                                           timespan_yr=15.,
                                           cadence=12, n_freqs=400,
                                           use_best_instr=False,
                                           # GWB doesnt contribute to pulsar PSD
                                           gwb_strainamp=0.,
                                           gwb_spindex=0.)
        # update psrdict["spectra"] in place
        instr_update = "testconfig_tint1"
        self.psrdict["instruments"] = [instr_update] * len(self.pta.psrlist)
        gw.update_noise_spectra_approx(self.psrdict, self.pta)
        # compute the value from gravitational_waves module
        gwb_snr = gw.gwb_snr(self.psrdict,
                             gwb_strainamp=self.snr_gwbamp,
                             gwb_spindex=self.snr_gwbidx)
        # compute the value from quadratic form
        sigmas_sec = np.array([p.sigmas[instr_update]["sigma_tot"] / 1e6
                               for p in self.pta.psrlist])
        invsig2 =  sigmas_sec ** -2
        psrdict_quad = deepcopy(self.psrdict)
        # hack GWB values back into psrdict to compute Q
        psrdict_quad["gwb_strainamp"] = self.snr_gwbamp
        psrdict_quad["gwb_spindex"] = self.snr_gwbidx
        Q = gw.build_Q_matrix(psrdict_quad)
        snr_quadratic = np.sqrt(np.dot(np.dot(invsig2.T, Q), invsig2))
        np.testing.assert_allclose(gwb_snr,
                                   snr_quadratic,
                                   rtol=1e-8)

class test_gwb_snr_quad_white_plus_red_noise(unittest.TestCase):
    """
    Test gravitational_wave.gwb_snr_quad when the pulsar PSD contains
    both white and red noise
    """
    def setUp(self):
        self.n_lut_samp = 20
        # realistic pulsar LUTs w/ sigma \propto 1 / sqrt(t)
        tint_psr1 = np.logspace(np.log10(60.), 6.086, self.n_lut_samp)
        self.pulsar1 = Pulsar(name="testpulsar1", #J1713+0747
                        dec=7.79,
                        ra=258.46,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr1)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.125 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr1)})
        tint_psr2 = np.logspace(np.log10(60.), 5.984, self.n_lut_samp)
        self.pulsar2 = Pulsar(name="testpulsar2", #J1643-1224
                        dec=-12.42,
                        ra=250.91,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr2)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 2.651 / np.sqrt(t) + 0.054}
                                for i, t in enumerate(tint_psr2)})
        tint_psr3 = np.logspace(np.log10(60.), 6.012, self.n_lut_samp)
        self.pulsar3 = Pulsar(name="testpulsar3", #J2145-0750
                        dec=-7.84,
                        ra=326.46,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr3)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 14.368 / np.sqrt(t) + 0.004}
                                for i, t in enumerate(tint_psr3)})
        tint_psr4 = np.logspace(np.log10(60.), 6.079, self.n_lut_samp)
        self.pulsar4 = Pulsar(name="testpulsar4", #J2017+0603
                        dec=6.05,
                        ra=304.35,
                        t_int={"testconfig_tint{}".format(i) : t
                               for i, t in enumerate(tint_psr4)},
                        sigmas={"testconfig_tint{}".format(i):
                                {"sigma_tot": 5.841 / np.sqrt(t) + 0.005}
                                for i, t in enumerate(tint_psr4)})
        tint_psr5 = np.logspace(np.log10(60.), 6.065, self.n_lut_samp)
        self.pulsar5 = Pulsar(name="testpulsar5", #J1102+0249
                        dec=2.824,
                        ra=165.675,
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
        # values to use only when computing the GWB strain PSD S_h
        self.snr_gwbamp = 2.4e-15
        self.snr_gwbidx = -2/3

    def test_gwb_snr_quad_matches_gwb_snr(self):
        """
        Test quadratic form in gravitational_waves.gwb_snr_quad
        matches gravitational_waves.gwb_snr for pulsar PSDs containing
        white and red noise
        """
        # just test the middle sigma value for now
        lut_idx = self.n_lut_samp // 2
        instrument = "testconfig_tint{}".format(lut_idx)
        self.psrdict = gw.get_hasasia_psrs(self.pta,
                                           instrument,
                                           timespan_yr=15.,
                                           cadence=12, n_freqs=400,
                                           use_best_instr=False,
                                           gwb_strainamp=self.snr_gwbamp,
                                           gwb_spindex=self.snr_gwbidx)
        # compute the value from gravitational_waves module
        gwb_snr = gw.gwb_snr(self.psrdict)
        # compute the value from quadratic form
        Q_fk = gw.build_tildeQ_blocks(self.psrdict)
        a_fk = np.array([1 / s.S_I for s in self.psrdict["spectra"].values()]).T
        snr_quadratic = np.sqrt(gw.gwb_snr_quad(Q_fk, a_fk))
        np.testing.assert_allclose(gwb_snr,
                                   snr_quadratic,
                                   rtol=1e-8)

    def test_gwb_snr_quad_matches_gwb_snr_after_update_approx(self):
        """
        test gravitational_waves.gwb_snr_quad matches 
        gravitational_waves.gwb_snr when the pulsar PSDs contain both
        white and red noise after updating spectra with 
        gravitational_waves.update_noise_spectra_approx
        """
        instr_init = "testconfig_tint0"
        self.psrdict = gw.get_hasasia_psrs(self.pta,
                                           instr_init,
                                           timespan_yr=15.,
                                           cadence=12, n_freqs=400,
                                           use_best_instr=False,
                                           # GWB doesnt contribute to pulsar PSD
                                           gwb_strainamp=0.,
                                           gwb_spindex=0.)
        # update psrdict["spectra"] in place
        instr_update = "testconfig_tint1"
        self.psrdict["instruments"] = [instr_update] * len(self.pta.psrlist)
        gw.update_noise_spectra_approx(self.psrdict, self.pta)
        # compute the value from gravitational_waves module
        gwb_snr = gw.gwb_snr(self.psrdict)
        # compute the value from quadratic form
        Q_fk = gw.build_tildeQ_blocks(self.psrdict)
        a_fk = np.array([1 / s.S_I for s in self.psrdict["spectra"].values()]).T
        snr_quadratic = np.sqrt(gw.gwb_snr_quad(Q_fk, a_fk))
        np.testing.assert_allclose(gwb_snr,
                                   snr_quadratic,
                                   rtol=1e-8)

                
