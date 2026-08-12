"""
Consistency/integration tests for the values in Pulsar.sigmas dicts
"""
import numpy as np
import unittest
import pickle

@unittest.skip("NG15yr.pta no longer included in repo. Will need to have a test object of PTA and Pulsars, run calc_timing on database of instruments/specs, then run this test.")
class Test_sigma_tot_eq_quadsum(unittest.TestCase):
    """
    Parameterized test to check that for all visible pulsars, with all receiver
    combos, sigma_tot ** 2 = sigma_dm ** 2 + sigma_tel ** 2
    """
    def setUp(self):
        with open("NG15yr.pta" ,"rb") as f:
            self.pta = pickle.load(f, encoding="latin1")
        self.rcvr_keys = self.pta.psrlist[0].get_instr_keys()

    def test_sigma_tot_eq_quadsum(self):
        for k in self.rcvr_keys:
            with self.subTest(instr_key=k):
                sigma_tot = [p.sigmas[k]["sigma_tot"] for p in self.pta.psrlist if not p.sigmas[k]["sigma_tot"] < 0]
                quadsum = [np.sqrt(p.sigmas[k]["sigma_dm"]**2 + p.sigmas[k]["sigma_tel"]**2) for p in self.pta.psrlist if not p.sigmas[k]["sigma_tot"] < 0 ]
                np.testing.assert_allclose(sigma_tot, quadsum)

if __name__ == '__main__':
    unittest.main()
        
