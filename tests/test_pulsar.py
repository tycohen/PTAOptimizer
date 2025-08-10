"""
Unit tests for Pulsar class
"""

import unittest
from pulsar import Pulsar
import parameterized as ptzd

class test_Pulsar__init__(unittest.TestCase):

    def setUp(self):
        self.psr = Pulsar(name='J0605+3757',
                          dm=20.946279,
                          dist=0.6964,
                          taud=0.06034000000000001,
                          s_1000=0.8309647666059053,
                          period=0.0027279541573215337,
                          dtd=371.9,
                          weff=452.3658965742157,
                          w50=150.68814201799654,
                          sig_j_single=94.93352947133782,
                          dnud=0.0030600000000000002,
                          dec=37.95998669958335,
                          ra=91.2730997034583,
                          spindex=-1.7575325430182809,
                          uscale=5.133881283010092)

    def test_Pulsar_sigmas__init__empty_dict(self):
        self.assertEqual(self.psr.sigmas, {})

    def test_Pulsar_sigmas__init__empty_dict(self):
        self.assertEqual(self.psr.telescope_noise, {})

class test_Pulsar_add_sigmas(unittest.TestCase):

    def setUp(self):
        self.psr = Pulsar(name='J0605+3757',
                          dm=20.946279,
                          dist=0.6964,
                          taud=0.06034000000000001,
                          s_1000=0.8309647666059053,
                          period=0.0027279541573215337,
                          dtd=371.9,
                          weff=452.3658965742157,
                          w50=150.68814201799654,
                          sig_j_single=94.93352947133782,
                          dnud=0.0030600000000000002,
                          dec=37.95998669958335,
                          ra=91.2730997034583,
                          spindex=-1.7575325430182809,
                          uscale=5.133881283010092)

    def test_add_sigmas_value_order(self):
        inject_sigmas = (0.1, 0.05, 0.02, 0.01, 0.04)
        ordered_keys = ['sigma_tot',
                        'sigma_white',
                        'sigma_dm',
                        'sigma_tel',
                        'sigma_rn']
        self.psr.add_sigmas("test_instr", inject_sigmas)
        self.assertTrue([self.psr.sigmas["test_instr"][k] == s
                         for k,s in zip(ordered_keys, inject_sigmas)])
        
class test_Pulsar_detected(unittest.TestCase):

    def setUp(self):
        self.psr = Pulsar(name='J0605+3757',
                          dm=20.946279,
                          dist=0.6964,
                          taud=0.06034000000000001,
                          s_1000=0.8309647666059053,
                          period=0.0027279541573215337,
                          dtd=371.9,
                          weff=452.3658965742157,
                          w50=150.68814201799654,
                          sig_j_single=94.93352947133782,
                          dnud=0.0030600000000000002,
                          dec=37.95998669958335,
                          ra=91.2730997034583,
                          spindex=-1.7575325430182809,
                          uscale=5.133881283010092)

    @ptzd.parameterized.expand([(True, {"instr1" : {'sigma_tot': 1.,
                                                      'sigma_white': 1.,
                                                      'sigma_tel': 1.,
                                                      'sigma_rn': 1.,
                                                      'sigma_dm': 1.},
                                          "instr2" : {'sigma_tot': 2.,
                                                      'sigma_white': 2.,
                                                      'sigma_tel': 2.,
                                                      'sigma_rn': 2.,
                                                      'sigma_dm': 2.}}),
                                (False, {"instr1" : {'sigma_tot': 1.,
                                                      'sigma_white': 1.,
                                                      'sigma_tel': 1.,
                                                      'sigma_rn': 1.,
                                                      'sigma_dm': 1.},
                                          "instr2" : {'sigma_tot': -2.,
                                                      'sigma_white': -2.,
                                                      'sigma_tel': -2.,
                                                      'sigma_rn': -2.,
                                                      'sigma_dm': -2.}})],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(par.args[0]))).join(fxn.__name__.split("_bool_")))
    def test_Pulsar_detected_every_instr_returns_bool_(self,
                                                       detected,
                                                       sigmas):
        self.psr.sigmas = sigmas
        if detected:
            self.assertTrue(self.psr.detected())
        else:
            self.assertFalse(self.psr.detected())
    
        
if __name__ == '__main__':
    unittest.main()
