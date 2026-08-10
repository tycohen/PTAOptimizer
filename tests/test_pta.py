import unittest
from ptaoptimizer.pta import PTA
from ptaoptimizer.pulsar import Pulsar
import parameterized as ptzd

'''
Test PTA class
'''

@ptzd.parameterized_class(("sigmadict_test", "answer"),
                          [((.1, .2, .3), "1"),
                           ((.1, .2, -2), "1"),
                           ((.2, .3, .1), "3")])
class Test_sigma_best(unittest.TestCase):

    def setUp(self):
        self.psr = Pulsar(name="pulsar")
        for i, instr_name in enumerate(["1", "2", "3"]):
            # add only the total noise, set other components to None
            sigdict = {'sigma_tot' : self.sigmadict_test[i],
                       'sigma_white' : None,
                       'sigma_dm' : None,
                       'sigma_tel' : None,
                       'sigma_rn' : None}
            self.psr.add_sigmas(instr_name, sigdict)
        self.pta = PTA(psrlist=[self.psr])

    def test_sigma_best_picks_smallest_positive_sigma(self):
#        print(self.pta.sigma_best())
        best_instr = self.pta.sigma_best()[0][1]
        self.assertEqual(best_instr, self.answer)
         
if __name__ == '__main__':
    unittest.main()
