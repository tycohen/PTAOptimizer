import unittest
from pta import PTA
from pulsar import Pulsar
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
            self.psr.add_sigmas(instr_name, (self.sigmadict_test[i],
                                             None, None, None, None))
        self.pta = PTA(psrlist=[self.psr])

    def test_sigma_best_picks_smallest_positive_sigma(self):
#        print(self.pta.sigma_best())
        best_instr = self.pta.sigma_best()[0][1]
        self.assertEqual(best_instr, self.answer)
         
if __name__ == '__main__':
    unittest.main()
