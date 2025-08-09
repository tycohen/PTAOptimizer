import unittest
from pta import PTA
from pulsar import Pulsar
'''
Test PTA class
'''

class ParametrizedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parametrized should
        inherit from this class.
    """
    def __init__(self, methodName='runTest',
                 sigmadict_test=None,
                 answer=None):
        super().__init__(methodName)
        self.sigmadict_test = sigmadict_test
        self.answer = answer

        
    @staticmethod
    def parametrize(testcase_klass,
                    sigmadict_test=None,
                    answer=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing parameters
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(methodName=name,
                                         sigmadict_test=sigmadict_test,
                                         answer=answer))
        return suite

class Test_sigma_best(ParametrizedTestCase):

    def runTest(self):
        pass

    def setUp(self):
        self.psr = Pulsar(name="pulsar",
                          sigmas = {"1": {"sigma_tot": self.sigmadict_test[0]},
                                    "2": {"sigma_tot": self.sigmadict_test[1]},
                                    "3": {"sigma_tot": self.sigmadict_test[2]}})
        self.pta = PTA(psrlist=[self.psr])

    def test_sigma_best_picks_smallest_positive_sigma(self):
        print("sigmadict_test = {}".format(self.psr.sigmas))
        best_instr = self.pta.sigma_best()[1]
        self.assertEqual(best_instr, self.answer)
         
if __name__ == '__main__':
    suite = unittest.TestSuite()
    sigmadict_test_params = [[(.1, .2, .3), "1"],
                             [(.1, .2, -2), "1"],
                             [(.2, .3, .1), "3"]]
    for d, a in sigmadict_test_params:
        suite.addTest(ParametrizedTestCase.parametrize(Test_sigma_best,
                                                       sigmadict_test=d,
                                                       answer=a))

    unittest.TextTestRunner(verbosity=2).run(suite)
