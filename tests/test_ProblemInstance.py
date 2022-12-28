'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemReader.ProblemReader import loadProblemInstance
from ProblemInstance.ProblemInstance import ProblemInstance

class TestProblemReader(unittest.TestCase):
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(np.array([[1, 0, 0],
                                                            [0, 1, 0],
                                                            [0, 0, 1]])),
                                                            4.79)

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(np.array([[1, 1, 0],
                                                            [1, 1, 1],
                                                            [0, 0, 1]])),
                                                            10.2185)

    def test3_data(self):
        pi = loadProblemInstance('tests/test3_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(np.array([[1, 1],
                                                            [1, 1]])),
                                                            6.59)

    def test_iters(self):
        iters = 1000
        m = 100
        n = 100
        P = np.random.rand(m, n)
        Q = np.random.rand(m, n)
        R = np.random.rand(m, n) * 10
        pi = ProblemInstance(P, Q, R)

        for _ in range(iters):
            Y = np.random.randint(0, 2, size=(m, n))
            np.testing.assert_allclose(pi.expectedRevenue(Y), pi.expectedRevenueSlow(Y))
            

if __name__ == '__main__':
    unittest.main()
